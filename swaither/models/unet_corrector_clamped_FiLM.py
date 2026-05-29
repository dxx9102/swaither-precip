import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo import Module
from physicsnemo.models.diffusion import UNet

# If UNetResidualCorrectorClamped3 is in another file, adjust this import:
from .unet_corrector_clamped_3 import UNetResidualCorrectorClamped3

def _pad_to_multiple(x: torch.Tensor | None, mult: int):
    if x is None:
        return None, None
    B, C, H, W = x.shape
    ph = (-H) % mult
    pw = (-W) % mult
    if ph == 0 and pw == 0:
        return x, (H, W)
    return F.pad(x, (0, pw, 0, ph)), (H, W)


def _crop_to(x: torch.Tensor, hw: tuple[int, int]):
    H, W = hw
    return x[..., :H, :W]


def _to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return type(o)(_to_py(v) for v in o)
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    return o


class UNetResidualCorrectorClampedFiLM(Module):
    """
    Wrapper around UNetResidualCorrectorClamped3 with FiLM-style lead-time conditioning.

    - Builds an internal UNetResidualCorrectorClamped3 ("base" model).
    - Does *not* change the number of channels.
    - Only modulates the first `n_film_channels` of img_lr as:

         x_phys_cond = x_phys * (1 + gamma(lead)) + beta(lead)

      where gamma,beta are produced by a small MLP from a lead-time embedding.

    Expected call signature (as in CorrDiff regression):

        y_hat = model(x0, img_lr, lead_time_label=lead, augment_pipe=..., ...)

    where:
        x0  : (B, 1, H, W)  zero-latent (ignored by FiLM, passed through)
        img_lr : (B, C_total, H, W)
        lead_time_label : (B,) or (B,1) integer lead indices [0..n_leads-1]
    """

    __model_checkpoint_version__ = 1

    def __init__(
        self,
        # --- arguments used to build the *base* residual model ---
        base_model_kwargs: dict,
        # total number of input channels img_lr will have at runtime
        total_in_channels: int,
        # how many of those channels we FiLM-modulate (the "physical" channels)
        n_film_channels: int,
        # --- FiLM hyperparameters ---
        n_leads: int = 24,
        emb_dim: int = 32,
        mlp_hidden: int = 128,
    ):
        """
        Parameters
        ----------
        base_model_kwargs : dict
            Keyword args for UNetResidualCorrectorClamped3, e.g.:

                {
                    "img_in_channels": 16,
                    "img_out_channels": 1,
                    "unet_kwargs": {...},
                    "clamp_min": 0.0,
                    "clamp_max": 20.0,
                }

        total_in_channels : int
            Total number of channels in img_lr (physical + optional grid channels).
            Must match base_model_kwargs["img_in_channels"].

        n_film_channels : int
            Number of channels (from the front) to FiLM-modulate. Typically
            = "physical" channels (AIFS vars + altitude), leaving grid channels
            (if any) untouched.

        n_leads : int
            Number of discrete lead indices (0..n_leads-1).

        emb_dim : int
            Dimension of the lead-time embedding.

        mlp_hidden : int
            Hidden size of the FiLM MLP.
        """
        super().__init__()

        # Build the underlying residual corrector
        self.base_model = UNetResidualCorrectorClamped3(**base_model_kwargs)

        self.total_in_channels = int(total_in_channels)
        self.n_film_channels   = int(n_film_channels)
        self.n_leads           = int(n_leads)
        self.emb_dim           = int(emb_dim)
        self.mlp_hidden        = int(mlp_hidden)

        if self.n_film_channels > self.total_in_channels:
            raise ValueError(
                f"n_film_channels={self.n_film_channels} > total_in_channels={self.total_in_channels}"
            )

        # simple integer lead embedding
        self.lead_emb = nn.Embedding(self.n_leads, self.emb_dim)

        # FiLM MLP: emb_dim → 2 * n_film_channels (gamma & beta)
        self.film_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, 2 * self.n_film_channels),
        )

        # expose same meta & img_out_channels as base model
        if getattr(self, "meta", None) is None:
            self.meta = types.SimpleNamespace(name="UNetResidualCorrectorClampedFiLM")
        self.img_out_channels = self.base_model.img_out_channels

        # for PhysicsNeMo checkpointing
        self._args = {
            "__name__":   self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "__args__": {
                "base_model_kwargs": _to_py(base_model_kwargs),
                "total_in_channels": int(self.total_in_channels),
                "n_film_channels":   int(self.n_film_channels),
                "n_leads":           int(self.n_leads),
                "emb_dim":           int(self.emb_dim),
                "mlp_hidden":        int(self.mlp_hidden),
            },
        }

    # ------------- forward -------------

    def forward(self, x0, img_lr, lead_time_label=None, augment_pipe=None, **kwargs):
        """
        x0  : (B, 1, H, W)
        img_lr : (B, C_total, H, W), C_total = total_in_channels
        lead_time_label : (B,) or (B,1) long
        """
        if lead_time_label is None:
            raise RuntimeError(
                "UNetResidualCorrectorClampedFiLM expects lead_time_label for FiLM conditioning"
            )

        B, C, H, W = img_lr.shape
        if C != self.total_in_channels:
            raise AssertionError(
                f"img_lr has {C} channels, but total_in_channels={self.total_in_channels}"
            )

        # ensure shape (B,)
        if lead_time_label.dim() == 2 and lead_time_label.size(1) == 1:
            lead_time_label = lead_time_label[:, 0]

        # clip or wrap lead indices to [0, n_leads-1] to be safe
        lead_time_label = lead_time_label.clamp(0, self.n_leads - 1)

        # split: [physical (FiLM-ed) | rest]
        x_phys = img_lr[:, : self.n_film_channels, :, :]   # (B, n_film, H, W)
        x_rest = img_lr[:, self.n_film_channels :, :, :]   # (B, C - n_film, H, W)

        # --- FiLM parameters from lead index ---
        z = self.lead_emb(lead_time_label)  # (B, emb_dim)
        film = self.film_mlp(z)             # (B, 2 * n_film)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.view(B, self.n_film_channels, 1, 1)
        beta  = beta.view(B, self.n_film_channels, 1, 1)

        # apply FiLM
        x_phys_cond = x_phys * (1.0 + gamma) + beta  # (B, n_film, H, W)

        # re-concatenate with untouched channels (e.g. grid)
        img_lr_cond = torch.cat([x_phys_cond, x_rest], dim=1)  # (B, C_total, H, W)

        # call underlying residual corrector
        # NOTE: we *do not* pass lead_time_label further – base model stays "blind" to lead.
        y_hat = self.base_model(
            x0,
            img_lr_cond,
            augment_pipe=augment_pipe,
            **kwargs,
        )
        return y_hat

