"""Deterministic U-Net for the low-resolution hourly wind forecast."""

from __future__ import annotations

import types
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo import Module
from physicsnemo.models.diffusion import EDMPrecondSuperResolution, UNet
import physicsnemo.models.diffusion as network_module
from physicsnemo.models.diffusion.song_unet import SongUNet


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_python(item) for item in value)
    return value


class WindForecastUNet(Module):
    """PhysicsNeMo U-Net wrapper that pads arbitrary LR grids before downsampling.

    The caller supplies a zero latent ``x0`` with two channels, matching the
    CorrDiff regression convention.  No physical clipping is applied because
    wind components are signed.
    """

    __model_checkpoint_version__ = 1

    def __init__(
        self,
        img_in_channels: int,
        img_out_channels: int = 2,
        unet_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if int(img_out_channels) != 2:
            raise ValueError("WindForecastUNet requires exactly two output channels")
        self.img_in_channels = int(img_in_channels)
        self.img_out_channels = int(img_out_channels)
        kwargs = dict(unet_kwargs or {})
        kwargs.setdefault("img_in_channels", self.img_in_channels)
        kwargs.setdefault("img_out_channels", self.img_out_channels)
        kwargs.setdefault("model_type", "SongUNet")
        self.unet = UNet(**kwargs)
        channel_mult = list(kwargs.get("channel_mult", [1, 2, 2]))
        self.pad_multiple = 2 ** len(channel_mult)

        if getattr(self, "meta", None) is None:
            self.meta = types.SimpleNamespace(name="WindForecastUNet")
        self._args = {
            "__name__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "__args__": {
                "img_in_channels": self.img_in_channels,
                "img_out_channels": self.img_out_channels,
                "unet_kwargs": _to_python(kwargs),
            },
        }

    def forward(
        self,
        x0: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe: Any = None,
        **unused: Any,
    ) -> torch.Tensor:
        del augment_pipe, unused
        if img_lr.shape[1] != self.img_in_channels:
            raise ValueError(
                f"Expected {self.img_in_channels} input channels, got {img_lr.shape[1]}"
            )
        if x0.shape[1] != self.img_out_channels:
            raise ValueError(
                f"Expected {self.img_out_channels} latent channels, got {x0.shape[1]}"
            )
        height, width = img_lr.shape[-2:]
        pad_h = (-height) % self.pad_multiple
        pad_w = (-width) % self.pad_multiple
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            img_lr = F.pad(img_lr, padding, mode="replicate")
            x0 = F.pad(x0, padding, mode="replicate")
        prediction = self.unet(x0, img_lr, force_fp32=False)
        return prediction[..., :height, :width]


class WindSongUNet(SongUNet):
    """Song U-Net that ignores CorrDiff patch-selection keywords.

    Wind coordinates are explicit data channels, so no internal positional grid
    is appended. The residual diffusion loss still supplies the generic
    ``embedding_selector`` and ``global_index`` keywords.
    """

    def forward(
        self,
        x,
        noise_labels,
        class_labels,
        global_index=None,
        embedding_selector=None,
        augment_labels=None,
        lead_time_label=None,
    ):
        del global_index, embedding_selector, lead_time_label
        return super().forward(x, noise_labels, class_labels, augment_labels)


class WindCorrDiffUNet(UNet):
    """CorrDiff regression U-Net with explicit coordinate channels."""

    def __init__(
        self,
        img_resolution,
        img_in_channels,
        img_out_channels,
        use_fp16: bool = False,
        **model_kwargs,
    ) -> None:
        model_kwargs.pop("gridtype", None)
        model_kwargs.pop("N_grid_channels", None)
        model_kwargs.pop("model_grid_channels", None)
        model_kwargs.pop("model_type", None)
        super().__init__(
            img_resolution=img_resolution,
            img_in_channels=img_in_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type="SongUNet",
            **model_kwargs,
        )

    def forward(self, x, img_lr, force_fp32: bool = False, **model_kwargs) -> torch.Tensor:
        model_kwargs.pop("embedding_selector", None)
        model_kwargs.pop("global_index", None)
        return super().forward(x, img_lr, force_fp32=force_fp32, **model_kwargs)


# EDMPrecondSuperResolution resolves this name from PhysicsNeMo's network
# module. Registration is local to the wind model import and keeps checkpoints
# loadable by their stable class path.
setattr(network_module, "WindSongUNet", WindSongUNet)


class WindEDMPrecondSuperResolution(EDMPrecondSuperResolution):
    """CorrDiff EDM preconditioner backed by :class:`WindSongUNet`."""

    _wrapped_classes = {"WindSongUNet"}

    def __init__(
        self,
        img_resolution,
        img_in_channels,
        img_out_channels,
        use_fp16: bool = False,
        model_type: str = "WindSongUNet",
        sigma_data: float = 0.5,
        sigma_min=0.0,
        sigma_max=float("inf"),
        **model_kwargs,
    ) -> None:
        model_kwargs.pop("gridtype", None)
        model_kwargs.pop("N_grid_channels", None)
        model_kwargs.pop("model_grid_channels", None)
        super().__init__(
            img_resolution=img_resolution,
            img_in_channels=img_in_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            model_type="WindSongUNet",
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            **model_kwargs,
        )


__all__ = [
    "WindForecastUNet",
    "WindCorrDiffUNet",
    "WindEDMPrecondSuperResolution",
]
