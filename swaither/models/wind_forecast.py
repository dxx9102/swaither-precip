"""Regression U-Net and residual EDM model for the wind pipeline."""

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
from physicsnemo.utils.patching import RandomPatching2D


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


def _validate_augment_labels(
    augment_labels: torch.Tensor | None,
    augment_dim: int,
    batch_size: int,
    model_name: str,
) -> torch.Tensor | None:
    expected = int(augment_dim)
    if expected == 0:
        if augment_labels is not None:
            raise ValueError(f"{model_name} was not configured for augment labels")
        return None
    if augment_labels is None:
        raise ValueError(f"{model_name} requires augment_labels with {expected} features")
    if augment_labels.ndim != 2 or tuple(augment_labels.shape) != (
        batch_size,
        expected,
    ):
        raise ValueError(
            f"Expected augment_labels {(batch_size, expected)}, "
            f"got {tuple(augment_labels.shape)}"
        )
    return augment_labels


class WindForecastUNet(Module):
    """One-shot 72-hour high-resolution absolute wind regression.

    The caller supplies a zero latent with ``forecast_hours * 2`` channels and
    a condition tensor containing six history times, HR static fields and
    coordinates.  Outputs use lead-major channel order
    ``[h1_u, h1_v, ..., h72_u, h72_v]``.
    """

    __model_checkpoint_version__ = 4
    __supported_model_checkpoint_version__ = {
        3: "Loading WindForecastUNet checkpoint version 3 without time conditioning."
    }

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: int, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if int(version) == 3:
            unet_kwargs = args.setdefault("unet_kwargs", {})
            unet_kwargs.setdefault("augment_dim", 0)
        return args

    def __init__(
        self,
        img_in_channels: int,
        forecast_hours: int = 72,
        wind_components: int = 2,
        unet_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if int(forecast_hours) < 1:
            raise ValueError("forecast_hours must be >= 1")
        if int(wind_components) < 1:
            raise ValueError("wind_components must be >= 1")
        self.img_in_channels = int(img_in_channels)
        self.forecast_hours = int(forecast_hours)
        self.wind_components = int(wind_components)
        self.img_out_channels = self.forecast_hours * self.wind_components
        kwargs = dict(unet_kwargs or {})
        kwargs.setdefault("img_in_channels", self.img_in_channels)
        kwargs.setdefault("img_out_channels", self.img_out_channels)
        kwargs.setdefault("model_type", "SongUNet")
        self.augment_dim = int(kwargs.get("augment_dim", 0))
        if self.augment_dim < 0:
            raise ValueError("augment_dim must be non-negative")
        self.unet = UNet(**kwargs)
        self.pad_multiple = 2 ** len(list(kwargs.get("channel_mult", [1, 2, 2])))

        if getattr(self, "meta", None) is None:
            self.meta = types.SimpleNamespace(name="WindForecastUNet")
        self._args = {
            "__name__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "__args__": {
                "img_in_channels": self.img_in_channels,
                "forecast_hours": self.forecast_hours,
                "wind_components": self.wind_components,
                "unet_kwargs": _to_python(kwargs),
            },
        }

    def forward(
        self,
        x0: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe: Any = None,
        augment_labels: torch.Tensor | None = None,
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
        if x0.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError("Latent and condition spatial shapes must match")
        augment_labels = _validate_augment_labels(
            augment_labels,
            self.augment_dim,
            img_lr.shape[0],
            "WindForecastUNet",
        )
        height, width = img_lr.shape[-2:]
        pad_h = (-height) % self.pad_multiple
        pad_w = (-width) % self.pad_multiple
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            img_lr = F.pad(img_lr, padding, mode="replicate")
            x0 = F.pad(x0, padding, mode="replicate")
        prediction = self.unet(
            x0,
            img_lr,
            force_fp32=False,
            augment_labels=augment_labels,
        )
        return prediction[..., :height, :width]


class WindSongUNet(SongUNet):
    """Song U-Net that accepts PhysicsNeMo residual-loss keywords."""

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


setattr(network_module, "WindSongUNet", WindSongUNet)


class WindEDMPrecondSuperResolution(EDMPrecondSuperResolution):
    """Two-channel EDM preconditioner backed by :class:`WindSongUNet`."""

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
        channel_mult = list(model_kwargs.get("channel_mult", [1, 2, 2, 2]))
        self.pad_multiple = 2 ** len(channel_mult)
        self.augment_dim = int(model_kwargs.get("augment_dim", 0))
        if self.augment_dim < 0:
            raise ValueError("augment_dim must be non-negative")
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

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        if x.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError("Latent and condition spatial shapes must match")
        augment_labels = model_kwargs.get("augment_labels")
        augment_labels = _validate_augment_labels(
            augment_labels,
            self.augment_dim,
            img_lr.shape[0],
            "WindEDMPrecondSuperResolution",
        )
        if augment_labels is not None:
            model_kwargs["augment_labels"] = augment_labels
        height, width = img_lr.shape[-2:]
        pad_h = (-height) % self.pad_multiple
        pad_w = (-width) % self.pad_multiple
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            x = F.pad(x, padding, mode="replicate")
            img_lr = F.pad(img_lr, padding, mode="replicate")
        prediction = super().forward(
            x,
            img_lr,
            sigma,
            force_fp32=force_fp32,
            **model_kwargs,
        )
        return prediction[..., :height, :width]


class WindRegressionLeadAdapter(torch.nn.Module):
    """Expose one two-channel lead from the frozen 72-hour regression U-Net.

    PhysicsNeMo's ``ResidualLoss`` expects a mean prediction with the same
    shape as a single diffusion target.  The diffusion condition has two extra
    lead-encoding channels, so this adapter strips those channels before
    invoking the frozen regression network and selects ``2 * (lead - 1)``.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        forecast_hours: int = 72,
        wind_components: int = 2,
        conditioning_input_channels: int = 18,
        lead_encoding_channels: int = 2,
    ) -> None:
        super().__init__()
        self.regression_net = regression_net.eval().requires_grad_(False)
        self.regression_augment_dim = int(
            getattr(self.regression_net, "augment_dim", 0)
        )
        self.forecast_hours = int(forecast_hours)
        self.wind_components = int(wind_components)
        self.conditioning_input_channels = int(conditioning_input_channels)
        self.lead_encoding_channels = int(lead_encoding_channels)

    def forward(
        self,
        x0: torch.Tensor,
        img_lr: torch.Tensor,
        lead_time_label: torch.Tensor | None = None,
        augment_labels: torch.Tensor | None = None,
        **unused: Any,
    ) -> torch.Tensor:
        del unused
        if lead_time_label is None:
            raise ValueError("WindRegressionLeadAdapter requires lead_time_label")
        lead = lead_time_label.reshape(-1).to(device=img_lr.device, dtype=torch.long)
        if torch.any(lead < 1) or torch.any(lead > self.forecast_hours):
            raise ValueError("lead_time_label must be within the forecast horizon")
        expected_channels = self.conditioning_input_channels + self.lead_encoding_channels
        if img_lr.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} diffusion condition channels, got {img_lr.shape[1]}"
            )
        regression_condition = img_lr[:, : self.conditioning_input_channels]
        regression_augment_labels = None
        if self.regression_augment_dim:
            if (
                augment_labels is None
                or augment_labels.ndim != 2
                or augment_labels.shape[1] < self.regression_augment_dim
            ):
                raise ValueError(
                    "Diffusion augment_labels must include the frozen regression "
                    f"time features ({self.regression_augment_dim})"
                )
            labels = augment_labels[:, : self.regression_augment_dim]
            # ResidualLoss invokes the regression network once on full-domain
            # inputs before patchifying the diffusion inputs. Multi-patch
            # labels are P-major, so take the first copy of every sample.
            patch_num, remainder = divmod(labels.shape[0], img_lr.shape[0])
            if remainder or patch_num < 1:
                raise ValueError(
                    "Diffusion augment_labels batch must be a whole multiple "
                    "of the regression input batch"
                )
            regression_augment_labels = labels.reshape(
                patch_num, img_lr.shape[0], self.regression_augment_dim
            )[0]
        zero_latent = torch.zeros(
            (
                img_lr.shape[0],
                self.forecast_hours * self.wind_components,
                *img_lr.shape[-2:],
            ),
            dtype=img_lr.dtype,
            device=img_lr.device,
        )
        with torch.no_grad():
            if regression_augment_labels is None:
                forecast = self.regression_net(zero_latent, regression_condition)
            else:
                forecast = self.regression_net(
                    zero_latent,
                    regression_condition,
                    augment_labels=regression_augment_labels,
                )
        selected = []
        for item, lead_value in zip(forecast, lead.tolist()):
            start = (int(lead_value) - 1) * self.wind_components
            selected.append(item[start : start + self.wind_components])
        return torch.stack(selected, dim=0)


class WindTimeConditionedModel(torch.nn.Module):
    """Inject fixed global time labels into a sampler's repeated model calls."""

    def __init__(
        self,
        model: torch.nn.Module,
        time_labels: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        labels = torch.as_tensor(time_labels)
        if labels.ndim != 2 or labels.shape[0] != 1:
            raise ValueError("time_labels must have shape (1, features)")
        self.register_buffer("time_labels", labels.detach().clone())
        self.sigma_min = model.sigma_min
        self.sigma_max = model.sigma_max

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        del class_labels
        labels = self.time_labels.to(device=x.device, dtype=x.dtype)
        if labels.shape[0] == 1:
            labels = labels.expand(x.shape[0], -1)
        elif labels.shape[0] != x.shape[0]:
            raise ValueError(
                "time_labels batch does not match the conditioned model batch"
            )
        return self.model(
            x,
            img_lr,
            sigma,
            augment_labels=labels,
            **model_kwargs,
        )

    def round_sigma(self, sigma: Any) -> torch.Tensor:
        return self.model.round_sigma(sigma)


class WindRandomPatching2D(RandomPatching2D):
    """Random crops that preserve the fixed 22-channel diffusion condition.

    ``ResidualLoss`` normally concatenates an additional full-domain view when
    patching is enabled.  Wind conditioning already contains the HR U-Net mean,
    history, terrain, coordinates and lead encoding, so the approved model
    contract stays at 22 channels and only the spatial crop is retained.
    """

    def apply(
        self,
        input: torch.Tensor,
        additional_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del additional_input
        return super().apply(input)


__all__ = [
    "WindForecastUNet",
    "WindEDMPrecondSuperResolution",
    "WindRegressionLeadAdapter",
    "WindRandomPatching2D",
    "WindTimeConditionedModel",
]
