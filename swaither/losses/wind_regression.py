"""Configurable composite losses for wind-field regression."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn


REGRESSION_LOSS_PARAMETERS = {
    "normalized_mse": frozenset(),
    "physical_vector_mse": frozenset(),
    "physical_vector_charbonnier": frozenset({"epsilon"}),
    "physical_speed_mae": frozenset(),
    "direction_cosine": frozenset({"min_speed"}),
}
SUPPORTED_REGRESSION_LOSSES = tuple(REGRESSION_LOSS_PARAMETERS)


@dataclass(frozen=True)
class RegressionLossTerm:
    """One weighted loss term parsed from the Hydra configuration."""

    name: str
    kind: str
    weight: float
    epsilon: float = 1.0
    min_speed: float = 2.0


@dataclass
class RegressionLossResult:
    """Composite scalar plus raw and weighted component tensors for logging."""

    total: torch.Tensor
    raw: Dict[str, torch.Tensor] = field(default_factory=dict)
    weighted: Dict[str, torch.Tensor] = field(default_factory=dict)


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return parsed


def _parse_term(
    name: str,
    config: Mapping[str, Any],
) -> RegressionLossTerm:
    if not name or not isinstance(name, str):
        raise ValueError("Each regression loss term must have a non-empty name")
    if not isinstance(config, Mapping):
        raise ValueError(f"Regression loss term {name!r} must be a mapping")

    kind = str(config.get("type", ""))
    if kind not in SUPPORTED_REGRESSION_LOSSES:
        supported = ", ".join(SUPPORTED_REGRESSION_LOSSES)
        raise ValueError(
            f"Unknown regression loss type {kind!r} for term {name!r}; "
            f"expected one of: {supported}"
        )
    weight = _finite_float(config.get("weight", 1.0), f"{name}.weight")
    if weight < 0.0:
        raise ValueError(f"{name}.weight must be non-negative, got {weight}")

    allowed_keys = {"type", "weight"} | REGRESSION_LOSS_PARAMETERS[kind]
    unexpected = sorted(set(config) - allowed_keys)
    if unexpected:
        raise ValueError(
            f"Unsupported keys for regression loss term {name!r}: {unexpected}"
        )

    epsilon = _finite_float(config.get("epsilon", 1.0), f"{name}.epsilon")
    min_speed = _finite_float(config.get("min_speed", 2.0), f"{name}.min_speed")
    if kind == "physical_vector_charbonnier" and epsilon <= 0.0:
        raise ValueError(f"{name}.epsilon must be positive, got {epsilon}")
    if kind == "direction_cosine" and min_speed < 0.0:
        raise ValueError(f"{name}.min_speed must be non-negative, got {min_speed}")

    return RegressionLossTerm(
        name=name,
        kind=kind,
        weight=weight,
        epsilon=epsilon,
        min_speed=min_speed,
    )


class WindRegressionLoss(nn.Module):
    """Weighted sum of normalized and physical wind-field regression losses.

    Predictions use the dataset's lead-major channel order
    ``[h1_u, h1_v, ..., hT_u, hT_v]``. Physical losses convert the normalized
    residual back to m/s before combining the two components.
    """

    def __init__(
        self,
        terms: Sequence[RegressionLossTerm],
        *,
        output_min: Sequence[float] | torch.Tensor,
        output_max: Sequence[float] | torch.Tensor,
        forecast_hours: int,
        wind_components: int = 2,
    ) -> None:
        super().__init__()
        if not terms:
            raise ValueError("At least one regression loss term is required")
        if int(wind_components) != 2:
            raise ValueError("WindRegressionLoss requires exactly two wind components")
        if int(forecast_hours) < 1:
            raise ValueError("forecast_hours must be positive")
        if not any(term.weight > 0.0 for term in terms):
            raise ValueError("At least one regression loss term must have weight > 0")

        minimum = torch.as_tensor(output_min, dtype=torch.float32).reshape(-1)
        maximum = torch.as_tensor(output_max, dtype=torch.float32).reshape(-1)
        if minimum.numel() != 2 or maximum.numel() != 2:
            raise ValueError(
                "output_min and output_max must each contain u and v statistics"
            )
        if not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all():
            raise ValueError("Wind output ranges must be finite")
        output_range = maximum - minimum
        if torch.any(output_range <= 0.0):
            raise ValueError("Wind output ranges must be positive")

        names = [term.name for term in terms]
        if len(names) != len(set(names)):
            raise ValueError("Regression loss term names must be unique")

        self.terms = tuple(terms)
        self.forecast_hours = int(forecast_hours)
        self.wind_components = int(wind_components)
        self.register_buffer("output_min", minimum)
        self.register_buffer("output_max", maximum)

    @property
    def output_range(self) -> torch.Tensor:
        return self.output_max - self.output_min

    @property
    def vector_squared_scale(self) -> torch.Tensor:
        return self.output_range.square().mean()

    def _pairs(self, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_channels = self.forecast_hours * self.wind_components
        if prediction.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} forecast channels, "
                f"got {prediction.shape[1]}"
            )
        batch, _, height, width = prediction.shape
        return prediction.reshape(
            batch,
            self.forecast_hours,
            self.wind_components,
            height,
            width,
        ).unbind(dim=2)

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> RegressionLossResult:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must have identical shapes; "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        if prediction.ndim != 4:
            raise ValueError(
                "prediction and target must be 4-D tensors in (B, C, H, W) order"
            )
        expected_channels = self.forecast_hours * self.wind_components
        if prediction.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} forecast channels, "
                f"got {prediction.shape[1]}"
            )

        prediction_fp32 = prediction.float()
        target_fp32 = target.float()
        active = [term for term in self.terms if term.weight > 0.0]
        needs_physical = any(term.kind != "normalized_mse" for term in active)

        normalized_mse: torch.Tensor | None = None
        du: torch.Tensor | None = None
        dv: torch.Tensor | None = None
        true_u: torch.Tensor | None = None
        true_v: torch.Tensor | None = None
        if needs_physical:
            prediction_u, prediction_v = self._pairs(prediction_fp32)
            true_u, true_v = self._pairs(target_fp32)
            output_range = self.output_range.to(
                device=prediction_fp32.device, dtype=prediction_fp32.dtype
            )
            du = (prediction_u - true_u) * output_range[0]
            dv = (prediction_v - true_v) * output_range[1]
            output_min_device = self.output_min.to(
                device=prediction_fp32.device, dtype=prediction_fp32.dtype
            )
            prediction_u_physical = (
                prediction_u * output_range[0] + output_min_device[0]
            )
            prediction_v_physical = (
                prediction_v * output_range[1] + output_min_device[1]
            )
            true_u_physical = true_u * output_range[0] + output_min_device[0]
            true_v_physical = true_v * output_range[1] + output_min_device[1]

        raw: Dict[str, torch.Tensor] = {}
        weighted: Dict[str, torch.Tensor] = {}
        for term in active:
            if term.kind == "normalized_mse":
                if normalized_mse is None:
                    normalized_mse = (prediction_fp32 - target_fp32).square().mean()
                value = normalized_mse
            else:
                assert du is not None and dv is not None
                assert true_u is not None and true_v is not None
                vector_squared_scale = self.vector_squared_scale.to(
                    device=du.device, dtype=du.dtype
                )
                if term.kind == "physical_vector_mse":
                    value = (du.square() + dv.square()).mean() / vector_squared_scale
                elif term.kind == "physical_vector_charbonnier":
                    epsilon = du.new_tensor(float(term.epsilon))
                    distance = torch.sqrt(du.square() + dv.square() + epsilon.square())
                    value = (distance - epsilon).mean() / torch.sqrt(
                        vector_squared_scale
                    )
                elif term.kind == "physical_speed_mae":
                    prediction_speed = torch.hypot(
                        prediction_u_physical, prediction_v_physical
                    )
                    true_speed = torch.hypot(true_u_physical, true_v_physical)
                    value = (prediction_speed - true_speed).abs().mean() / torch.sqrt(
                        vector_squared_scale
                    )
                elif term.kind == "direction_cosine":
                    prediction_speed = torch.hypot(
                        prediction_u_physical, prediction_v_physical
                    )
                    true_speed = torch.hypot(true_u_physical, true_v_physical)
                    dot = (
                        prediction_u_physical * true_u_physical
                        + prediction_v_physical * true_v_physical
                    )
                    denominator = (prediction_speed * true_speed).clamp_min(
                        torch.finfo(du.dtype).eps
                    )
                    cosine = (dot / denominator).clamp(-1.0, 1.0)
                    valid = true_speed >= float(term.min_speed)
                    value = ((1.0 - cosine) * valid).sum() / valid.sum().clamp_min(1)
                else:  # Defensive: parsing above rejects unsupported names.
                    raise RuntimeError(f"Unhandled regression loss type: {term.kind}")
            raw[term.name] = value
            weighted[term.name] = value * float(term.weight)

        if len(weighted) == 1:
            total = next(iter(weighted.values()))
        else:
            total = torch.stack(tuple(weighted.values())).sum()
        return RegressionLossResult(total=total, raw=raw, weighted=weighted)


def build_wind_regression_loss(
    config: Mapping[str, Any],
    *,
    output_min: Sequence[float] | torch.Tensor,
    output_max: Sequence[float] | torch.Tensor,
    forecast_hours: int,
    wind_components: int = 2,
) -> WindRegressionLoss:
    """Build a composite loss from ``training.loss`` configuration."""

    if not isinstance(config, Mapping):
        raise ValueError("training.loss must be a mapping")
    unexpected = sorted(set(config) - {"terms"})
    if unexpected:
        raise ValueError(f"Unsupported training.loss keys: {unexpected}")
    term_configs = config.get("terms")
    if not isinstance(term_configs, Mapping) or not term_configs:
        raise ValueError("training.loss.terms must be a non-empty mapping")
    terms = [
        _parse_term(str(name), term_config)
        for name, term_config in term_configs.items()
    ]
    return WindRegressionLoss(
        terms,
        output_min=output_min,
        output_max=output_max,
        forecast_hours=forecast_hours,
        wind_components=wind_components,
    )
