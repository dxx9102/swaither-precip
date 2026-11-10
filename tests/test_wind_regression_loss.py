from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from swaither.losses.wind_regression import build_wind_regression_loss


REPO = Path(__file__).resolve().parents[1]


def _build(terms: dict):
    return build_wind_regression_loss(
        {"terms": terms},
        output_min=[0.0, 0.0],
        output_max=[10.0, 10.0],
        forecast_hours=1,
    )


def _physical_prediction() -> torch.Tensor:
    prediction = torch.zeros((1, 2, 1, 1), requires_grad=True)
    with torch.no_grad():
        prediction[..., 0, 0] = torch.tensor((0.3, 0.4))
    return prediction


def test_normalized_mse_matches_existing_formula():
    loss_fn = _build(
        {
            "mse": {
                "type": "normalized_mse",
                "weight": 1.0,
            }
        }
    )
    target = torch.zeros((1, 2, 1, 1))
    prediction = _physical_prediction()
    result = loss_fn(prediction, target)

    torch.testing.assert_close(result.total, torch.tensor(0.125))
    torch.testing.assert_close(result.raw["mse"], torch.tensor(0.125))


def test_physical_losses_use_lead_major_u_v_pairs():
    prediction = _physical_prediction()
    target = torch.zeros((1, 2, 1, 1))
    cases = {
        "vector": ("physical_vector_mse", {}, 0.25),
        "charbonnier": (
            "physical_vector_charbonnier",
            {"epsilon": 1.0},
            (torch.sqrt(torch.tensor(26.0)) - 1.0).item() / 10.0,
        ),
        "speed": ("physical_speed_mae", {}, 0.5),
    }
    for name, (kind, parameters, expected) in cases.items():
        result = _build({name: {"type": kind, "weight": 1.0, **parameters}})(
            prediction, target
        )
        torch.testing.assert_close(
            result.total, torch.tensor(expected), rtol=1e-6, atol=1e-6
        )


def test_speed_and_direction_include_output_offsets_when_denormalizing():
    prediction = torch.zeros((1, 2, 1, 1))
    target = torch.zeros_like(prediction)
    prediction[0, :, 0, 0] = torch.tensor((0.5, 0.55))
    target[0, :, 0, 0] = torch.tensor((0.5, 0.5))
    speed = build_wind_regression_loss(
        {"terms": {"speed": {"type": "physical_speed_mae", "weight": 1.0}}},
        output_min=[-10.0, -20.0],
        output_max=[10.0, 20.0],
        forecast_hours=1,
    )(prediction, target)
    torch.testing.assert_close(speed.total, torch.tensor(2.0 / math.sqrt(1000.0)))

    direction_prediction = torch.zeros_like(prediction)
    direction_target = torch.zeros_like(prediction)
    direction_prediction[0, :, 0, 0] = torch.tensor((0.5, 0.5))
    direction_target[0, :, 0, 0] = torch.tensor((0.5, 0.75))
    direction = build_wind_regression_loss(
        {
            "terms": {
                "direction": {
                    "type": "direction_cosine",
                    "weight": 1.0,
                    "min_speed": 2.0,
                }
            }
        },
        output_min=[-10.0, -20.0],
        output_max=[10.0, 20.0],
        forecast_hours=1,
    )(direction_prediction, direction_target)
    torch.testing.assert_close(direction.total, torch.tensor(1.0))


def test_direction_cosine_masks_calm_truth_and_handles_no_valid_cells():
    target = torch.zeros((1, 2, 1, 2))
    prediction = torch.zeros_like(target, requires_grad=True)
    with torch.no_grad():
        target[0, :, 0, 0] = torch.tensor((0.3, 0.4))
        target[0, :, 0, 1] = torch.tensor((0.06, 0.08))
        prediction[0, :, 0, 0] = torch.tensor((-0.4, 0.3))
        prediction[0, :, 0, 1] = torch.tensor((100.0, -100.0))

    loss_fn = _build(
        {
            "direction": {
                "type": "direction_cosine",
                "weight": 1.0,
                "min_speed": 2.0,
            }
        }
    )
    result = loss_fn(prediction, target)
    torch.testing.assert_close(result.total, torch.tensor(1.0), rtol=1e-6, atol=1e-6)

    no_valid = _build(
        {
            "direction": {
                "type": "direction_cosine",
                "weight": 1.0,
                "min_speed": 10.0,
            }
        }
    )(prediction, target)
    torch.testing.assert_close(no_valid.total, torch.tensor(0.0))
    no_valid.total.backward()
    assert torch.isfinite(prediction.grad).all()


def test_composite_weights_are_summed_without_renormalization():
    prediction = _physical_prediction()
    target = torch.zeros((1, 2, 1, 1))
    loss_fn = _build(
        {
            "mse": {"type": "normalized_mse", "weight": 2.0},
            "vector": {"type": "physical_vector_mse", "weight": 3.0},
            "disabled": {"type": "physical_speed_mae", "weight": 0.0},
        }
    )
    result = loss_fn(prediction, target)

    assert set(result.raw) == {"mse", "vector"}
    assert set(result.weighted) == {"mse", "vector"}
    torch.testing.assert_close(result.total, torch.tensor(1.0))
    result.total.backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize(
    ("terms", "message"),
    [
        ({"bad": {"type": "missing", "weight": 1.0}}, "Unknown regression loss type"),
        ({"bad": {"type": "normalized_mse", "weight": -1.0}}, "must be non-negative"),
        ({"bad": {"type": "normalized_mse", "weight": 0.0}}, "weight > 0"),
        (
            {"bad": {"type": "physical_vector_mse", "weight": 1.0, "epsilon": 1.0}},
            "Unsupported keys",
        ),
        (
            {
                "bad": {
                    "type": "physical_vector_charbonnier",
                    "weight": 1.0,
                    "epsilon": 0.0,
                }
            },
            "epsilon must be positive",
        ),
        (
            {"bad": {"type": "direction_cosine", "weight": 1.0, "min_speed": -1.0}},
            "min_speed must be non-negative",
        ),
    ],
)
def test_invalid_loss_configuration_is_rejected(terms, message):
    with pytest.raises(ValueError, match=message):
        _build(terms)


def test_prediction_channel_contract_is_validated():
    loss_fn = _build({"mse": {"type": "normalized_mse", "weight": 1.0}})
    with pytest.raises(ValueError, match="Expected 2 forecast channels"):
        loss_fn(torch.zeros((1, 4, 1, 1)), torch.zeros((1, 4, 1, 1)))


def test_default_config_enables_only_normalized_mse():
    cfg = OmegaConf.load(REPO / "conf" / "config_training_wind_regression.yaml")
    loss_cfg = OmegaConf.to_container(cfg.training.loss, resolve=True)
    loss_fn = build_wind_regression_loss(
        loss_cfg,
        output_min=[-10.0, -20.0],
        output_max=[10.0, 20.0],
        forecast_hours=1,
    )

    active_names = [term.name for term in loss_fn.terms if term.weight > 0.0]
    assert active_names == ["normalized_mse"]
    prediction = torch.full((1, 2, 1, 1), 0.25)
    result = loss_fn(prediction, torch.zeros_like(prediction))
    torch.testing.assert_close(result.total, torch.tensor(0.0625))
