from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.inference_wind_pipeline import (
    _diffusion_indices,
    _regression_batch,
    _selected_indices,
)


class _Dataset:
    def __init__(self, times: pd.DatetimeIndex) -> None:
        self._times = times

    def time(self) -> pd.DatetimeIndex:
        return self._times


def test_diffusion_monthly_selection_does_not_change_regression_legacy_indices():
    dataset = _Dataset(pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h"))
    cfg = OmegaConf.create(
        {
            "initial_hour": 0,
            "initial_time_start": "2024-01-02T00:00:00",
            "initial_time_end": "2024-03-29T00:00:00",
            "max_initial_times": 2,
            "diffusion_samples_per_month": 3,
            "diffusion_sample_seed": 42,
        }
    )

    regression_indices = _selected_indices(dataset, cfg)
    diffusion_indices = _diffusion_indices(dataset, cfg)
    regression_times = dataset.time()[regression_indices]
    diffusion_times = dataset.time()[diffusion_indices]

    assert len(regression_indices) == 2
    assert regression_times.tolist() == [
        pd.Timestamp("2024-01-02 00:00"),
        pd.Timestamp("2024-01-03 00:00"),
    ]
    assert len(diffusion_indices) == 9
    assert diffusion_times.is_monotonic_increasing
    assert diffusion_times.month.value_counts().to_dict() == {1: 3, 2: 3, 3: 3}


def test_diffusion_monthly_selection_disabled_uses_legacy_indices():
    dataset = _Dataset(pd.date_range("2024-01-01", "2024-01-10 23:00", freq="h"))
    cfg = OmegaConf.create(
        {
            "initial_hour": 0,
            "initial_time_start": "2024-01-02T00:00:00",
            "initial_time_end": "2024-01-08T00:00:00",
            "max_initial_times": 3,
            "diffusion_samples_per_month": 0,
            "diffusion_sample_seed": 42,
        }
    )

    assert _diffusion_indices(dataset, cfg) == _selected_indices(dataset, cfg)


def test_regression_inference_passes_detected_initial_time_features():
    class Dataset:
        output_variables = ["u10", "v10"]
        forecast_horizon_hours = 2

        @staticmethod
        def condition(index):
            return np.full((18, 3, 5), float(index), dtype=np.float32)

        @staticmethod
        def image_shape():
            return (3, 5)

        @staticmethod
        def initial_time_features(index):
            values = np.arange(6, dtype=np.float32)
            return values + float(index)

    class Model(torch.nn.Module):
        augment_dim = 6

        def __init__(self):
            super().__init__()
            self.labels = None

        def forward(self, zeros, condition, augment_labels=None):
            self.labels = augment_labels
            return zeros

    model = Model()
    output = _regression_batch(
        model, Dataset(), [2, 4], torch.device("cpu")
    )
    assert output.shape == (2, 4, 3, 5)
    expected = torch.from_numpy(
        np.stack(
            [
                np.arange(6, dtype=np.float32) + 2.0,
                np.arange(6, dtype=np.float32) + 4.0,
            ]
        )
    )
    torch.testing.assert_close(model.labels, expected)
