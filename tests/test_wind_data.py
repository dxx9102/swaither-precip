from __future__ import annotations

from collections import Counter
import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

from swaither.datasets.wind import (
    FileTimeIndex,
    WindDiffusionDataset,
    WindForecastDataset,
    load_channel_stats,
    parse_hourly_times,
)
from swaither.wind_helpers import (
    EpochRandomSubsetSampler,
    resume_epoch_and_step,
    select_initial_indices,
    select_monthly_initial_indices,
    uniform_random_sample_indices,
)


def _grid(size: int, end: float) -> np.ndarray:
    return np.linspace(0.0, end, size, dtype=np.float64)


def _stats(path, input_names, output_names, low=-20.0, high=20.0) -> None:
    payload = {
        "input": {name: {"min": low, "max": high} for name in input_names},
        "output": {name: {"min": low, "max": high} for name in output_names},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_static(path, lat, lon) -> None:
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    xr.Dataset(
        {
            "elevation": (("lat", "lon"), (2.0 + yy + xx).astype(np.float32)),
            "land_sea_mask": (("lat", "lon"), (yy - xx).astype(np.float32)),
        },
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path, engine="netcdf4")


def _write_low(path, times: pd.DatetimeIndex, lat, lon) -> None:
    hours = np.arange(len(times), dtype=np.float32)[:, None, None, None]
    yy = np.arange(len(lat), dtype=np.float32)[None, None, :, None]
    xx = np.arange(len(lon), dtype=np.float32)[None, None, None, :]
    xr.Dataset(
        {
            "u10": (("time", "height", "lat", "lon"), 3.0 + hours + 0.1 * yy - 0.2 * xx),
            "v10": (("time", "height", "lat", "lon"), -4.0 - 0.2 * hours + 0.3 * yy + 0.1 * xx),
        },
        coords={"time": times, "height": [10.0], "lat": lat, "lon": lon},
    ).to_netcdf(path, engine="netcdf4")


def _write_high(path, times: pd.DatetimeIndex, lat, lon) -> None:
    hours = np.arange(len(times), dtype=np.float32)[:, None, None, None]
    yy = np.arange(len(lat), dtype=np.float32)[None, None, :, None]
    xx = np.arange(len(lon), dtype=np.float32)[None, None, None, :]
    xr.Dataset(
        {
            "u_10m": (("time", "height", "lat", "lon"), 10.0 + hours + 0.2 * yy + xx),
            "v_10m": (("time", "height", "lat", "lon"), -5.0 - hours + 0.1 * yy - xx),
        },
        coords={"time": times, "height": [10.0], "lat": lat, "lon": lon},
    ).to_netcdf(path, engine="netcdf4")


def _dataset(tmp_path, horizon: int = 72, low_times=None, high_times=None):
    low_lat = _grid(5, 2.0)
    low_lon = _grid(6, 2.0)
    high_lat = _grid(9, 1.5)
    high_lon = _grid(11, 1.8)
    if low_times is None:
        low_times = pd.date_range("2020-01-01", periods=horizon + 7, freq="h")
    if high_times is None:
        high_times = low_times
    low = tmp_path / "low.nc"
    high = tmp_path / "high.nc"
    static = tmp_path / "static_high.nc"
    stats = tmp_path / "stats.json"
    _write_low(low, low_times, low_lat, low_lon)
    _write_high(high, high_times, high_lat, high_lon)
    # Deliberately reverse static coordinates; interpolation must align it.
    _write_static(static, high_lat[::-1], high_lon[::-1])
    probe = WindForecastDataset(
        low,
        high,
        static,
        forecast_horizon_hours=horizon,
        years=[2020],
        normalize=False,
    )
    _stats(stats, probe.physical_input_names, probe.output_variables)
    data = WindForecastDataset(
        low,
        high,
        static,
        forecast_horizon_hours=horizon,
        years=[2020],
        stats_path=stats,
    )
    return data


def test_parse_icon_hourly_times_and_reject_off_hour():
    parsed = parse_hourly_times(np.array([20200228.0, 20200228.04166667, 20200229.0]))
    assert parsed.tolist() == [
        pd.Timestamp("2020-02-28 00:00"),
        pd.Timestamp("2020-02-28 01:00"),
        pd.Timestamp("2020-02-29 00:00"),
    ]
    with pytest.raises(ValueError, match="exact hour"):
        parse_hourly_times(np.array([20240101.02]))
    with pytest.raises(ValueError, match="exact hour"):
        parse_hourly_times(np.array(["2024-01-01T00:30:00Z"]))


def test_select_initial_indices_uses_inclusive_2024_test_window():
    times = pd.date_range("2024-01-01", "2024-10-31", freq="D")
    selected = select_initial_indices(
        times,
        initial_hour=0,
        start="2024-01-02T00:00:00",
        end="2024-10-28T00:00:00",
    )
    assert len(selected) == 301
    assert times[int(selected[0])] == pd.Timestamp("2024-01-02 00:00")
    assert times[int(selected[-1])] == pd.Timestamp("2024-10-28 00:00")
    assert pd.Timestamp("2024-01-01 00:00") not in times[selected]
    assert pd.Timestamp("2024-10-29 00:00") not in times[selected]
    assert len(
        select_initial_indices(
            times,
            initial_hour=0,
            start="2024-01-02T00:00:00",
            end="2024-10-28T00:00:00",
            max_initial_times=2,
        )
    ) == 2
    with pytest.raises(ValueError, match="no later"):
        select_initial_indices(
            times, start="2024-10-28T00:00:00", end="2024-01-02T00:00:00"
        )


def test_select_monthly_initial_indices_is_balanced_random_and_reproducible():
    times = pd.date_range("2024-01-01", "2024-03-31 23:00", freq="h")
    filters = dict(
        initial_hour=0,
        start="2024-01-02T00:00:00",
        end="2024-03-29T00:00:00",
    )
    first = select_monthly_initial_indices(
        times, samples_per_month=3, seed=42, **filters
    )
    repeated = select_monthly_initial_indices(
        times, samples_per_month=3, seed=42, **filters
    )
    alternative = select_monthly_initial_indices(
        times, samples_per_month=3, seed=43, **filters
    )

    selected_times = times[first]
    assert len(first) == 9
    assert len(np.unique(first)) == len(first)
    assert np.all(selected_times.hour == 0)
    assert selected_times.is_monotonic_increasing
    assert Counter(selected_times.month) == {1: 3, 2: 3, 3: 3}
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, alternative)
    np.testing.assert_array_equal(
        select_monthly_initial_indices(times, samples_per_month=0, **filters),
        select_initial_indices(times, **filters),
    )


def test_select_monthly_initial_indices_rejects_invalid_or_insufficient_counts():
    times = pd.date_range("2024-01-01", "2024-02-29 23:00", freq="h")
    with pytest.raises(ValueError, match="non-negative"):
        select_monthly_initial_indices(times, samples_per_month=-1)
    with pytest.raises(ValueError, match="Month 2024-02 has only 29"):
        select_monthly_initial_indices(
            times, initial_hour=0, samples_per_month=30
        )


def test_uniform_random_sample_indices_are_deterministic_and_unique():
    first = uniform_random_sample_indices(10_000, 512, seed=42)
    repeated = uniform_random_sample_indices(10_000, 512, seed=42)
    alternative = uniform_random_sample_indices(10_000, 512, seed=43)

    assert first.shape == (512,)
    assert np.all(first == repeated)
    assert not np.array_equal(first, alternative)
    assert np.all(np.diff(first) > 0)
    assert len(np.unique(first)) == 512
    assert int(first.min()) >= 0
    assert int(first.max()) < 10_000
    np.testing.assert_array_equal(
        uniform_random_sample_indices(10, 0, seed=42), np.arange(10)
    )
    np.testing.assert_array_equal(
        uniform_random_sample_indices(10, 10, seed=42), np.arange(10)
    )


def test_epoch_random_subset_sampler_is_reproducible_and_epoch_dependent():
    sampler = EpochRandomSubsetSampler(10_000, 512, seed=42)
    first = list(sampler)
    repeated = list(sampler)

    assert len(first) == 512
    assert first == repeated
    assert len(set(first)) == 512
    assert min(first) >= 0
    assert max(first) < 10_000
    loaded = [
        int(value)
        for batch in DataLoader(range(10_000), batch_size=8, sampler=sampler)
        for value in batch
    ]
    assert loaded == first

    sampler.set_epoch(1)
    second = list(sampler)
    assert second != first
    assert len(set(second)) == 512


def test_epoch_random_subset_sampler_full_mode_and_resume_offset():
    sampler = EpochRandomSubsetSampler(32, 0, seed=7)
    indices = list(sampler)
    assert sorted(indices) == list(range(32))
    assert len(indices) == len(set(indices))
    assert len(EpochRandomSubsetSampler(32, 100, seed=7)) == 32

    sampler.set_epoch(3)
    full_epoch = list(sampler)
    sampler.set_start_index(7)
    assert list(sampler) == full_epoch[7:]
    assert len(sampler) == 25


def test_resume_epoch_and_step_supports_new_and_legacy_checkpoints():
    assert resume_epoch_and_step({"epoch": 4}) == (5, 0)
    assert resume_epoch_and_step(
        {"epoch": 4, "epoch_complete": True, "next_batch_step": 500}
    ) == (5, 0)
    assert resume_epoch_and_step(
        {"epoch": 4, "epoch_complete": False, "next_batch_step": 504}
    ) == (4, 504)


def test_joint_dataset_returns_hr_absolute_target_and_high_only_static(tmp_path):
    data = _dataset(tmp_path, horizon=72)
    y, x = data[0]
    assert y.shape == (144, 9, 11)
    assert x.shape == (18, 9, 11)
    assert data.image_shape() == (9, 11)
    assert [c.name for c in data.input_channels()] == ["u10", "v10"] * 6 + [
        "elevation",
        "land_sea_mask",
    ]
    assert torch.equal(x[-4:], torch.from_numpy(data.grid).float())
    # Static fields were regridded from descending coordinates to the asc HR grid.
    assert float(data.static[0, 0, 0]) < float(data.static[0, -1, 0])
    metadata = data.sample_metadata(0)
    assert metadata["initial_time"] == pd.Timestamp("2020-01-01 05:00")
    assert metadata["target_times"][0] == pd.Timestamp("2020-01-01 06:00")
    assert metadata["target_times"][-1] == pd.Timestamp("2020-01-04 05:00")


def test_normalization_and_lead_major_channel_order(tmp_path):
    data = _dataset(tmp_path, horizon=3)
    y, x = data[0]
    physical_x = data.denormalize_input(x.numpy())
    restored_x = data.normalize_input(physical_x)
    np.testing.assert_allclose(restored_x, x.numpy(), rtol=0, atol=1e-6)
    fields = data.forecast_fields(y.numpy())
    assert fields.shape == (3, 2, 9, 11)
    target = data.read_physical_targets(0)
    np.testing.assert_allclose(
        data.denormalize_output(fields), target, rtol=0, atol=2e-5
    )
    # Flattening remains [lead, u, v] with u/v contiguous after H/W.
    np.testing.assert_allclose(y[0], fields[0, 0], atol=1e-6)
    np.testing.assert_allclose(y[1], fields[0, 1], atol=1e-6)


def test_diffusion_residual_reconstructs_normalized_hr_truth(tmp_path):
    data = _dataset(tmp_path, horizon=3)
    truth = torch.from_numpy(data.target_at(0, 1))
    unet_mean = torch.zeros_like(truth)
    residual = truth - unet_mean
    reconstructed = data.denormalize_output((unet_mean + residual).numpy())
    np.testing.assert_allclose(
        reconstructed, data.read_physical_target(0, 1), rtol=0, atol=2e-5
    )


def test_diffusion_view_expands_initial_times_by_lead(tmp_path):
    base = _dataset(tmp_path, horizon=4)
    diffusion = WindDiffusionDataset(base, leads=[1, 3, 4])
    assert len(diffusion) == len(base) * 3
    y, x, lead = diffusion[1]
    assert y.shape == (2, 9, 11)
    assert x.shape == (20, 9, 11)
    assert int(lead) == 3
    phase = 2.0 * np.pi * 2.0 / 4.0
    assert torch.allclose(x[-2], torch.full_like(x[-2], np.sin(phase)), atol=1e-6)
    assert torch.allclose(x[-1], torch.full_like(x[-1], np.cos(phase)), atol=1e-6)
    metadata = diffusion.sample_metadata(1)
    assert metadata["lead_hour"] == 3


def test_dataset_never_crosses_year_boundary(tmp_path):
    low_times = pd.date_range("2019-12-31 00:00", periods=80, freq="h")
    with pytest.raises(ValueError, match="No complete same-year"):
        _dataset(tmp_path, horizon=72, low_times=low_times, high_times=low_times)


def test_constant_channel_statistics_remain_normalizable(tmp_path):
    names = ["u10", "v10"] * 6 + ["elevation", "land_sea_mask"]
    stats = tmp_path / "stats.json"
    _stats(stats, names, ["u10", "v10"])
    payload = json.loads(stats.read_text(encoding="utf-8"))
    payload["input"]["elevation"] = {"min": 0.0, "max": 0.0}
    stats.write_text(json.dumps(payload), encoding="utf-8")
    in_min, in_max, _, _ = load_channel_stats(stats, names, ["u10", "v10"])
    assert in_min[12] == in_max[12] - 1.0


def test_file_time_index_merges_and_sorts_files(tmp_path):
    lat = _grid(3, 1.0)
    lon = _grid(4, 1.0)
    times = pd.date_range("2020-01-01", periods=8, freq="h")
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    _write_low(first, times[:4], lat, lon)
    _write_low(second, times[4:], lat, lon)
    index = FileTimeIndex([second, first])
    assert index.times.tolist() == times.tolist()
    assert index.position(times[5]) == 5
