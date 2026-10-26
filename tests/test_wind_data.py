from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from swaither.datasets.wind import (
    FileTimeIndex,
    WindForecastDataset,
    WindSuperResolutionDataset,
    load_channel_stats,
    parse_hourly_times,
)


def _icon_values(times: pd.DatetimeIndex) -> np.ndarray:
    dates = times.year * 10000 + times.month * 100 + times.day
    return dates.astype(np.float64) + times.hour.to_numpy(dtype=np.float64) / 24.0


def _grid(size: int, end: float) -> np.ndarray:
    return np.linspace(0.0, end, size, dtype=np.float64)


def _stats(path, input_names, output_names, low=-10.0, high=10.0) -> None:
    payload = {
        "input": {name: {"min": low, "max": high} for name in input_names},
        "output": {name: {"min": low, "max": high} for name in output_names},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_static(path, lat, lon) -> None:
    yy, xx = np.meshgrid(np.linspace(0, 1, len(lat)), np.linspace(0, 1, len(lon)), indexing="ij")
    xr.Dataset(
        {
            "elevation": (("lat", "lon"), 2.0 + yy + xx),
            "land_sea_mask": (("lat", "lon"), yy - xx),
        },
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path, engine="netcdf4")


def _write_low(path, times: pd.DatetimeIndex, lat, lon) -> None:
    hours = np.arange(len(times), dtype=np.float32)[:, None, None, None]
    yy = np.arange(len(lat), dtype=np.float32)[None, None, :, None]
    xx = np.arange(len(lon), dtype=np.float32)[None, None, None, :]
    u = 3.0 + hours + 0.1 * yy - 0.2 * xx
    v = -4.0 - 0.5 * hours + 0.3 * yy + 0.1 * xx
    xr.Dataset(
        {
            "u10": (("time", "height", "lat", "lon"), u),
            "v10": (("time", "height", "lat", "lon"), v),
        },
        coords={"time": times, "height": [10.0], "lat": lat, "lon": lon},
    ).to_netcdf(path, engine="netcdf4")


def _write_condition(path, initial_times: pd.DatetimeIndex, leads, lat, lon) -> None:
    lead_values = np.asarray(leads, dtype=np.float64)[None, :, None, None]
    base_u = np.arange(len(initial_times), dtype=np.float32)[:, None, None, None]
    base_v = -base_u
    u = 20.0 + base_u + lead_values
    v = -20.0 + base_v - lead_values
    xr.Dataset(
        {
            "u10": (("initial_time", "lead_hour", "lat", "lon"), u + np.zeros((1, 1, len(lat), len(lon)), np.float32)),
            "v10": (("initial_time", "lead_hour", "lat", "lon"), v + np.zeros((1, 1, len(lat), len(lon)), np.float32)),
        },
        coords={"initial_time": initial_times, "lead_hour": leads, "lat": lat, "lon": lon},
    ).to_netcdf(path, engine="netcdf4")


def _write_high(path, times: pd.DatetimeIndex, lat, lon, keep=None) -> None:
    hours = np.arange(len(times), dtype=np.float32)[:, None, None, None]
    u = 100.0 + hours + np.zeros((1, 1, len(lat), len(lon)), np.float32)
    v = -50.0 - hours + np.zeros((1, 1, len(lat), len(lon)), np.float32)
    if keep is None:
        keep = np.ones(len(times), dtype=np.float32)
    keep = np.asarray(keep, dtype=np.float32).reshape((len(times), 1, 1, 1))
    xr.Dataset(
        {
            "u_10m": (("time", "height", "lat", "lon"), u * keep),
            "v_10m": (("time", "height", "lat", "lon"), v * keep),
        },
        coords={"time": _icon_values(times), "height": [10.0], "lat": lat, "lon": lon},
    ).to_netcdf(path, engine="netcdf4")


def test_parse_icon_hourly_times_and_reject_off_hour():
    parsed = parse_hourly_times(np.array([20200228.0, 20200228.04166667, 20200229.0]))
    assert parsed.tolist() == [
        pd.Timestamp("2020-02-28 00:00"),
        pd.Timestamp("2020-02-28 01:00"),
        pd.Timestamp("2020-02-29 00:00"),
    ]
    strings = parse_hourly_times(np.array(["20240229.0", "20240229.5"], dtype=object))
    assert strings[1] == pd.Timestamp("2024-02-29 12:00")
    with pytest.raises(ValueError, match="exact hour"):
        parse_hourly_times(np.array([20240101.02]))
    with pytest.raises(ValueError, match="exact hour"):
        parse_hourly_times(np.array([np.datetime64("2024-01-01T00:30")]))


def test_forecast_dataset_channels_negative_wind_causality_and_roundtrip(tmp_path):
    lat = _grid(3, 1.0)
    lon = _grid(4, 1.0)
    times = pd.date_range("2020-01-01", periods=13, freq="h")
    low = tmp_path / "low.nc"
    static = tmp_path / "static.nc"
    stats = tmp_path / "stats.json"
    _write_low(low, times, lat, lon)
    _write_static(static, lat, lon)
    probe = WindForecastDataset(
        low,
        static,
        history_hours=6,
        forecast_horizon_hours=1,
        years=[2020],
        normalize=False,
        preload_low=True,
    )
    _stats(stats, probe.physical_input_names, probe.output_variables)
    data = WindForecastDataset(
        low,
        static,
        history_hours=6,
        forecast_horizon_hours=1,
        years=[2020],
        stats_path=stats,
    )

    assert len(data) == 7
    y, x = data[0]
    assert y.shape == (2, 3, 4)
    assert x.shape == (18, 3, 4)
    assert data.input_channels()[0].name == "u10"
    assert [channel.name for channel in data.input_channels()[:12]] == ["u10", "v10"] * 6
    metadata = data.sample_metadata(0)
    assert metadata["history_times"][-1] == pd.Timestamp("2020-01-01 05:00")
    assert metadata["target_times"] == [pd.Timestamp("2020-01-01 06:00")]

    physical = data.denormalize_input(x.numpy())
    assert physical[:12].min() < 0
    restored = data.denormalize_input(data.normalize_input(physical))
    np.testing.assert_allclose(restored, physical, rtol=0, atol=2e-5)
    physical_batch = np.stack([physical, physical * 0.5], axis=0)
    normalized_batch = data.normalize_input(physical_batch)
    np.testing.assert_allclose(
        normalized_batch[0], data.normalize_input(physical), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        data.denormalize_input(normalized_batch),
        physical_batch,
        rtol=0,
        atol=2e-5,
    )
    physical_y = data.denormalize_output(y.numpy())
    restored_y = data.denormalize_output(data.normalize_output(physical_y))
    np.testing.assert_allclose(restored_y, physical_y, rtol=0, atol=2e-5)


def test_constant_channel_statistics_are_normalizable(tmp_path):
    input_names = ["u10", "v10", "elevation", "land_sea_mask"]
    output_names = ["u10", "v10"]
    stats = tmp_path / "stats.json"
    _stats(stats, input_names, output_names)
    payload = json.loads(stats.read_text(encoding="utf-8"))
    payload["input"]["elevation"] = {"min": 0.0, "max": 0.0}
    stats.write_text(json.dumps(payload), encoding="utf-8")

    in_min, in_max, _, _ = load_channel_stats(stats, input_names, output_names)
    assert in_min[2] == in_max[2] - 1.0

    lat = _grid(3, 1.0)
    lon = _grid(4, 1.0)
    times = pd.date_range("2020-01-01", periods=7, freq="h")
    low = tmp_path / "low.nc"
    static = tmp_path / "static.nc"
    _write_low(low, times, lat, lon)
    _write_static(static, lat, lon)
    with xr.open_dataset(static, engine="netcdf4") as ds:
        constant = ds.load()
    constant["elevation"] = xr.zeros_like(constant["elevation"])
    constant.to_netcdf(static, engine="netcdf4")
    data = WindForecastDataset(
        low,
        static,
        forecast_horizon_hours=1,
        years=[2020],
        stats_path=stats,
    )
    _, x = data[0]
    assert torch.isfinite(x).all()
    assert float(x[12].abs().max()) == 0.0


def test_forecast_dataset_never_crosses_year(tmp_path):
    lat = _grid(3, 1.0)
    lon = _grid(3, 1.0)
    times = pd.date_range("2019-12-31 20:00", "2020-01-01 10:00", freq="h")
    low = tmp_path / "low.nc"
    static = tmp_path / "static.nc"
    _write_low(low, times, lat, lon)
    _write_static(static, lat, lon)
    data = WindForecastDataset(
        low,
        static,
        history_hours=6,
        forecast_horizon_hours=1,
        years=[2020],
        normalize=False,
    )
    assert len(data) == 5
    for index in range(len(data)):
        metadata = data.sample_metadata(index)
        assert {value.year for value in metadata["history_times"]} == {2020}


def test_forecast_dataset_72_hour_residual_contract(tmp_path):
    lat = _grid(3, 1.0)
    lon = _grid(4, 1.0)
    times = pd.date_range("2020-01-01", periods=78, freq="h")
    low = tmp_path / "low.nc"
    static = tmp_path / "static.nc"
    stats = tmp_path / "stats.json"
    _write_low(low, times, lat, lon)
    _write_static(static, lat, lon)
    probe = WindForecastDataset(
        low,
        static,
        history_hours=6,
        forecast_horizon_hours=72,
        years=[2020],
        normalize=False,
    )
    _stats(stats, probe.physical_input_names, probe.output_variables)
    data = WindForecastDataset(
        low,
        static,
        history_hours=6,
        forecast_horizon_hours=72,
        years=[2020],
        stats_path=stats,
    )

    assert len(data) == 1
    residual, x = data[0]
    assert residual.shape == (144, 3, 4)
    assert x.shape == (18, 3, 4)
    metadata = data.sample_metadata(0)
    assert metadata["initial_time"] == pd.Timestamp("2020-01-01 05:00")
    assert metadata["lead_hours"] == list(range(1, 73))
    assert metadata["target_times"][0] == pd.Timestamp("2020-01-01 06:00")
    assert metadata["target_times"][-1] == pd.Timestamp("2020-01-04 05:00")

    targets = data.read_physical_targets(0)
    absolute = data.absolute_output_from_prediction(residual.numpy(), x.numpy())
    assert absolute.shape == (72, 2, 3, 4)
    np.testing.assert_allclose(absolute, targets, rtol=0, atol=2e-5)

    history = data.read_physical_times(range(0, 6))
    expected = (
        data.normalize_output(targets) - data.normalize_output(history[-1])[None, ...]
    ).reshape(144, 3, 4)
    np.testing.assert_allclose(residual.numpy(), expected, rtol=0, atol=1e-5)


def test_forecast_dataset_72_hours_requires_complete_contiguous_window(tmp_path):
    lat = _grid(3, 1.0)
    lon = _grid(3, 1.0)
    complete_times = pd.date_range("2020-01-01", periods=80, freq="h")
    gapped_times = complete_times.delete(20)
    earlier_times = pd.date_range("2020-01-01", periods=77, freq="h")
    for name, times in (("gap", gapped_times), ("short", earlier_times)):
        low = tmp_path / f"{name}_low.nc"
        static = tmp_path / f"{name}_static.nc"
        _write_low(low, times, lat, lon)
        _write_static(static, lat, lon)
        with pytest.raises(ValueError, match="No causal wind samples"):
            WindForecastDataset(
                low,
                static,
                history_hours=6,
                forecast_horizon_hours=72,
                years=[2020],
                normalize=False,
            )


def test_file_time_index_merges_and_sorts_files(tmp_path):
    lat = _grid(3, 1.0)
    lon = _grid(3, 1.0)
    times = pd.date_range("2020-01-01", periods=8, freq="h")
    first = tmp_path / "first.nc"
    second = tmp_path / "second.nc"
    _write_low(first, times[:4], lat, lon)
    _write_low(second, times[4:], lat, lon)
    index = FileTimeIndex([second, first])
    assert index.times.tolist() == times.tolist()
    assert len(index) == len(times)
    assert index.position(times[5]) == 5


def test_super_resolution_dataset_alignment_mapping_and_hybrid_sampling(tmp_path):
    condition_lat = _grid(5, 1.0)
    condition_lon = _grid(5, 1.0)
    high_lat = _grid(17, 1.0)
    high_lon = _grid(17, 1.0)
    initial_times = pd.date_range("2020-01-01", periods=6, freq="h")
    high_times = pd.date_range("2020-01-01", periods=9, freq="h")
    condition = tmp_path / "condition.nc"
    high = tmp_path / "high.nc"
    static = tmp_path / "static_high.nc"
    _write_condition(condition, initial_times, [1, 2, 3], condition_lat, condition_lon)
    _write_high(high, high_times, high_lat, high_lon)
    _write_static(static, high_lat, high_lon)
    data = WindSuperResolutionDataset(
        condition,
        high,
        static,
        leads=(1, 2, 3),
        condition_sampling="hybrid",
        grid_channels=4,
        normalize=False,
        crop_mult=16,
    )
    assert len(data) == 6 + 1 + 1
    assert data.image_shape() == (16, 16)
    assert [channel.name for channel in data.input_channels()] == ["u10", "v10", "elevation", "land_sea_mask"]
    assert [channel.name for channel in data.output_channels()] == ["u10", "v10"]
    y, x = data[0]
    assert y.shape == (2, 16, 16)
    assert x.shape == (8, 16, 16)
    assert float(y[0].mean()) == pytest.approx(101.0)
    assert float(y[1].mean()) == pytest.approx(-51.0)


def test_super_resolution_reports_missing_high_resolution_time(tmp_path):
    condition_lat = _grid(5, 1.0)
    condition_lon = _grid(5, 1.0)
    high_lat = _grid(17, 1.0)
    high_lon = _grid(17, 1.0)
    initial_times = pd.date_range("2020-01-01", periods=4, freq="h")
    high_times = pd.date_range("2020-01-01", periods=3, freq="h")
    condition = tmp_path / "condition.nc"
    high = tmp_path / "high.nc"
    static = tmp_path / "static_high.nc"
    _write_condition(condition, initial_times, [1], condition_lat, condition_lon)
    _write_high(high, high_times, high_lat, high_lon)
    _write_static(static, high_lat, high_lon)
    with pytest.warns(RuntimeWarning, match="Skipped 2 Step-1 conditions"):
        data = WindSuperResolutionDataset(
            condition, high, static, leads=(1,), condition_sampling="all", normalize=False
        )
    assert len(data) == 2
