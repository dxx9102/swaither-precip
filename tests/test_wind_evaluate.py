from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts import evaluate_wind
from swaither.wind_helpers import (
    crps_ensemble,
    write_ensemble_netcdf,
    write_forecast_netcdf,
)


def _truth_file(tmp_path: Path, lat: np.ndarray, lon: np.ndarray) -> tuple[Path, pd.DatetimeIndex]:
    truth_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 01:00"),
            pd.Timestamp("2024-01-01 02:00"),
            pd.Timestamp("2024-01-02 00:00"),
            pd.Timestamp("2024-01-02 01:00"),
            pd.Timestamp("2024-01-02 02:00"),
        ]
    )
    truth = tmp_path / "truth.nc"
    xr.Dataset(
        {
            "u_10m": (
                ("time", "lat", "lon"),
                np.zeros((len(truth_times), len(lat), len(lon)), dtype=np.float32),
            ),
            "v_10m": (
                ("time", "lat", "lon"),
                np.ones((len(truth_times), len(lat), len(lon)), dtype=np.float32),
            ),
        },
        coords={"time": truth_times, "lat": lat, "lon": lon},
    ).to_netcdf(truth, engine="netcdf4")
    return truth, truth_times


def _run_main(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_wind.py", *arguments])
    evaluate_wind.main()


def _assert_json_close(first, second) -> None:
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_json_close(first[key], second[key])
    elif isinstance(first, list):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_json_close(left, right)
    elif isinstance(first, (int, float)) and not isinstance(first, bool):
        assert second == pytest.approx(first, rel=1e-12, abs=1e-12)
    else:
        assert first == second


def test_evaluate_wind_keeps_legacy_combined_output(tmp_path, monkeypatch):
    lat = np.linspace(1.0, 2.0, 3)
    lon = np.linspace(3.0, 4.0, 4)
    truth, _ = _truth_file(tmp_path, lat, lon)
    initial = [pd.Timestamp("2024-01-01 00:00")]
    leads = [1, 2]

    forecast = tmp_path / "forecast.nc"
    write_forecast_netcdf(
        forecast,
        np.zeros((1, 2, len(lat), len(lon))),
        np.ones((1, 2, len(lat), len(lon))),
        initial,
        leads,
        lat,
        lon,
    )
    ensemble = tmp_path / "ensemble.nc"
    write_ensemble_netcdf(
        ensemble,
        np.zeros((1, 2, 3, len(lat), len(lon))),
        np.ones((1, 2, 3, len(lat), len(lon))),
        initial,
        leads,
        lat,
        lon,
    )
    low_condition = tmp_path / "low_condition.nc"
    xr.Dataset(
        {
            "u10": (
                ("initial_time", "lead_hour", "lat", "lon"),
                np.zeros((1, 2, len(lat), len(lon)), dtype=np.float32),
            ),
            "v10": (
                ("initial_time", "lead_hour", "lat", "lon"),
                np.ones((1, 2, len(lat), len(lon)), dtype=np.float32),
            ),
        },
        coords={"initial_time": initial, "lead_hour": leads, "lat": lat, "lon": lon},
    ).to_netcdf(low_condition, engine="netcdf4")
    output = tmp_path / "metrics.json"
    _run_main(
        monkeypatch,
        [
            "--forecast",
            str(forecast),
            "--truth",
            str(truth),
            "--truth-u",
            "u_10m",
            "--truth-v",
            "v_10m",
            "--ensemble",
            str(ensemble),
            "--persistence",
            "--low-condition",
            str(low_condition),
            "--output",
            str(output),
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["overall"]["vector_rmse"] == pytest.approx(0.0)
    assert set(payload["ensemble"]) == {"lead_1", "lead_2"}
    assert set(payload["diffusion_mean"]["per_lead"]) == {"lead_1", "lead_2"}
    assert set(payload["persistence"]) == {"lead_1", "lead_2"}
    assert set(payload["bilinear"]) == {"lead_1", "lead_2"}


def test_staged_metrics_include_diffusion_mean_and_comparison(tmp_path, monkeypatch):
    lat = np.linspace(1.0, 2.0, 3)
    lon = np.linspace(3.0, 4.0, 4)
    truth, _ = _truth_file(tmp_path, lat, lon)
    initial = [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-02 00:00")]
    leads = [1, 2]

    forecast = tmp_path / "forecast.nc"
    write_forecast_netcdf(
        forecast,
        np.ones((2, 2, len(lat), len(lon))),
        np.full((2, 2, len(lat), len(lon)), 2.0),
        initial,
        leads,
        lat,
        lon,
    )
    members_u = np.asarray([0.0, 2.0, 4.0]).reshape(1, 1, 3, 1, 1)
    members_v = np.asarray([1.0, 3.0, 5.0]).reshape(1, 1, 3, 1, 1)
    ensemble_u = np.broadcast_to(
        members_u, (len(initial), len(leads), 3, len(lat), len(lon))
    ).copy()
    ensemble_v = np.broadcast_to(
        members_v, (len(initial), len(leads), 3, len(lat), len(lon))
    ).copy()
    ensemble = tmp_path / "ensemble.nc"
    write_ensemble_netcdf(
        ensemble,
        ensemble_u,
        ensemble_v,
        initial,
        leads,
        lat,
        lon,
    )

    regression_json = tmp_path / "regression.json"
    diffusion_json = tmp_path / "diffusion.json"
    comparison_json = tmp_path / "comparison.json"
    arguments = [
        "--forecast",
        str(forecast),
        "--ensemble",
        str(ensemble),
        "--truth",
        str(truth),
        "--truth-u",
        "u_10m",
        "--truth-v",
        "v_10m",
        "--batch-size",
        "1",
        "--regression-output",
        str(regression_json),
        "--diffusion-output",
        str(diffusion_json),
        "--comparison-output",
        str(comparison_json),
        "--regression-checkpoint",
        "regression.mdlus",
        "--diffusion-checkpoint",
        "diffusion.mdlus",
    ]
    _run_main(monkeypatch, arguments)

    regression = json.loads(regression_json.read_text(encoding="utf-8"))
    diffusion = json.loads(diffusion_json.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_json.read_text(encoding="utf-8"))

    assert regression["overall"]["vector_rmse"] == pytest.approx(np.sqrt(2.0))
    assert diffusion["diffusion_mean"]["overall"]["vector_rmse"] == pytest.approx(
        np.sqrt(8.0)
    )
    assert diffusion["metadata"]["initial_time_count"] == 2
    assert diffusion["metadata"]["ensemble_size"] == 3
    assert diffusion["metadata"]["matched_initial_times_by_lead"] == {
        "lead_1": 2,
        "lead_2": 2,
    }
    expected_crps = crps_ensemble(
        np.moveaxis(ensemble_u[:, 0], 1, 0),
        np.zeros((2, len(lat), len(lon))),
    )
    assert diffusion["ensemble"]["lead_1"]["crps_u"] == pytest.approx(expected_crps)
    assert comparison["diffusion_minus_regression"]["overall"]["vector_rmse"] == (
        pytest.approx(np.sqrt(8.0) - np.sqrt(2.0))
    )
    assert comparison["metadata"]["regression_checkpoint"] == "regression.mdlus"
    assert comparison["metadata"]["diffusion_checkpoint"] == "diffusion.mdlus"

    batched_regression = tmp_path / "regression_batched.json"
    batched_diffusion = tmp_path / "diffusion_batched.json"
    batched_comparison = tmp_path / "comparison_batched.json"
    batched_arguments = list(arguments)
    batched_arguments[batched_arguments.index("--batch-size") + 1] = "2"
    batched_arguments[
        batched_arguments.index(str(regression_json))
    ] = str(batched_regression)
    batched_arguments[
        batched_arguments.index(str(diffusion_json))
    ] = str(batched_diffusion)
    batched_arguments[
        batched_arguments.index(str(comparison_json))
    ] = str(batched_comparison)
    _run_main(monkeypatch, batched_arguments)

    _assert_json_close(
        regression["overall"],
        json.loads(batched_regression.read_text(encoding="utf-8"))["overall"],
    )
    _assert_json_close(
        diffusion["diffusion_mean"],
        json.loads(batched_diffusion.read_text(encoding="utf-8"))[
            "diffusion_mean"
        ],
    )
    _assert_json_close(
        diffusion["ensemble"],
        json.loads(batched_diffusion.read_text(encoding="utf-8"))["ensemble"],
    )
    _assert_json_close(
        comparison["diffusion_minus_regression"],
        json.loads(batched_comparison.read_text(encoding="utf-8"))[
            "diffusion_minus_regression"
        ],
    )


def test_comparison_restricts_regression_to_diffusion_initial_times(
    tmp_path, monkeypatch
):
    lat = np.linspace(1.0, 2.0, 3)
    lon = np.linspace(3.0, 4.0, 4)
    truth, _ = _truth_file(tmp_path, lat, lon)
    leads = [1, 2]
    regression_initial = [
        pd.Timestamp("2024-01-01 00:00"),
        pd.Timestamp("2024-01-02 00:00"),
    ]
    diffusion_initial = [pd.Timestamp("2024-01-02 00:00")]

    forecast = tmp_path / "forecast.nc"
    write_forecast_netcdf(
        forecast,
        np.asarray([10.0, 0.0]).reshape(2, 1, 1, 1)
        * np.ones((2, len(leads), len(lat), len(lon))),
        np.asarray([10.0, 1.0]).reshape(2, 1, 1, 1)
        * np.ones((2, len(leads), len(lat), len(lon))),
        regression_initial,
        leads,
        lat,
        lon,
    )
    ensemble = tmp_path / "ensemble.nc"
    write_ensemble_netcdf(
        ensemble,
        np.asarray([1.0, 3.0]).reshape(1, 1, 2, 1, 1)
        * np.ones((1, len(leads), 2, len(lat), len(lon))),
        np.asarray([2.0, 4.0]).reshape(1, 1, 2, 1, 1)
        * np.ones((1, len(leads), 2, len(lat), len(lon))),
        diffusion_initial,
        leads,
        lat,
        lon,
    )

    regression_json = tmp_path / "regression.json"
    diffusion_json = tmp_path / "diffusion.json"
    comparison_json = tmp_path / "comparison.json"
    _run_main(
        monkeypatch,
        [
            "--forecast",
            str(forecast),
            "--ensemble",
            str(ensemble),
            "--truth",
            str(truth),
            "--truth-u",
            "u_10m",
            "--truth-v",
            "v_10m",
            "--regression-output",
            str(regression_json),
            "--diffusion-output",
            str(diffusion_json),
            "--comparison-output",
            str(comparison_json),
        ],
    )

    regression = json.loads(regression_json.read_text(encoding="utf-8"))
    diffusion = json.loads(diffusion_json.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_json.read_text(encoding="utf-8"))
    assert regression["metadata"]["initial_time_count"] == 2
    assert regression["overall"]["vector_rmse"] > 0.0
    assert diffusion["metadata"]["initial_time_count"] == 1
    assert comparison["metadata"]["initial_time_count"] == 1
    assert comparison["metadata"]["initial_time_scope"] == "diffusion_initial_times"
    assert comparison["metadata"]["full_regression_initial_time_count"] == 2
    assert comparison["regression"]["overall"]["vector_rmse"] == pytest.approx(0.0)
    assert comparison["diffusion_minus_regression"]["overall"][
        "vector_rmse"
    ] == pytest.approx(np.sqrt(8.0))


def test_ensemble_can_be_evaluated_without_a_regression_forecast(tmp_path, monkeypatch):
    lat = np.linspace(1.0, 2.0, 3)
    lon = np.linspace(3.0, 4.0, 4)
    truth, _ = _truth_file(tmp_path, lat, lon)
    initial = [pd.Timestamp("2024-01-01 00:00")]
    leads = [1]
    ensemble = tmp_path / "ensemble.nc"
    write_ensemble_netcdf(
        ensemble,
        np.zeros((1, 1, 2, len(lat), len(lon))),
        np.ones((1, 1, 2, len(lat), len(lon))),
        initial,
        leads,
        lat,
        lon,
    )
    output = tmp_path / "diffusion.json"
    _run_main(
        monkeypatch,
        [
            "--ensemble",
            str(ensemble),
            "--truth",
            str(truth),
            "--truth-u",
            "u_10m",
            "--truth-v",
            "v_10m",
            "--diffusion-output",
            str(output),
        ],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["diffusion_mean"]["overall"]["vector_rmse"] == pytest.approx(0.0)
    assert payload["ensemble"]["lead_1"]["crps_u"] == pytest.approx(0.0)
