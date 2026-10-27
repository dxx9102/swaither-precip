from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import xarray as xr

from scripts import evaluate_wind
from swaither.wind_helpers import write_ensemble_netcdf, write_forecast_netcdf


def test_evaluate_wind_reads_deterministic_and_ensemble_outputs(tmp_path, monkeypatch):
    lat = np.linspace(1.0, 2.0, 3)
    lon = np.linspace(3.0, 4.0, 4)
    initial = [pd.Timestamp("2024-01-01 00:00")]
    leads = [1, 2]
    truth_times = pd.date_range("2024-01-01 01:00", periods=2, freq="h")

    truth = tmp_path / "truth.nc"
    xr.Dataset(
        {
            "u_10m": (
                ("time", "lat", "lon"),
                np.zeros((2, len(lat), len(lon)), dtype=np.float32),
            ),
            "v_10m": (
                ("time", "lat", "lon"),
                np.ones((2, len(lat), len(lon)), dtype=np.float32),
            ),
        },
        coords={"time": truth_times, "lat": lat, "lon": lon},
    ).to_netcdf(truth, engine="netcdf4")

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
    output = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wind.py",
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
            "--output",
            str(output),
        ],
    )

    evaluate_wind.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["overall"]["vector_rmse"] == 0.0
    assert set(payload["ensemble"]) == {"lead_1", "lead_2"}
