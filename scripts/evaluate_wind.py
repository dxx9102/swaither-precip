#!/usr/bin/env python3
"""Evaluate deterministic and ensemble wind forecasts against NetCDF truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.datasets.wind import FileTimeIndex, parse_hourly_times
from swaither.wind_helpers import (
    crps_ensemble,
    ensemble_spread_skill,
    radial_spectrum,
    save_metrics_json,
    wind_metrics,
)


def _times(values) -> pd.DatetimeIndex:
    return parse_hourly_times(values)


def _aggregate(per_lead: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    names = sorted(next(iter(per_lead.values())).keys())
    return {name: float(np.mean([metrics[name] for metrics in per_lead.values()])) for name in names}


def _radial_mean(spectra: List[np.ndarray], length: int) -> List[float]:
    values = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for item in spectra:
        take = min(len(item), length)
        values[:take] += item[:take]
        counts[:take] += 1
    return (values / np.maximum(counts, 1)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--truth-u", required=True)
    parser.add_argument("--truth-v", required=True)
    parser.add_argument("--forecast-u", default="u10")
    parser.add_argument("--forecast-v", default="v10")
    parser.add_argument("--ensemble", default=None, help="Optional ensemble NetCDF")
    parser.add_argument("--low-condition", default=None, help="Optional Step-1 condition for bilinear baseline")
    parser.add_argument("--persistence", action="store_true", help="Add last-observation baseline")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    truth_index = FileTimeIndex([args.truth])
    with xr.open_dataset(args.forecast, engine="netcdf4") as forecast_ds:
        initial_times = _times(forecast_ds["initial_time"].values)
        lead_values = forecast_ds["lead_hour"].values
        lat = np.asarray(forecast_ds["lat"].values)
        lon = np.asarray(forecast_ds["lon"].values)
        forecast_u = forecast_ds[args.forecast_u]
        forecast_v = forecast_ds[args.forecast_v]
        per_lead: Dict[str, Dict[str, float]] = {}
        persistence: Dict[str, Dict[str, float]] = {}
        bilinear: Dict[str, Dict[str, float]] = {}
        spectrum: Dict[str, Dict[str, List[float]]] = {}
        low_ds = xr.open_dataset(args.low_condition, engine="netcdf4") if args.low_condition else None
        try:
            for lead_index, lead in enumerate(lead_values):
                lead = int(lead)
                valid_times = initial_times + pd.Timedelta(hours=lead)
                positions, keep = [], []
                for index, timestamp in enumerate(valid_times):
                    position = truth_index.time_to_position.get(timestamp)
                    if position is not None:
                        positions.append(position)
                        keep.append(index)
                if not positions:
                    continue
                pred_u = np.asarray(forecast_u.isel(lead_hour=lead_index).values, dtype=np.float32)[keep]
                pred_v = np.asarray(forecast_v.isel(lead_hour=lead_index).values, dtype=np.float32)[keep]
                true_u = truth_index.read_many(args.truth_u, positions, lat, lon, engine="netcdf4")
                true_v = truth_index.read_many(args.truth_v, positions, lat, lon, engine="netcdf4")
                per_lead[f"lead_{lead}"] = wind_metrics(pred_u, pred_v, true_u, true_v)

                if args.persistence:
                    initial_positions, persistence_keep = [], []
                    for kept_position, index in enumerate(keep):
                        initial_position = truth_index.time_to_position.get(initial_times[index])
                        if initial_position is not None:
                            initial_positions.append(initial_position)
                            persistence_keep.append(kept_position)
                    if initial_positions:
                        pers_u = truth_index.read_many(args.truth_u, initial_positions, lat, lon, engine="netcdf4")
                        pers_v = truth_index.read_many(args.truth_v, initial_positions, lat, lon, engine="netcdf4")
                        persistence[f"lead_{lead}"] = wind_metrics(
                            pers_u,
                            pers_v,
                            true_u[persistence_keep],
                            true_v[persistence_keep],
                        )

                if low_ds is not None:
                    low_initial_times = _times(low_ds["initial_time"].values)
                    low_initial_to_position = {
                        timestamp: position for position, timestamp in enumerate(low_initial_times)
                    }
                    low_positions = [
                        low_initial_to_position[initial_times[index]]
                        for index in keep
                        if initial_times[index] in low_initial_to_position
                    ]
                    matched_keep = [
                        index
                        for index in keep
                        if initial_times[index] in low_initial_to_position
                    ]
                    if not low_positions:
                        continue
                    low_u = low_ds["u10"].isel(lead_hour=lead_index).values[low_positions]
                    low_v = low_ds["v10"].isel(lead_hour=lead_index).values[low_positions]
                    low_lat = low_ds["lat"].values
                    low_lon = low_ds["lon"].values
                    bi_u = xr.DataArray(low_u, dims=("initial_time", "lat", "lon"), coords={"lat": low_lat, "lon": low_lon}).interp(lat=lat, lon=lon, method="linear").values
                    bi_v = xr.DataArray(low_v, dims=("initial_time", "lat", "lon"), coords={"lat": low_lat, "lon": low_lon}).interp(lat=lat, lon=lon, method="linear").values
                    bilinear[f"lead_{lead}"] = wind_metrics(
                        bi_u,
                        bi_v,
                        true_u[[keep.index(index) for index in matched_keep]],
                        true_v[[keep.index(index) for index in matched_keep]],
                    )

                spectrum[f"lead_{lead}"] = {
                    "forecast_u": _radial_mean([radial_spectrum(item) for item in pred_u[: min(8, len(pred_u))]], max(pred_u.shape[-2:]) // 2 + 1),
                    "truth_u": _radial_mean([radial_spectrum(item) for item in true_u[: min(8, len(true_u))]], max(true_u.shape[-2:]) // 2 + 1),
                }
        finally:
            if low_ds is not None:
                low_ds.close()

    ensemble_metrics = None
    if args.ensemble:
        with xr.open_dataset(args.ensemble, engine="netcdf4") as ensemble_ds:
            ensemble_initial = _times(ensemble_ds["initial_time"].values)
            ensemble_leads = ensemble_ds["lead_hour"].values
            ensemble_lat = np.asarray(ensemble_ds["lat"].values)
            ensemble_lon = np.asarray(ensemble_ds["lon"].values)
            ensemble_metrics = {}
            for lead_index, lead in enumerate(ensemble_leads):
                valid_times = ensemble_initial + pd.Timedelta(hours=int(lead))
                positions, keep = [], []
                for index, timestamp in enumerate(valid_times):
                    position = truth_index.time_to_position.get(timestamp)
                    if position is not None:
                        positions.append(position)
                        keep.append(index)
                if not positions:
                    continue
                members_u = np.moveaxis(
                    np.asarray(ensemble_ds["u10"].isel(lead_hour=lead_index).values, dtype=np.float32)[keep],
                    1,
                    0,
                )
                members_v = np.moveaxis(
                    np.asarray(ensemble_ds["v10"].isel(lead_hour=lead_index).values, dtype=np.float32)[keep],
                    1,
                    0,
                )
                true_u = truth_index.read_many(args.truth_u, positions, ensemble_lat, ensemble_lon, engine="netcdf4")
                true_v = truth_index.read_many(args.truth_v, positions, ensemble_lat, ensemble_lon, engine="netcdf4")
                spread_u, skill_u = ensemble_spread_skill(members_u, true_u)
                spread_v, skill_v = ensemble_spread_skill(members_v, true_v)
                ensemble_metrics[f"lead_{int(lead)}"] = {
                    "crps_u": crps_ensemble(members_u, true_u),
                    "crps_v": crps_ensemble(members_v, true_v),
                    "spread_u": spread_u,
                    "spread_v": spread_v,
                    "skill_u": skill_u,
                    "skill_v": skill_v,
                    "spread_skill_u": spread_u / max(skill_u, 1e-12),
                    "spread_skill_v": spread_v / max(skill_v, 1e-12),
                }

    payload = {
        "forecast": str(Path(args.forecast).resolve()),
        "truth": str(Path(args.truth).resolve()),
        "per_lead": per_lead,
        "overall": _aggregate(per_lead),
        "persistence": persistence,
        "bilinear": bilinear,
        "ensemble": ensemble_metrics,
        "radial_spectrum": spectrum,
    }
    save_metrics_json(Path(args.output), payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
