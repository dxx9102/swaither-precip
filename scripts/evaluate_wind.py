#!/usr/bin/env python3
"""Evaluate deterministic and ensemble wind forecasts against NetCDF truth."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
)


def _times(values) -> pd.DatetimeIndex:
    return parse_hourly_times(values)


def _aggregate(per_lead: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not per_lead:
        raise ValueError("No lead metrics are available for aggregation")
    names = sorted(next(iter(per_lead.values())).keys())
    return {
        name: float(np.mean([metrics[name] for metrics in per_lead.values()]))
        for name in names
    }


def _radial_mean(spectra: List[np.ndarray], length: int) -> List[float]:
    values = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for item in spectra:
        take = min(len(item), length)
        values[:take] += item[:take]
        counts[:take] += 1
    return (values / np.maximum(counts, 1)).tolist()


class _MetricAccumulator:
    """Streaming equivalent of ``wind_metrics`` over initial-time chunks."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def _add(self, name: str, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        self.sums[name] = self.sums.get(name, 0.0) + float(
            np.sum(values[finite], dtype=np.float64)
        )
        self.counts[name] = self.counts.get(name, 0) + int(finite.sum())

    @property
    def count(self) -> int:
        return max(self.counts.values(), default=0)

    def update(
        self,
        pred_u: np.ndarray,
        pred_v: np.ndarray,
        true_u: np.ndarray,
        true_v: np.ndarray,
    ) -> None:
        u = np.asarray(pred_u, dtype=np.float64)
        v = np.asarray(pred_v, dtype=np.float64)
        tu = np.asarray(true_u, dtype=np.float64)
        tv = np.asarray(true_v, dtype=np.float64)
        if not (u.shape == v.shape == tu.shape == tv.shape):
            raise ValueError("u, v, and truth arrays must have identical shapes")

        du = u - tu
        dv = v - tv
        vector_valid = np.isfinite(du) & np.isfinite(dv)
        vector_squared = np.where(vector_valid, du**2 + dv**2, np.nan)
        vector_distance = np.where(vector_valid, np.hypot(du, dv), np.nan)

        pred_speed = np.hypot(u, v)
        true_speed = np.hypot(tu, tv)
        speed_delta = pred_speed - true_speed
        direction_delta = np.abs(
            np.angle(np.exp(1j * (np.arctan2(v, u) - np.arctan2(tv, tu))))
        )

        self._add("u_mae", np.abs(du))
        self._add("v_mae", np.abs(dv))
        self._add("u_rmse", du**2)
        self._add("v_rmse", dv**2)
        self._add("vector_mae", vector_distance)
        self._add("vector_rmse", vector_squared)
        self._add("u_bias", du)
        self._add("v_bias", dv)
        self._add("speed_mae", np.abs(speed_delta))
        self._add("speed_rmse", speed_delta**2)
        self._add("direction_mae_deg", np.degrees(direction_delta))

    def _mean(self, name: str) -> float:
        if self.counts.get(name, 0) < 1:
            return float("nan")
        return self.sums[name] / self.counts[name]

    def result(self) -> Dict[str, float]:
        return {
            "u_mae": self._mean("u_mae"),
            "v_mae": self._mean("v_mae"),
            "u_rmse": float(np.sqrt(self._mean("u_rmse"))),
            "v_rmse": float(np.sqrt(self._mean("v_rmse"))),
            "vector_mae": self._mean("vector_mae"),
            "vector_rmse": float(np.sqrt(self._mean("vector_rmse"))),
            "u_bias": self._mean("u_bias"),
            "v_bias": self._mean("v_bias"),
            "speed_mae": self._mean("speed_mae"),
            "speed_rmse": float(np.sqrt(self._mean("speed_rmse"))),
            "direction_mae_deg": self._mean("direction_mae_deg"),
        }


class _WeightedAccumulator:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int) -> None:
        if count < 1:
            return
        if not np.isfinite(value):
            raise ValueError("A chunked ensemble metric is non-finite")
        self.total += float(value) * int(count)
        self.count += int(count)

    def result(self) -> float:
        if self.count < 1:
            raise ValueError("No finite values were accumulated")
        return self.total / self.count


@dataclass
class _EvaluationResult:
    payload: Dict[str, Any]
    initial_times: pd.DatetimeIndex
    leads: Tuple[int, ...]


def _match_valid_times(
    initial_times: pd.DatetimeIndex,
    lead: int,
    truth_index: FileTimeIndex,
) -> Tuple[List[int], np.ndarray]:
    positions: List[int] = []
    keep: List[int] = []
    for index, timestamp in enumerate(initial_times + pd.Timedelta(hours=lead)):
        position = truth_index.time_to_position.get(timestamp)
        if position is not None:
            positions.append(position)
            keep.append(index)
    if not positions:
        raise ValueError(f"No truth times match forecast lead {lead}")
    return positions, np.asarray(keep, dtype=np.int64)


def _read_truth_pair(
    truth_index: FileTimeIndex,
    positions: Sequence[int],
    lat: np.ndarray,
    lon: np.ndarray,
    truth_u_name: str,
    truth_v_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    truth = truth_index.read_variables_many(
        [truth_u_name, truth_v_name], positions, lat, lon, engine="netcdf4"
    )
    return truth[:, 0], truth[:, 1]


def _assert_finite_metrics(metrics: Dict[str, float], location: str) -> None:
    invalid = [name for name, value in metrics.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(f"Non-finite metrics at {location}: {invalid}")


def _metric_payload(
    path: Path,
    truth_path: Path,
    truth_index: FileTimeIndex,
    initial_times: pd.DatetimeIndex,
    leads: Sequence[int],
    per_lead: Dict[str, Dict[str, float]],
    matched_counts: Dict[str, int],
    batch_size: int,
    checkpoint: str | None,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = {
        "forecast": str(path.resolve()),
        "truth": str(truth_path.resolve()),
        "checkpoint": checkpoint,
        "truth_start": truth_index.times.min().isoformat(),
        "truth_end": truth_index.times.max().isoformat(),
        "truth_time_count": len(truth_index),
        "initial_time_start": initial_times.min().isoformat(),
        "initial_time_end": initial_times.max().isoformat(),
        "initial_time_count": len(initial_times),
        "leads": [int(lead) for lead in leads],
        "matched_initial_times_by_lead": matched_counts,
        "evaluation_batch_size": int(batch_size),
        **extra,
    }
    payload = {
        **metadata,
        "metadata": metadata,
        "per_lead": per_lead,
        "overall": _aggregate(per_lead),
    }
    _assert_finite_metrics(payload["overall"], "overall")
    for lead, metrics in per_lead.items():
        _assert_finite_metrics(metrics, lead)
    return payload


def _evaluate_deterministic(
    *,
    forecast_path: Path,
    truth_path: Path,
    truth_index: FileTimeIndex,
    truth_u_name: str,
    truth_v_name: str,
    forecast_u_name: str,
    forecast_v_name: str,
    batch_size: int,
    persistence: bool,
    low_condition: Path | None,
    checkpoint: str | None,
    initial_times_filter: Sequence[pd.Timestamp] | None = None,
) -> _EvaluationResult:
    with xr.open_dataset(forecast_path, engine="netcdf4") as forecast_ds:
        initial_times = _times(forecast_ds["initial_time"].values)
        lead_values = tuple(int(value) for value in forecast_ds["lead_hour"].values)
        lat = np.asarray(forecast_ds["lat"].values)
        lon = np.asarray(forecast_ds["lon"].values)
        forecast_u = forecast_ds[forecast_u_name]
        forecast_v = forecast_ds[forecast_v_name]
        if initial_times_filter is not None:
            requested_times = _times(initial_times_filter)
            missing = requested_times.difference(initial_times)
            if len(missing):
                raise ValueError(
                    "Requested regression initial times are absent from the "
                    f"forecast: {missing.tolist()}"
                )
            selected_positions = initial_times.get_indexer(requested_times)
            initial_times = initial_times[selected_positions]
            forecast_u = forecast_u.isel(initial_time=selected_positions)
            forecast_v = forecast_v.isel(initial_time=selected_positions)

        forecast_acc = {
            f"lead_{lead}": _MetricAccumulator() for lead in lead_values
        }
        persistence_acc = {
            f"lead_{lead}": _MetricAccumulator() for lead in lead_values
        }
        bilinear_acc = {
            f"lead_{lead}": _MetricAccumulator() for lead in lead_values
        }
        matched_counts: Dict[str, int] = {}
        spectrum: Dict[str, Dict[str, List[float]]] = {}
        low_ds = (
            xr.open_dataset(low_condition, engine="netcdf4")
            if low_condition is not None
            else None
        )
        try:
            low_initial_to_position = (
                {
                    timestamp: position
                    for position, timestamp in enumerate(
                        _times(low_ds["initial_time"].values)
                    )
                }
                if low_ds is not None
                else {}
            )
            low_leads = (
                np.asarray(low_ds["lead_hour"].values, dtype=np.int64)
                if low_ds is not None and "lead_hour" in low_ds.coords
                else None
            )
            for lead_index, lead in enumerate(lead_values):
                positions, keep = _match_valid_times(initial_times, lead, truth_index)
                key = f"lead_{lead}"
                matched_counts[key] = len(positions)
                lead_true_u, lead_true_v = _read_truth_pair(
                    truth_index,
                    positions,
                    lat,
                    lon,
                    truth_u_name,
                    truth_v_name,
                )
                for start in range(0, len(keep), batch_size):
                    stop = min(start + batch_size, len(keep))
                    keep_chunk = keep[start:stop]
                    pred_u = np.asarray(
                        forecast_u.isel(
                            lead_hour=lead_index, initial_time=keep_chunk
                        ).values,
                        dtype=np.float32,
                    )
                    pred_v = np.asarray(
                        forecast_v.isel(
                            lead_hour=lead_index, initial_time=keep_chunk
                        ).values,
                        dtype=np.float32,
                    )
                    true_u = lead_true_u[start:stop]
                    true_v = lead_true_v[start:stop]
                    forecast_acc[key].update(pred_u, pred_v, true_u, true_v)

                    if persistence:
                        initial_positions: List[int] = []
                        persistence_keep: List[int] = []
                        for local_index, initial_index in enumerate(keep_chunk):
                            position = truth_index.time_to_position.get(
                                initial_times[int(initial_index)]
                            )
                            if position is not None:
                                initial_positions.append(position)
                                persistence_keep.append(local_index)
                        if initial_positions:
                            pers_u, pers_v = _read_truth_pair(
                                truth_index,
                                initial_positions,
                                lat,
                                lon,
                                truth_u_name,
                                truth_v_name,
                            )
                            persistence_acc[key].update(
                                pers_u,
                                pers_v,
                                true_u[persistence_keep],
                                true_v[persistence_keep],
                            )

                    if low_ds is not None:
                        low_positions: List[int] = []
                        bilinear_keep: List[int] = []
                        for local_index, initial_index in enumerate(keep_chunk):
                            position = low_initial_to_position.get(
                                initial_times[int(initial_index)]
                            )
                            if position is not None:
                                low_positions.append(position)
                                bilinear_keep.append(local_index)
                        if low_positions:
                            low_lead_index = (
                                int(np.flatnonzero(low_leads == lead)[0])
                                if low_leads is not None and np.any(low_leads == lead)
                                else lead_index
                            )
                            low_u = np.asarray(
                                low_ds["u10"]
                                .isel(
                                    lead_hour=low_lead_index,
                                    initial_time=low_positions,
                                )
                                .values,
                                dtype=np.float32,
                            )
                            low_v = np.asarray(
                                low_ds["v10"]
                                .isel(
                                    lead_hour=low_lead_index,
                                    initial_time=low_positions,
                                )
                                .values,
                                dtype=np.float32,
                            )
                            low_lat = np.asarray(low_ds["lat"].values)
                            low_lon = np.asarray(low_ds["lon"].values)
                            bilinear_u = xr.DataArray(
                                low_u,
                                dims=("initial_time", "lat", "lon"),
                                coords={"lat": low_lat, "lon": low_lon},
                            ).interp(lat=lat, lon=lon, method="linear").values
                            bilinear_v = xr.DataArray(
                                low_v,
                                dims=("initial_time", "lat", "lon"),
                                coords={"lat": low_lat, "lon": low_lon},
                            ).interp(lat=lat, lon=lon, method="linear").values
                            bilinear_acc[key].update(
                                bilinear_u,
                                bilinear_v,
                                true_u[bilinear_keep],
                                true_v[bilinear_keep],
                            )

                    if key not in spectrum:
                        sample_count = min(8, len(pred_u))
                        spectrum[key] = {
                            "forecast_u": _radial_mean(
                                [
                                    radial_spectrum(item)
                                    for item in pred_u[:sample_count]
                                ],
                                max(pred_u.shape[-2:]) // 2 + 1,
                            ),
                            "truth_u": _radial_mean(
                                [
                                    radial_spectrum(item)
                                    for item in true_u[:sample_count]
                                ],
                                max(true_u.shape[-2:]) // 2 + 1,
                            ),
                        }
        finally:
            if low_ds is not None:
                low_ds.close()

    per_lead = {
        key: accumulator.result() for key, accumulator in forecast_acc.items()
    }
    payload = _metric_payload(
        forecast_path,
        truth_path,
        truth_index,
        initial_times,
        lead_values,
        per_lead,
        matched_counts,
        batch_size,
        checkpoint,
        {"kind": "deterministic", "model": "regression"},
    )
    payload["persistence"] = {
        key: accumulator.result()
        for key, accumulator in persistence_acc.items()
        if accumulator.count
    }
    payload["bilinear"] = {
        key: accumulator.result()
        for key, accumulator in bilinear_acc.items()
        if accumulator.count
    }
    payload["radial_spectrum"] = spectrum
    return _EvaluationResult(payload, initial_times, lead_values)


def _evaluate_ensemble(
    *,
    ensemble_path: Path,
    truth_path: Path,
    truth_index: FileTimeIndex,
    truth_u_name: str,
    truth_v_name: str,
    forecast_u_name: str,
    forecast_v_name: str,
    batch_size: int,
    checkpoint: str | None,
) -> _EvaluationResult:
    with xr.open_dataset(ensemble_path, engine="netcdf4") as ensemble_ds:
        if "ensemble" not in ensemble_ds.sizes:
            raise ValueError("The ensemble NetCDF must have an 'ensemble' dimension")
        initial_times = _times(ensemble_ds["initial_time"].values)
        lead_values = tuple(int(value) for value in ensemble_ds["lead_hour"].values)
        lat = np.asarray(ensemble_ds["lat"].values)
        lon = np.asarray(ensemble_ds["lon"].values)
        ensemble_u = ensemble_ds[forecast_u_name]
        ensemble_v = ensemble_ds[forecast_v_name]
        member_count = int(ensemble_ds.sizes["ensemble"])

        mean_acc = {f"lead_{lead}": _MetricAccumulator() for lead in lead_values}
        crps_u_acc = {f"lead_{lead}": _WeightedAccumulator() for lead in lead_values}
        crps_v_acc = {f"lead_{lead}": _WeightedAccumulator() for lead in lead_values}
        spread_u_acc = {
            f"lead_{lead}": _WeightedAccumulator() for lead in lead_values
        }
        spread_v_acc = {
            f"lead_{lead}": _WeightedAccumulator() for lead in lead_values
        }
        skill_u_acc = {
            f"lead_{lead}": _WeightedAccumulator() for lead in lead_values
        }
        skill_v_acc = {
            f"lead_{lead}": _WeightedAccumulator() for lead in lead_values
        }
        matched_counts: Dict[str, int] = {}
        spectrum: Dict[str, Dict[str, List[float]]] = {}

        for lead_index, lead in enumerate(lead_values):
            positions, keep = _match_valid_times(initial_times, lead, truth_index)
            key = f"lead_{lead}"
            matched_counts[key] = len(positions)
            lead_true_u, lead_true_v = _read_truth_pair(
                truth_index,
                positions,
                lat,
                lon,
                truth_u_name,
                truth_v_name,
            )
            for start in range(0, len(keep), batch_size):
                stop = min(start + batch_size, len(keep))
                keep_chunk = keep[start:stop]
                current_u = np.asarray(
                    ensemble_u.isel(
                        lead_hour=lead_index, initial_time=keep_chunk
                    ).values,
                    dtype=np.float32,
                )
                current_v = np.asarray(
                    ensemble_v.isel(
                        lead_hour=lead_index, initial_time=keep_chunk
                    ).values,
                    dtype=np.float32,
                )
                members_u = np.moveaxis(current_u, 1, 0)
                members_v = np.moveaxis(current_v, 1, 0)
                true_u = lead_true_u[start:stop]
                true_v = lead_true_v[start:stop]

                mean_u = np.mean(members_u, axis=0)
                mean_v = np.mean(members_v, axis=0)
                mean_acc[key].update(mean_u, mean_v, true_u, true_v)
                count = len(keep_chunk)
                crps_u_acc[key].update(crps_ensemble(members_u, true_u), count)
                crps_v_acc[key].update(crps_ensemble(members_v, true_v), count)
                spread_u, skill_u = ensemble_spread_skill(members_u, true_u)
                spread_v, skill_v = ensemble_spread_skill(members_v, true_v)
                spread_u_acc[key].update(spread_u, count)
                spread_v_acc[key].update(spread_v, count)
                skill_u_acc[key].update(skill_u, count)
                skill_v_acc[key].update(skill_v, count)

                if key not in spectrum:
                    sample_count = min(8, len(mean_u))
                    spectrum[key] = {
                        "forecast_u": _radial_mean(
                            [radial_spectrum(item) for item in mean_u[:sample_count]],
                            max(mean_u.shape[-2:]) // 2 + 1,
                        ),
                        "truth_u": _radial_mean(
                            [radial_spectrum(item) for item in true_u[:sample_count]],
                            max(true_u.shape[-2:]) // 2 + 1,
                        ),
                    }

    mean_per_lead = {
        key: accumulator.result() for key, accumulator in mean_acc.items()
    }
    ensemble_per_lead: Dict[str, Dict[str, float]] = {}
    for lead in lead_values:
        key = f"lead_{lead}"
        spread_u = spread_u_acc[key].result()
        spread_v = spread_v_acc[key].result()
        skill_u = skill_u_acc[key].result()
        skill_v = skill_v_acc[key].result()
        ensemble_per_lead[key] = {
            "crps_u": crps_u_acc[key].result(),
            "crps_v": crps_v_acc[key].result(),
            "spread_u": spread_u,
            "spread_v": spread_v,
            "skill_u": skill_u,
            "skill_v": skill_v,
            "spread_skill_u": spread_u / max(skill_u, 1e-12),
            "spread_skill_v": spread_v / max(skill_v, 1e-12),
        }

    base = _metric_payload(
        ensemble_path,
        truth_path,
        truth_index,
        initial_times,
        lead_values,
        mean_per_lead,
        matched_counts,
        batch_size,
        checkpoint,
        {
            "kind": "ensemble",
            "model": "diffusion",
            "ensemble_size": member_count,
        },
    )
    ensemble_overall = _aggregate(ensemble_per_lead)
    _assert_finite_metrics(ensemble_overall, "ensemble_overall")
    for lead, metrics in ensemble_per_lead.items():
        _assert_finite_metrics(metrics, f"ensemble_{lead}")
    payload = {
        **base,
        "diffusion_mean": {
            "per_lead": mean_per_lead,
            "overall": _aggregate(mean_per_lead),
        },
        "ensemble": ensemble_per_lead,
        "ensemble_overall": ensemble_overall,
        "radial_spectrum": spectrum,
    }
    return _EvaluationResult(payload, initial_times, lead_values)


def _comparison_payload(
    regression: _EvaluationResult,
    diffusion: _EvaluationResult,
    *,
    full_regression_initial_time_count: int,
) -> Dict[str, Any]:
    if not regression.initial_times.equals(diffusion.initial_times):
        raise ValueError(
            "Regression and diffusion forecasts must use identical initial times"
        )
    diffusion_leads = set(diffusion.leads)
    common_leads = [lead for lead in regression.leads if lead in diffusion_leads]
    if not common_leads:
        raise ValueError("Regression and diffusion forecasts share no lead times")

    regression_per_lead = regression.payload["per_lead"]
    diffusion_per_lead = diffusion.payload["diffusion_mean"]["per_lead"]
    delta_per_lead: Dict[str, Dict[str, float]] = {}
    for lead in common_leads:
        key = f"lead_{lead}"
        common_metrics = sorted(
            set(regression_per_lead[key]) & set(diffusion_per_lead[key])
        )
        delta_per_lead[key] = {
            name: diffusion_per_lead[key][name] - regression_per_lead[key][name]
            for name in common_metrics
        }

    metadata = {
        "regression_forecast": regression.payload["forecast"],
        "diffusion_forecast": diffusion.payload["forecast"],
        "truth": regression.payload["truth"],
        "initial_time_start": regression.initial_times.min().isoformat(),
        "initial_time_end": regression.initial_times.max().isoformat(),
        "initial_time_count": len(regression.initial_times),
        "initial_time_scope": "diffusion_initial_times",
        "full_regression_initial_time_count": int(
            full_regression_initial_time_count
        ),
        "regression_leads": list(regression.leads),
        "diffusion_leads": list(diffusion.leads),
        "common_leads": common_leads,
        "regression_checkpoint": regression.payload.get("checkpoint"),
        "diffusion_checkpoint": diffusion.payload.get("checkpoint"),
        "delta_definition": "diffusion_mean_minus_regression",
    }
    return {
        **metadata,
        "metadata": metadata,
        "regression": {
            "per_lead": regression_per_lead,
            "overall": regression.payload["overall"],
        },
        "diffusion_mean": {
            "per_lead": diffusion_per_lead,
            "overall": diffusion.payload["diffusion_mean"]["overall"],
        },
        "diffusion_minus_regression": {
            "per_lead": delta_per_lead,
            "overall": _aggregate(delta_per_lead),
        },
    }


def _write_json(path: Path | None, payload: Dict[str, Any], label: str) -> None:
    if path is None:
        return
    save_metrics_json(path, payload)
    print(f"Wrote {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", default=None, help="Deterministic regression NetCDF")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--truth-u", required=True)
    parser.add_argument("--truth-v", required=True)
    parser.add_argument("--forecast-u", default="u10")
    parser.add_argument("--forecast-v", default="v10")
    parser.add_argument("--ensemble", default=None, help="Optional ensemble NetCDF")
    parser.add_argument(
        "--low-condition",
        default=None,
        help="Optional low-resolution field for the bilinear baseline",
    )
    parser.add_argument(
        "--persistence", action="store_true", help="Add last-observation baseline"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None, help="Legacy combined metrics JSON")
    parser.add_argument("--regression-output", default=None)
    parser.add_argument("--diffusion-output", default=None)
    parser.add_argument("--comparison-output", default=None)
    parser.add_argument("--regression-checkpoint", default=None)
    parser.add_argument("--diffusion-checkpoint", default=None)
    args = parser.parse_args()

    if args.forecast is None and args.ensemble is None:
        parser.error("at least one of --forecast or --ensemble is required")
    if not any(
        (args.output, args.regression_output, args.diffusion_output, args.comparison_output)
    ):
        parser.error("at least one metrics output path is required")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.regression_output and args.forecast is None:
        parser.error("--regression-output requires --forecast")
    if args.diffusion_output and args.ensemble is None:
        parser.error("--diffusion-output requires --ensemble")
    if args.comparison_output and (args.forecast is None or args.ensemble is None):
        parser.error("--comparison-output requires both --forecast and --ensemble")
    if (args.persistence or args.low_condition) and args.forecast is None:
        parser.error("--persistence and --low-condition require --forecast")

    truth_path = Path(args.truth)
    truth_index = FileTimeIndex([truth_path])
    regression = None
    diffusion = None
    if args.forecast is not None:
        regression = _evaluate_deterministic(
            forecast_path=Path(args.forecast),
            truth_path=truth_path,
            truth_index=truth_index,
            truth_u_name=args.truth_u,
            truth_v_name=args.truth_v,
            forecast_u_name=args.forecast_u,
            forecast_v_name=args.forecast_v,
            batch_size=args.batch_size,
            persistence=args.persistence,
            low_condition=Path(args.low_condition) if args.low_condition else None,
            checkpoint=args.regression_checkpoint,
        )
    if args.ensemble is not None:
        diffusion = _evaluate_ensemble(
            ensemble_path=Path(args.ensemble),
            truth_path=truth_path,
            truth_index=truth_index,
            truth_u_name=args.truth_u,
            truth_v_name=args.truth_v,
            forecast_u_name=args.forecast_u,
            forecast_v_name=args.forecast_v,
            batch_size=args.batch_size,
            checkpoint=args.diffusion_checkpoint,
        )

    _write_json(
        Path(args.regression_output) if args.regression_output else None,
        regression.payload if regression else {},
        "regression metrics",
    )
    _write_json(
        Path(args.diffusion_output) if args.diffusion_output else None,
        diffusion.payload if diffusion else {},
        "diffusion metrics",
    )
    if args.comparison_output:
        assert regression is not None and diffusion is not None
        comparison_regression = regression
        if not regression.initial_times.equals(diffusion.initial_times):
            missing = diffusion.initial_times.difference(regression.initial_times)
            if len(missing):
                raise ValueError(
                    "Diffusion initial times must be a subset of the regression "
                    f"forecast; missing from regression: {missing.tolist()}"
                )
            comparison_regression = _evaluate_deterministic(
                forecast_path=Path(args.forecast),
                truth_path=truth_path,
                truth_index=truth_index,
                truth_u_name=args.truth_u,
                truth_v_name=args.truth_v,
                forecast_u_name=args.forecast_u,
                forecast_v_name=args.forecast_v,
                batch_size=args.batch_size,
                persistence=False,
                low_condition=None,
                checkpoint=args.regression_checkpoint,
                initial_times_filter=diffusion.initial_times,
            )
        _write_json(
            Path(args.comparison_output),
            _comparison_payload(
                comparison_regression,
                diffusion,
                full_regression_initial_time_count=len(regression.initial_times),
            ),
            "comparison metrics",
        )

    if args.output:
        payload: Dict[str, Any] = {
            "forecast": args.forecast,
            "diffusion": args.ensemble,
            "truth": str(truth_path.resolve()),
        }
        if regression is not None:
            payload.update(
                {
                    "per_lead": regression.payload["per_lead"],
                    "overall": regression.payload["overall"],
                    "persistence": regression.payload["persistence"],
                    "bilinear": regression.payload["bilinear"],
                    "radial_spectrum": regression.payload["radial_spectrum"],
                }
            )
        payload["diffusion_mean"] = (
            diffusion.payload["diffusion_mean"] if diffusion is not None else None
        )
        payload["ensemble"] = (
            diffusion.payload["ensemble"] if diffusion is not None else None
        )
        payload["ensemble_overall"] = (
            diffusion.payload["ensemble_overall"] if diffusion is not None else None
        )
        payload["metadata"] = {
            "regression": regression.payload["metadata"] if regression else None,
            "diffusion": diffusion.payload["metadata"] if diffusion else None,
        }
        _write_json(Path(args.output), payload, "combined metrics")


if __name__ == "__main__":
    main()
