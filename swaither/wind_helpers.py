"""Shared storage, NetCDF, and verification helpers for the wind pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORAGE_ROOT = REPO_ROOT


def configure_storage(root: Optional[Path] = None) -> Dict[str, Path]:
    """Create and return wind-pipeline artifact and cache paths.

    Environment variables are set for the process before libraries that capture
    cache locations are imported. Callers can select a different storage root
    through ``root`` or ``SWAITHER_WIND_ROOT``.
    """
    storage = Path(root or os.environ.get("SWAITHER_WIND_ROOT", DEFAULT_STORAGE_ROOT)).resolve()
    paths = {
        "root": storage,
        "outputs": storage / "outputs",
        "generated": storage / "data" / "wind_generated",
        "stats": storage / "data" / "stats",
        "cache": storage / ".cache",
        "wandb": storage / "outputs" / "wandb",
        "tensorboard": storage / "outputs" / "tensorboard",
    }
    for key, path in paths.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    cache = str(paths["cache"])
    # These caches are process-wide and some libraries capture their values at
    # import time, so assign them rather than leaving a HOME-based default in
    # place.  This function is called before physicsnemo/matplotlib imports in
    # the wind entry points.
    defaults = {
        "WANDB_DIR": str(paths["wandb"]),
        "WANDB_CACHE_DIR": str(paths["cache"] / "wandb"),
        "XDG_CACHE_HOME": cache,
        "TORCH_HOME": str(paths["cache"] / "torch"),
        "HF_HOME": str(paths["cache"] / "huggingface"),
        "TRITON_CACHE_DIR": str(paths["cache"] / "triton"),
        "MPLCONFIGDIR": str(paths["cache"] / "matplotlib"),
        "CARTOPY_DATA_DIR": str(paths["cache"] / "cartopy"),
        "LOCAL_CACHE": str(paths["cache"] / "physicsnemo"),
        "NUMBA_CACHE_DIR": str(paths["cache"] / "numba"),
        "JOBLIB_TEMP_FOLDER": str(paths["cache"] / "joblib"),
    }
    for name, value in defaults.items():
        os.environ[name] = value
        Path(value).mkdir(parents=True, exist_ok=True)
    return paths


def _netcdf_initialize(path: Path, dimensions: Dict[str, int]) -> Any:
    import netCDF4 as nc

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = nc.Dataset(path, "w", format="NETCDF4")
    for name, size in dimensions.items():
        handle.createDimension(name, size)
    return handle


class WindForecastNetCDFWriter:
    """Stream deterministic or ensemble wind forecast slices to NetCDF.

    The writer owns the NetCDF handle so callers never need to allocate arrays
    spanning the full test year.  Use as a context manager and write one initial
    time or one ``(initial time, lead)`` slice at a time.
    """

    def __init__(
        self,
        path: Path,
        initial_times: Sequence[pd.Timestamp],
        lead_hours: Sequence[int],
        lat: np.ndarray,
        lon: np.ndarray,
        source: str,
        ensemble_size: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.initial_times = [pd.Timestamp(value) for value in initial_times]
        self.lead_hours = [int(value) for value in lead_hours]
        self.lat = np.asarray(lat, dtype=np.float32)
        self.lon = np.asarray(lon, dtype=np.float32)
        self.source = str(source)
        self.ensemble_size = (
            None if ensemble_size is None else int(ensemble_size)
        )
        if self.ensemble_size is not None and self.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive when provided")
        self._handle: Any = None
        self._u: Any = None
        self._v: Any = None

    @property
    def dimensions(self) -> Tuple[str, ...]:
        base = ("initial_time", "lead_hour")
        if self.ensemble_size is not None:
            base += ("ensemble",)
        return base + ("lat", "lon")

    def __enter__(self) -> "WindForecastNetCDFWriter":
        dimensions = {
            "initial_time": len(self.initial_times),
            "lead_hour": len(self.lead_hours),
            "lat": len(self.lat),
            "lon": len(self.lon),
        }
        if self.ensemble_size is not None:
            dimensions["ensemble"] = self.ensemble_size
        self._handle = _netcdf_initialize(self.path, dimensions)
        handle = self._handle
        try:
            time_var = handle.createVariable("initial_time", str, ("initial_time",))
            time_var[:] = np.asarray(
                [value.strftime("%Y-%m-%dT%H:%M:%SZ") for value in self.initial_times],
                dtype=object,
            )
            time_var.units = "ISO 8601 UTC"
            lead_var = handle.createVariable("lead_hour", "i4", ("lead_hour",))
            lead_var[:] = np.asarray(self.lead_hours, dtype=np.int32)
            lead_var.units = "hours"
            valid_var = handle.createVariable(
                "valid_time", str, ("initial_time", "lead_hour")
            )
            valid_var[:] = np.asarray(
                [
                    [
                        (initial + pd.Timedelta(hours=lead)).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"
                        )
                        for lead in self.lead_hours
                    ]
                    for initial in self.initial_times
                ],
                dtype=object,
            )
            valid_var.units = "ISO 8601 UTC"
            lat_var = handle.createVariable("lat", "f4", ("lat",))
            lat_var[:] = self.lat
            lat_var.units = "degrees_north"
            lon_var = handle.createVariable("lon", "f4", ("lon",))
            lon_var[:] = self.lon
            lon_var.units = "degrees_east"
            self._u = handle.createVariable(
                "u10",
                "f4",
                self.dimensions,
                zlib=True,
                complevel=2,
                shuffle=True,
            )
            self._v = handle.createVariable(
                "v10",
                "f4",
                self.dimensions,
                zlib=True,
                complevel=2,
                shuffle=True,
            )
            for name, variable in (("u10", self._u), ("v10", self._v)):
                variable.units = "m s-1"
                variable.coordinates = "valid_time lat lon"
                variable.long_name = (
                    "10 metre eastward wind"
                    if name == "u10"
                    else "10 metre northward wind"
                )
                variable.source = self.source
            handle.source = self.source
            return self
        except Exception:
            self.close()
            raise

    def write_lead(
        self,
        initial_index: int,
        lead_index: int,
        u: np.ndarray,
        v: np.ndarray,
    ) -> None:
        """Write one spatial field, or ensemble of fields, for one lead."""
        if self._handle is None:
            raise RuntimeError("WindForecastNetCDFWriter is not open")
        initial_index = int(initial_index)
        lead_index = int(lead_index)
        if not 0 <= initial_index < len(self.initial_times):
            raise IndexError("initial_index is outside the writer dimension")
        if not 0 <= lead_index < len(self.lead_hours):
            raise IndexError("lead_index is outside the writer dimension")
        spatial = (len(self.lat), len(self.lon))
        expected = spatial if self.ensemble_size is None else (
            self.ensemble_size,
            *spatial,
        )
        u_values = np.asarray(u, dtype=np.float32)
        v_values = np.asarray(v, dtype=np.float32)
        if u_values.shape != expected or v_values.shape != expected:
            raise ValueError(
                f"Expected wind field shape {expected}, got {u_values.shape} and {v_values.shape}"
            )
        self._u[initial_index, lead_index] = u_values
        self._v[initial_index, lead_index] = v_values

    def write_initial(self, initial_index: int, u: np.ndarray, v: np.ndarray) -> None:
        """Write all configured leads for one initial time."""
        initial_index = int(initial_index)
        lead_shape = (len(self.lead_hours),)
        if self.ensemble_size is not None:
            lead_shape += (self.ensemble_size,)
        expected = lead_shape + (len(self.lat), len(self.lon))
        u_values = np.asarray(u, dtype=np.float32)
        v_values = np.asarray(v, dtype=np.float32)
        if u_values.shape != expected or v_values.shape != expected:
            raise ValueError(
                f"Expected initial-time forecast shape {expected}, got "
                f"{u_values.shape} and {v_values.shape}"
            )
        for lead_index in range(len(self.lead_hours)):
            self.write_lead(initial_index, lead_index, u_values[lead_index], v_values[lead_index])

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


def write_forecast_netcdf(
    path: Path,
    u: np.ndarray,
    v: np.ndarray,
    initial_times: Sequence[pd.Timestamp],
    lead_hours: Sequence[int],
    lat: np.ndarray,
    lon: np.ndarray,
    source: str = "wind-regression",
) -> None:
    """Write deterministic forecasts with dims ``initial_time, lead_hour, lat, lon``."""
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    expected = (len(initial_times), len(lead_hours), len(lat), len(lon))
    if u.shape != expected or v.shape != expected:
        raise ValueError(f"Expected forecast arrays {expected}, got {u.shape} and {v.shape}")
    with WindForecastNetCDFWriter(
        Path(path), initial_times, lead_hours, lat, lon, source
    ) as writer:
        for index in range(len(initial_times)):
            writer.write_initial(index, u[index], v[index])


def write_ensemble_netcdf(
    path: Path,
    u: np.ndarray,
    v: np.ndarray,
    initial_times: Sequence[pd.Timestamp],
    lead_hours: Sequence[int],
    lat: np.ndarray,
    lon: np.ndarray,
    source: str = "corrdiff",
) -> None:
    """Write ensemble forecasts with dims ``initial_time, lead_hour, ensemble, lat, lon``."""
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if u.ndim != 5 or v.shape != u.shape:
        raise ValueError("Ensemble u and v must be 5-D arrays with identical shapes")
    expected = (len(initial_times), len(lead_hours), u.shape[2], len(lat), len(lon))
    if u.shape != expected:
        raise ValueError(f"Expected ensemble shape {expected}, got {u.shape}")
    with WindForecastNetCDFWriter(
        Path(path),
        initial_times,
        lead_hours,
        lat,
        lon,
        source,
        ensemble_size=u.shape[2],
    ) as writer:
        for index in range(len(initial_times)):
            writer.write_initial(index, u[index], v[index])


def _circular_direction_error_deg(pred_u: np.ndarray, pred_v: np.ndarray, true_u: np.ndarray, true_v: np.ndarray) -> float:
    pred = np.arctan2(pred_v, pred_u)
    truth = np.arctan2(true_v, true_u)
    delta = np.abs(np.angle(np.exp(1j * (pred - truth))))
    return float(np.degrees(np.nanmean(delta)))


def wind_metrics(precip_u: np.ndarray, precip_v: np.ndarray, truth_u: np.ndarray, truth_v: np.ndarray) -> Dict[str, float]:
    """Compute deterministic signed-vector wind metrics."""
    u = np.asarray(precip_u, dtype=np.float64)
    v = np.asarray(precip_v, dtype=np.float64)
    tu = np.asarray(truth_u, dtype=np.float64)
    tv = np.asarray(truth_v, dtype=np.float64)
    if not (u.shape == v.shape == tu.shape == tv.shape):
        raise ValueError("u, v, and truth arrays must have identical shapes")
    du, dv = u - tu, v - tv
    pred_speed = np.hypot(u, v)
    true_speed = np.hypot(tu, tv)
    return {
        "u_mae": float(np.nanmean(np.abs(du))),
        "v_mae": float(np.nanmean(np.abs(dv))),
        "u_rmse": float(np.sqrt(np.nanmean(du**2))),
        "v_rmse": float(np.sqrt(np.nanmean(dv**2))),
        "vector_mae": float(np.nanmean(np.hypot(du, dv))),
        "vector_rmse": float(np.sqrt(np.nanmean(du**2 + dv**2))),
        "u_bias": float(np.nanmean(du)),
        "v_bias": float(np.nanmean(dv)),
        "speed_mae": float(np.nanmean(np.abs(pred_speed - true_speed))),
        "speed_rmse": float(np.sqrt(np.nanmean((pred_speed - true_speed) ** 2))),
        "direction_mae_deg": _circular_direction_error_deg(u, v, tu, tv),
    }


def crps_ensemble(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Finite-ensemble CRPS with ensemble on axis 0."""
    members = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if members.ndim < 2 or members.shape[1:] != truth.shape:
        raise ValueError("Ensemble shape must be (M, ...) and match truth trailing shape")
    m = members.shape[0]
    term1 = np.mean(np.abs(members - truth[None, ...]), axis=0)
    ordered = np.sort(members, axis=0)
    ranks = np.arange(1, m + 1, dtype=np.float64).reshape((m,) + (1,) * truth.ndim)
    # The sorted-sample identity avoids an M x M x H x W temporary.
    term2 = 0.5 * np.sum((2.0 * ranks - m - 1.0) * ordered, axis=0) / (m**2)
    return float(np.nanmean(term1 - term2))


def ensemble_spread_skill(ensemble: np.ndarray, truth: np.ndarray) -> Tuple[float, float]:
    members = np.asarray(ensemble, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mean = np.mean(members, axis=0)
    spread = float(np.nanmean(np.std(members, axis=0)))
    skill = float(np.sqrt(np.nanmean((mean - truth) ** 2)))
    return spread, skill


def radial_spectrum(field: np.ndarray) -> np.ndarray:
    """Azimuthally averaged power spectrum for the final two dimensions."""
    values = np.asarray(field, dtype=np.float64)
    if values.shape[-2] < 2 or values.shape[-1] < 2:
        raise ValueError("Spectrum requires a 2-D field")
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(values - np.nanmean(values)), axes=(-2, -1))) ** 2
    y, x = np.indices(values.shape[-2:])
    cy, cx = (np.asarray(values.shape[-2:]) - 1) / 2
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    bins = np.bincount(radius.ravel())
    summed = np.bincount(radius.ravel(), weights=spectrum.ravel())
    return summed / np.maximum(bins, 1)


def save_metrics_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


__all__ = [
    "configure_storage",
    "WindForecastNetCDFWriter",
    "write_forecast_netcdf",
    "write_ensemble_netcdf",
    "wind_metrics",
    "crps_ensemble",
    "ensemble_spread_skill",
    "radial_spectrum",
    "save_metrics_json",
]
