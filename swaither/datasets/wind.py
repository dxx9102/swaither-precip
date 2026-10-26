"""Datasets and normalization helpers for hourly 10 m wind workflows.

The precipitation datasets in this repository use positive, log-transformed
fields.  Wind components are signed and must not be clipped or log-transformed,
so the wind pipeline uses the dedicated datasets in this module.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr

try:  # Imported as ``swaither.datasets.wind``.
    from .base import ChannelMetadata, DownscalingDataset
except ImportError:  # Loaded through ``path/to/file.py::ClassName``.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from base import ChannelMetadata, DownscalingDataset


PathLike = Union[str, os.PathLike]
GRID_CHANNEL_NAMES = (
    "grid_sin_lat",
    "grid_cos_lat",
    "grid_sin_lon",
    "grid_cos_lon",
)


def as_path_list(value: Union[PathLike, Sequence[PathLike], None]) -> List[Path]:
    """Normalize a scalar/list path configuration to an ordered path list."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [Path(value)]
    return [Path(item) for item in value]


def parse_hourly_times(values: Iterable[Any]) -> pd.DatetimeIndex:
    """Parse time coordinates into a canonical hourly ``DatetimeIndex``.

    ICON files supplied for this project encode time as ``YYYYMMDD.f`` where
    ``f`` is the fraction of the day. Floating-point conversion to fractional
    hours is tolerance-checked against an integer hour; off-hour values are
    rejected instead of being silently rounded.
    """
    raw = np.asarray(list(values))
    if raw.size == 0:
        return pd.DatetimeIndex([])
    if np.issubdtype(raw.dtype, np.datetime64):
        parsed = pd.DatetimeIndex(pd.to_datetime(raw))
        aligned = parsed.floor("h")
        if np.any(parsed != aligned):
            first = parsed[np.flatnonzero(parsed != aligned)[0]]
            raise ValueError(f"Time coordinate {first!r} is not aligned to an exact hour")
        return aligned
    if raw.dtype.kind in "OUS":
        strings = np.asarray([str(value).strip() for value in raw], dtype=object)
        icon_style = np.asarray(
            [bool(re.fullmatch(r"\d{8}(?:\.\d+)?", value)) for value in strings]
        )
        if icon_style.all():
            raw = strings.astype(np.float64)
        else:
            try:
                parsed = pd.DatetimeIndex(pd.to_datetime(strings, utc=True)).round("h")
                return parsed.tz_convert(None)
            except Exception as exc:  # pragma: no cover - defensive error path
                bad = next(
                    (value for value in strings if pd.isna(pd.to_datetime(value, errors="coerce"))),
                    strings[0],
                )
                raise ValueError(f"Unsupported time value {bad!r}") from exc
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"Unsupported time coordinate dtype {raw.dtype}")

    values_f = raw.astype(np.float64)
    dates = np.floor(values_f).astype(np.int64)
    fractional_hours = (values_f - dates) * 24.0
    hours = np.rint(fractional_hours).astype(np.int64)
    off_hour = np.abs(fractional_hours - hours) > 1e-3
    if np.any(off_hour):
        first = float(values_f[np.flatnonzero(off_hour)[0]])
        raise ValueError(f"Time coordinate {first!r} is not aligned to an exact hour")
    if np.any((hours < 0) | (hours > 23)):
        raise ValueError("ICON time fraction does not map to an hour in [0, 23]")
    date_index = pd.to_datetime(dates.astype(str), format="%Y%m%d")
    return pd.DatetimeIndex(date_index) + pd.to_timedelta(hours, unit="h")


def _coord_values(ds: xr.Dataset, name: str) -> np.ndarray:
    if name not in ds.coords and name not in ds.variables:
        raise KeyError(f"Required coordinate {name!r} is missing from {ds.encoding.get('source')}")
    return np.asarray(ds[name].values, dtype=np.float64)


def _drop_singleton_height(da: xr.DataArray) -> xr.DataArray:
    if "height" in da.dims and da.sizes["height"] == 1:
        da = da.isel(height=0, drop=True)
    return da


def _sanitize(arr: np.ndarray, nan_policy: str, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        if nan_policy == "error":
            bad = int(arr.size - finite.sum())
            raise ValueError(f"{name} contains {bad} non-finite values")
        if nan_policy != "zero":
            raise ValueError("nan_policy must be 'error' or 'zero'")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _select_time(ds: xr.Dataset, dim: str, index: int, var: str) -> xr.DataArray:
    if dim not in ds[var].dims:
        raise ValueError(f"Variable {var!r} has no {dim!r} dimension: {ds[var].dims}")
    return _drop_singleton_height(ds[var].isel({dim: index}))


def _regrid_2d(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected lat/lon dimensions, got {da.dims}")
    if da.sizes["lat"] != len(lat) or da.sizes["lon"] != len(lon):
        da = da.interp(lat=lat, lon=lon, method="linear")
    else:
        old_lat = np.asarray(da.lat.values)
        old_lon = np.asarray(da.lon.values)
        if not (
            np.allclose(old_lat, lat, rtol=0.0, atol=1e-6)
            and np.allclose(old_lon, lon, rtol=0.0, atol=1e-6)
        ):
            da = da.interp(lat=lat, lon=lon, method="linear")
    return np.asarray(da.values, dtype=np.float32)


def _regrid_batch(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Regrid a time batch with one interpolation call rather than one per hour."""
    da = _drop_singleton_height(da)
    if "time" not in da.dims or "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected time/lat/lon dimensions, got {da.dims}")
    old_lat = np.asarray(da.lat.values)
    old_lon = np.asarray(da.lon.values)
    same_grid = da.sizes["lat"] == len(lat) and da.sizes["lon"] == len(lon)
    same_grid = same_grid and np.allclose(old_lat, lat, rtol=0.0, atol=1e-6)
    same_grid = same_grid and np.allclose(old_lon, lon, rtol=0.0, atol=1e-6)
    if not same_grid:
        da = da.interp(lat=lat, lon=lon, method="linear")
    return np.asarray(da.transpose("time", "lat", "lon").values, dtype=np.float32)


def _grid_from_coords(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_norm = (lat.astype(np.float64) - lat.min()) / max(float(lat.max() - lat.min()), 1e-12)
    lon_norm = (lon.astype(np.float64) - lon.min()) / max(float(lon.max() - lon.min()), 1e-12)
    lat2d, lon2d = np.meshgrid(lat_norm, lon_norm, indexing="ij")
    return np.stack(
        [
            np.sin(np.pi * lat2d),
            np.cos(np.pi * lat2d),
            np.sin(np.pi * lon2d),
            np.cos(np.pi * lon2d),
        ],
        axis=0,
    ).astype(np.float32)


@dataclass(frozen=True)
class TimeEntry:
    path: Path
    local_index: int
    time: pd.Timestamp


class FileTimeIndex:
    """Lazy, sorted index over one or more NetCDF files sharing a time axis."""

    def __init__(
        self,
        files: Sequence[PathLike],
        engine: str = "netcdf4",
        time_name: str = "time",
    ) -> None:
        paths = as_path_list(files)
        if not paths:
            raise ValueError("At least one NetCDF file is required")
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)

        entries: List[TimeEntry] = []
        first_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None
        for path in paths:
            with xr.open_dataset(path, engine=engine) as ds:
                if time_name not in ds.coords:
                    raise ValueError(f"{path} has no {time_name!r} coordinate")
                times = parse_hourly_times(ds[time_name].values)
                entries.extend(
                    TimeEntry(path, local, timestamp)
                    for local, timestamp in enumerate(times)
                )
                coords = (_coord_values(ds, "lat"), _coord_values(ds, "lon"))
                if first_coords is None:
                    first_coords = coords
                elif not (
                    np.allclose(first_coords[0], coords[0], rtol=0.0, atol=1e-6)
                    and np.allclose(first_coords[1], coords[1], rtol=0.0, atol=1e-6)
                ):
                    raise ValueError(f"Grid coordinates in {path} differ from the first input file")

        entries.sort(key=lambda item: item.time)
        times = pd.DatetimeIndex([item.time for item in entries])
        if times.has_duplicates:
            duplicates = times[times.duplicated()].unique()[:5]
            raise ValueError(f"Duplicate timestamps across input files: {list(duplicates)}")
        assert first_coords is not None
        self.files = paths
        self.entries = entries
        self.times = times
        self.lat, self.lon = first_coords
        self.time_to_position = {timestamp: pos for pos, timestamp in enumerate(times)}

    def __len__(self) -> int:
        return len(self.entries)

    def position(self, timestamp: pd.Timestamp) -> int:
        key = pd.Timestamp(timestamp).round("h")
        try:
            return self.time_to_position[key]
        except KeyError as exc:
            raise KeyError(f"Time {key} is not present in the input files") from exc

    def read(
        self,
        variable: str,
        position: int,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        engine: str = "netcdf4",
    ) -> np.ndarray:
        entry = self.entries[position]
        with xr.open_dataset(entry.path, engine=engine) as ds:
            da = _select_time(ds, "time", entry.local_index, variable)
            values = _regrid_2d(da, self.lat if lat is None else lat, self.lon if lon is None else lon)
        return values

    def read_many(
        self,
        variable: str,
        positions: Sequence[int],
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        engine: str = "netcdf4",
    ) -> np.ndarray:
        """Read several times while opening each source file only once per group."""
        target_lat = self.lat if lat is None else np.asarray(lat)
        target_lon = self.lon if lon is None else np.asarray(lon)
        result: List[Optional[np.ndarray]] = [None] * len(positions)
        grouped: Dict[Path, List[Tuple[int, int]]] = {}
        for output_index, position in enumerate(positions):
            entry = self.entries[int(position)]
            grouped.setdefault(entry.path, []).append((output_index, entry.local_index))
        for path, members in grouped.items():
            with xr.open_dataset(path, engine=engine) as ds:
                for output_index, local_index in members:
                    da = _select_time(ds, "time", local_index, variable)
                    result[output_index] = _regrid_2d(da, target_lat, target_lon)
        if any(value is None for value in result):
            raise RuntimeError("Failed to read one or more indexed times")
        return np.stack([value for value in result if value is not None], axis=0)

    def read_variables_many(
        self,
        variables: Sequence[str],
        positions: Sequence[int],
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        engine: str = "netcdf4",
    ) -> np.ndarray:
        """Read several variables and times while opening each file once."""
        target_lat = self.lat if lat is None else np.asarray(lat)
        target_lon = self.lon if lon is None else np.asarray(lon)
        result: List[Optional[np.ndarray]] = [None] * len(positions)
        grouped: Dict[Path, List[Tuple[int, int]]] = {}
        for output_index, position in enumerate(positions):
            entry = self.entries[int(position)]
            grouped.setdefault(entry.path, []).append((output_index, entry.local_index))
        for path, members in grouped.items():
            with xr.open_dataset(path, engine=engine) as ds:
                values = np.stack(
                    [
                        _regrid_batch(
                            ds[name].isel(time=[local for _, local in members]),
                            target_lat,
                            target_lon,
                        )
                        for name in variables
                    ],
                    axis=1,
                )
                for source_index, (output_index, _) in enumerate(members):
                    result[output_index] = values[source_index]
        if any(value is None for value in result):
            raise RuntimeError("Failed to read one or more indexed times")
        return np.stack([value for value in result if value is not None], axis=0)


def load_channel_stats(
    stats_path: Optional[PathLike],
    input_variables: Sequence[str],
    output_variables: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load channel-wise min/max statistics from the repository JSON format."""
    if stats_path is None:
        n_in, n_out = len(input_variables), len(output_variables)
        return np.zeros(n_in, np.float32), np.ones(n_in, np.float32), np.zeros(n_out, np.float32), np.ones(n_out, np.float32)
    with open(stats_path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    try:
        in_stats, out_stats = stats["input"], stats["output"]
    except KeyError as exc:
        raise ValueError(f"{stats_path} must contain 'input' and 'output' objects") from exc

    def collect(group: Dict[str, Dict[str, float]], names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        missing = [name for name in names if name not in group]
        if missing:
            raise KeyError(f"Missing statistics for variables: {missing}")
        lo = np.asarray([group[name]["min"] for name in names], dtype=np.float32)
        hi = np.asarray([group[name]["max"] for name in names], dtype=np.float32)
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise ValueError("Normalization statistics must be finite")
        if np.any(hi < lo):
            raise ValueError("Each statistic max must be greater than or equal to min")
        # Statistics written by older versions may represent a constant
        # channel as min == max.  Use a unit denominator for backward
        # compatibility and for constant static fields.
        degenerate = hi == lo
        if np.any(degenerate):
            hi = hi.copy()
            hi[degenerate] = lo[degenerate] + 1.0
        return lo, hi

    in_min, in_max = collect(in_stats, input_variables)
    out_min, out_max = collect(out_stats, output_variables)
    return in_min, in_max, out_min, out_max


class WindForecastDataset(DownscalingDataset):
    """Causal low-resolution wind forecasting dataset.

    Each item uses observations at ``t-history_hours+1 ... t`` and predicts the
    wind components at ``t+1``.  Samples never cross a calendar-year boundary.
    ``input_channels()`` reports physical channels; coordinate embedding
    channels are appended to the returned tensor and counted separately
    through :attr:`grid_channels`.
    """

    def __init__(
        self,
        low_res_files: Union[PathLike, Sequence[PathLike]],
        static_low_file: PathLike,
        input_variables: Sequence[str] = ("u10", "v10"),
        output_variables: Optional[Sequence[str]] = None,
        invariant_variables: Sequence[str] = ("elevation", "land_sea_mask"),
        stats_path: Optional[PathLike] = None,
        history_hours: int = 6,
        forecast_horizon_hours: int = 1,
        years: Optional[Sequence[int]] = None,
        grid_channels: int = 4,
        preload_low: bool = True,
        preload_dtype: str = "float16",
        normalize: bool = True,
        nan_policy: str = "error",
        engine: str = "netcdf4",
    ) -> None:
        super().__init__()
        if history_hours < 1:
            raise ValueError("history_hours must be >= 1")
        if forecast_horizon_hours != 1:
            raise ValueError("This dataset is intentionally one-hour-ahead only")
        if grid_channels not in (0, 4):
            raise ValueError("grid_channels must be 0 or 4")
        if preload_dtype not in ("float16", "float32"):
            raise ValueError("preload_dtype must be float16 or float32")

        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables or input_variables)
        self.invariant_variables = list(invariant_variables)
        if len(self.input_variables) != 2 or len(self.output_variables) != 2:
            raise ValueError("WindForecastDataset expects two wind components (u and v)")
        self.history_hours = int(history_hours)
        self.forecast_horizon_hours = int(forecast_horizon_hours)
        self.grid_channels = int(grid_channels)
        self.normalize = bool(normalize)
        self.nan_policy = nan_policy
        self.engine = engine
        self.preload_low = bool(preload_low)
        self.low_index = FileTimeIndex(as_path_list(low_res_files), engine=engine)
        self._lat = self.low_index.lat.copy()
        self._lon = self.low_index.lon.copy()
        self.grid = _grid_from_coords(self._lat, self._lon) if self.grid_channels else np.zeros((0, len(self._lat), len(self._lon)), np.float32)

        with xr.open_dataset(static_low_file, engine=engine) as static_ds:
            static = np.stack(
                [
                    _regrid_2d(static_ds[var], self._lat, self._lon)
                    for var in self.invariant_variables
                ],
                axis=0,
            )
        self.static = _sanitize(static, self.nan_policy, "static_low_file")
        self.physical_input_names = [
            name
            for _ in range(self.history_hours)
            for name in self.input_variables
        ] + self.invariant_variables
        self.channel_names = self.physical_input_names + (
            list(GRID_CHANNEL_NAMES[: self.grid_channels]) if self.grid_channels else []
        )

        selected_years = set(int(year) for year in years) if years is not None else None
        self.samples: List[Tuple[Tuple[int, ...], int, pd.Timestamp]] = []
        times = self.low_index.times
        for current in range(self.history_hours - 1, len(times) - self.forecast_horizon_hours):
            history = tuple(range(current - self.history_hours + 1, current + 1))
            target = current + self.forecast_horizon_hours
            valid_time = times[target]
            if selected_years is not None and valid_time.year not in selected_years:
                continue
            chain = [times[pos] for pos in (*history, target)]
            if len({timestamp.year for timestamp in chain}) != 1:
                continue
            if any((chain[i + 1] - chain[i]) != pd.Timedelta(hours=1) for i in range(len(chain) - 1)):
                continue
            self.samples.append((history, target, valid_time))
        if not self.samples:
            raise ValueError("No causal wind samples remain after the year and hourly checks")

        self.input_min, self.input_max, self.output_min, self.output_max = load_channel_stats(
            stats_path, self.physical_input_names, self.output_variables
        )
        self._low_data: Optional[np.ndarray] = None
        if self.preload_low:
            dtype = np.float16 if preload_dtype == "float16" else np.float32
            self._low_data = np.empty(
                (len(self.low_index), len(self.input_variables), len(self._lat), len(self._lon)),
                dtype=dtype,
            )
            grouped: Dict[Path, List[Tuple[int, int]]] = {}
            for global_pos, entry in enumerate(self.low_index.entries):
                grouped.setdefault(entry.path, []).append((global_pos, entry.local_index))
            for path, members in grouped.items():
                # Keep each file open once while limiting the temporary batch size.
                with xr.open_dataset(path, engine=engine) as ds:
                    for offset in range(0, len(members), 1024):
                        chunk = members[offset : offset + 1024]
                        local_indices = [local for _, local in chunk]
                        global_positions = [global_pos for global_pos, _ in chunk]
                        for channel, name in enumerate(self.input_variables):
                            values = _regrid_batch(
                                ds[name].isel(time=local_indices), self._lat, self._lon
                            )
                            self._low_data[global_positions, channel] = values

    def __len__(self) -> int:
        return len(self.samples)

    def latitude(self) -> np.ndarray:
        return self._lat.copy()

    def longitude(self) -> np.ndarray:
        return self._lon.copy()

    def image_shape(self) -> Tuple[int, int]:
        return (len(self._lat), len(self._lon))

    def low_res_shape(self) -> Tuple[int, int]:
        return self.image_shape()

    def input_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=name) for name in self.physical_input_names]

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=name) for name in self.output_variables]

    def time(self) -> List[pd.Timestamp]:
        return [pd.Timestamp(sample[2]) for sample in self.samples]

    def sample_metadata(self, index: int) -> Dict[str, Any]:
        history, target, valid_time = self.samples[index]
        return {
            "history_times": [self.low_index.times[pos] for pos in history],
            "initial_time": self.low_index.times[history[-1]],
            "target_time": valid_time,
            "target_position": target,
        }

    def read_physical_times(self, positions: Sequence[int]) -> np.ndarray:
        if self._low_data is not None:
            values = self._low_data[np.asarray(positions, dtype=np.int64)].astype(np.float32)
        else:
            values = self.low_index.read_variables_many(
                self.input_variables, positions, self._lat, self._lon, self.engine
            )
        return _sanitize(values, self.nan_policy, "low_res wind")

    def build_input_from_history(self, history_fields: np.ndarray) -> np.ndarray:
        fields = np.asarray(history_fields, dtype=np.float32)
        expected = (self.history_hours, len(self.input_variables), len(self._lat), len(self._lon))
        if fields.shape != expected:
            raise ValueError(f"Expected history fields {expected}, got {fields.shape}")
        x = np.concatenate([fields.reshape(-1, *fields.shape[-2:]), self.static], axis=0)
        if self.grid_channels:
            x = np.concatenate([x, self.grid], axis=0)
        return x

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(x, dtype=np.float32)
        out = np.asarray(x, dtype=np.float32).copy()
        if out.ndim < 3:
            raise ValueError(
                "Expected input shape (..., channels, height, width), "
                f"got {out.shape}"
            )
        physical = len(self.input_min)
        channel_axis = out.ndim - 3
        selection = (slice(None),) * channel_axis + (slice(0, physical),)
        stats_shape = (
            (1,) * channel_axis
            + (physical,)
            + (1,) * (out.ndim - channel_axis - 1)
        )
        out[selection] = (
            out[selection] - self.input_min.reshape(stats_shape)
        ) / (self.input_max - self.input_min).reshape(stats_shape)
        return out

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(x, dtype=np.float32)
        out = np.asarray(x, dtype=np.float32).copy()
        if out.ndim < 3:
            raise ValueError(
                "Expected input shape (..., channels, height, width), "
                f"got {out.shape}"
            )
        physical = len(self.input_min)
        channel_axis = out.ndim - 3
        selection = (slice(None),) * channel_axis + (slice(0, physical),)
        stats_shape = (
            (1,) * channel_axis
            + (physical,)
            + (1,) * (out.ndim - channel_axis - 1)
        )
        out[selection] = (
            out[selection]
            * (self.input_max - self.input_min).reshape(stats_shape)
            + self.input_min.reshape(stats_shape)
        )
        return out

    def normalize_output(self, y: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(y, dtype=np.float32)
        return (np.asarray(y, dtype=np.float32) - self.output_min[:, None, None]) / (self.output_max - self.output_min)[:, None, None]

    def denormalize_output(self, y: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(y, dtype=np.float32)
        return np.asarray(y, dtype=np.float32) * (self.output_max - self.output_min)[:, None, None] + self.output_min[:, None, None]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        history, target, _valid_time = self.samples[index]
        x_phys = self.build_input_from_history(self.read_physical_times(history))
        y_phys = _sanitize(self.read_physical_times((target,))[0], self.nan_policy, "forecast target")
        x = self.normalize_input(x_phys)
        y = self.normalize_output(y_phys)
        return torch.from_numpy(y.copy()).float(), torch.from_numpy(x.copy()).float()


@dataclass(frozen=True)
class ConditionEntry:
    path: Path
    local_initial: int
    local_lead: int
    lead_hour: int
    initial_time: pd.Timestamp
    valid_time: pd.Timestamp


class ConditionStore:
    """Index a Step-1 forecast file with ``initial_time x lead_hour`` fields."""

    def __init__(
        self,
        files: Sequence[PathLike],
        sample_variables: Sequence[str],
        leads: Sequence[int],
        sampling: str,
        engine: str = "netcdf4",
    ) -> None:
        self.files = as_path_list(files)
        if not self.files:
            raise ValueError("At least one Step-1 condition file is required")
        allowed_leads = {int(lead) for lead in leads}
        self.sample_variables = list(sample_variables)
        entries: List[ConditionEntry] = []
        first_coords: Optional[Tuple[np.ndarray, np.ndarray]] = None
        for path in self.files:
            with xr.open_dataset(path, engine=engine) as ds:
                missing_variables = [name for name in self.sample_variables if name not in ds]
                if missing_variables:
                    raise KeyError(f"{path} is missing condition variables {missing_variables}")
                var = ds[self.sample_variables[0]]
                init_dim = "initial_time" if "initial_time" in var.dims else "time"
                lead_dim = next((name for name in ("lead_hour", "lead") if name in var.dims), None)
                if lead_dim is None:
                    raise ValueError(f"{path} must contain a lead_hour or lead dimension")
                initial_times = parse_hourly_times(ds[init_dim].values)
                file_leads = [int(value) for value in np.asarray(ds[lead_dim].values).reshape(-1)]
                coords = (_coord_values(ds, "lat"), _coord_values(ds, "lon"))
                if first_coords is None:
                    first_coords = coords
                elif not (
                    np.allclose(first_coords[0], coords[0], rtol=0.0, atol=1e-6)
                    and np.allclose(first_coords[1], coords[1], rtol=0.0, atol=1e-6)
                ):
                    raise ValueError(f"Condition grid coordinates in {path} differ from the first file")
                for local_initial, initial_time in enumerate(initial_times):
                    for local_lead, lead in enumerate(file_leads):
                        if lead not in allowed_leads:
                            continue
                        if sampling == "hybrid" and lead != 1 and initial_time.hour != 0:
                            continue
                        if sampling == "daily00" and initial_time.hour != 0:
                            continue
                        if sampling not in ("all", "hybrid", "daily00"):
                            raise ValueError("condition_sampling must be all, hybrid, or daily00")
                        entries.append(
                            ConditionEntry(
                                path=path,
                                local_initial=local_initial,
                                local_lead=local_lead,
                                lead_hour=lead,
                                initial_time=initial_time,
                                valid_time=initial_time + pd.Timedelta(hours=lead),
                            )
                        )
        entries.sort(key=lambda item: (item.initial_time, item.lead_hour))
        assert first_coords is not None
        self.entries = entries
        self.lat, self.lon = first_coords

    def __len__(self) -> int:
        return len(self.entries)

    def read(self, variable: str, entry: ConditionEntry, lat: np.ndarray, lon: np.ndarray, engine: str) -> np.ndarray:
        ds = _open_cached_dataset(str(entry.path), engine)
        var = ds[variable]
        init_dim = "initial_time" if "initial_time" in var.dims else "time"
        lead_dim = "lead_hour" if "lead_hour" in var.dims else "lead"
        da = _drop_singleton_height(
            var.isel({init_dim: entry.local_initial, lead_dim: entry.local_lead})
        )
        return _regrid_2d(da, lat, lon)


@lru_cache(maxsize=32)
def _open_cached_dataset(path: str, engine: str) -> xr.Dataset:
    """Reuse read-only NetCDF handles inside each DataLoader worker process."""
    return xr.open_dataset(path, engine=engine)


def clear_cached_datasets() -> None:
    """Clear process-local NetCDF handles (useful in tests)."""
    _open_cached_dataset.cache_clear()


def _trim_indices(indices: np.ndarray, multiple: int) -> Tuple[int, int]:
    start, end = int(indices.min()), int(indices.max()) + 1
    if multiple > 1:
        end = start + ((end - start) // multiple) * multiple
    return start, end


class WindSuperResolutionDataset(DownscalingDataset):
    """Step-2 CorrDiff dataset for Step-1 wind conditions and ICON targets."""

    def __init__(
        self,
        condition_files: Union[PathLike, Sequence[PathLike]],
        high_res_files: Union[PathLike, Sequence[PathLike]],
        static_high_file: PathLike,
        low_res_variables: Sequence[str] = ("u10", "v10"),
        high_res_variables: Sequence[str] = ("u_10m", "v_10m"),
        output_variables: Sequence[str] = ("u10", "v10"),
        invariant_variables: Sequence[str] = ("elevation", "land_sea_mask"),
        stats_path: Optional[PathLike] = None,
        leads: Sequence[int] = (1, 2, 3, 4, 5, 6),
        condition_sampling: str = "hybrid",
        crop_mult: int = 16,
        grid_channels: int = 4,
        normalize: bool = True,
        nan_policy: str = "error",
        engine: str = "netcdf4",
    ) -> None:
        super().__init__()
        if grid_channels not in (0, 4):
            raise ValueError("grid_channels must be 0 or 4")
        self.low_res_variables = list(low_res_variables)
        self.high_res_variables = list(high_res_variables)
        self.output_variables = list(output_variables)
        self.invariant_variables = list(invariant_variables)
        if not (len(self.low_res_variables) == len(self.high_res_variables) == len(self.output_variables) == 2):
            raise ValueError("WindSuperResolutionDataset expects two wind components")
        self.leads = tuple(int(lead) for lead in leads)
        self.normalize = bool(normalize)
        self.nan_policy = nan_policy
        self.engine = engine
        self.grid_channels = int(grid_channels)
        self.condition_store = ConditionStore(
            as_path_list(condition_files),
            self.low_res_variables,
            self.leads,
            condition_sampling,
            engine,
        )
        self.high_index = FileTimeIndex(as_path_list(high_res_files), engine=engine)

        hr_lat, hr_lon = self.high_index.lat, self.high_index.lon
        lat_indices = np.flatnonzero((hr_lat >= self.condition_store.lat.min()) & (hr_lat <= self.condition_store.lat.max()))
        lon_indices = np.flatnonzero((hr_lon >= self.condition_store.lon.min()) & (hr_lon <= self.condition_store.lon.max()))
        if not len(lat_indices) or not len(lon_indices):
            raise ValueError("No spatial overlap between Step-1 conditions and high-res grid")
        self.lat_start, self.lat_end = _trim_indices(lat_indices, crop_mult)
        self.lon_start, self.lon_end = _trim_indices(lon_indices, crop_mult)
        self._lat = hr_lat[self.lat_start : self.lat_end].copy()
        self._lon = hr_lon[self.lon_start : self.lon_end].copy()
        self.grid = _grid_from_coords(self._lat, self._lon) if self.grid_channels else np.zeros((0, len(self._lat), len(self._lon)), np.float32)

        with xr.open_dataset(static_high_file, engine=engine) as static_ds:
            static = np.stack(
                [
                    _regrid_2d(static_ds[var], self._lat, self._lon)
                    for var in self.invariant_variables
                ],
                axis=0,
            )
        self.static = _sanitize(static, self.nan_policy, "static_high_file")
        self.physical_input_names = self.low_res_variables + self.invariant_variables
        self.channel_names = self.physical_input_names + (
            list(GRID_CHANNEL_NAMES[: self.grid_channels]) if self.grid_channels else []
        )

        self.samples: List[Tuple[ConditionEntry, int]] = []
        missing_valid_times = []
        for entry in self.condition_store.entries:
            if entry.initial_time.year != entry.valid_time.year:
                continue
            high_position = self.high_index.time_to_position.get(entry.valid_time)
            if high_position is not None:
                self.samples.append((entry, high_position))
            else:
                missing_valid_times.append(entry.valid_time)
        if not self.samples:
            raise ValueError("No Step-1 condition/ICON target pairs share a valid time")
        if missing_valid_times:
            examples = ", ".join(str(value) for value in missing_valid_times[:3])
            warnings.warn(
                f"Skipped {len(missing_valid_times)} Step-1 conditions with no matching high-resolution "
                f"valid time (first examples: {examples})",
                RuntimeWarning,
                stacklevel=2,
            )

        self.input_min, self.input_max, self.output_min, self.output_max = load_channel_stats(
            stats_path, self.physical_input_names, self.output_variables
        )

    def __len__(self) -> int:
        return len(self.samples)

    def latitude(self) -> np.ndarray:
        return self._lat.copy()

    def longitude(self) -> np.ndarray:
        return self._lon.copy()

    def image_shape(self) -> Tuple[int, int]:
        return (len(self._lat), len(self._lon))

    def low_res_shape(self) -> Tuple[int, int]:
        return (len(self.condition_store.lat), len(self.condition_store.lon))

    def input_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=name) for name in self.physical_input_names]

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=name) for name in self.output_variables]

    def time(self) -> List[pd.Timestamp]:
        return [entry.valid_time for entry, _ in self.samples]

    def sample_metadata(self, index: int) -> Dict[str, Any]:
        entry, high_position = self.samples[index]
        return {
            "initial_time": entry.initial_time,
            "lead_hour": entry.lead_hour,
            "target_time": entry.valid_time,
            "high_position": high_position,
        }

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(x, dtype=np.float32)
        out = np.asarray(x, dtype=np.float32).copy()
        physical = len(self.input_min)
        out[:physical] = (out[:physical] - self.input_min[:, None, None]) / (self.input_max - self.input_min)[:, None, None]
        return out

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(x, dtype=np.float32)
        out = np.asarray(x, dtype=np.float32).copy()
        physical = len(self.input_min)
        out[:physical] = out[:physical] * (self.input_max - self.input_min)[:, None, None] + self.input_min[:, None, None]
        return out

    def normalize_output(self, y: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(y, dtype=np.float32)
        return (np.asarray(y, dtype=np.float32) - self.output_min[:, None, None]) / (self.output_max - self.output_min)[:, None, None]

    def denormalize_output(self, y: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(y, dtype=np.float32)
        return np.asarray(y, dtype=np.float32) * (self.output_max - self.output_min)[:, None, None] + self.output_min[:, None, None]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry, high_position = self.samples[index]
        dynamic = np.stack(
            [
                self.condition_store.read(name, entry, self._lat, self._lon, self.engine)
                for name in self.low_res_variables
            ],
            axis=0,
        )
        x_phys = np.concatenate([dynamic, self.static], axis=0)
        if self.grid_channels:
            x_phys = np.concatenate([x_phys, self.grid], axis=0)

        high_path = self.high_index.entries[high_position].path
        high_ds = _open_cached_dataset(str(high_path), self.engine)
        target = np.stack(
            [
                _regrid_2d(
                    _select_time(high_ds, "time", self.high_index.entries[high_position].local_index, name),
                    self.high_index.lat,
                    self.high_index.lon,
                )[
                    self.lat_start : self.lat_end,
                    self.lon_start : self.lon_end,
                ]
                for name in self.high_res_variables
            ],
            axis=0,
        )
        x_phys = _sanitize(x_phys, self.nan_policy, "Step-2 input")
        target = _sanitize(target, self.nan_policy, "high-res wind target")
        x = self.normalize_input(x_phys)
        y = self.normalize_output(target)
        return torch.from_numpy(y.copy()).float(), torch.from_numpy(x.copy()).float()


__all__ = [
    "WindForecastDataset",
    "WindSuperResolutionDataset",
    "parse_hourly_times",
    "load_channel_stats",
    "GRID_CHANNEL_NAMES",
]
