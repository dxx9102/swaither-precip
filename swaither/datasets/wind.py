"""Datasets and normalization helpers for the two-model wind pipeline.

The regression dataset maps six hours of low-resolution wind history to 72
hours of high-resolution absolute wind.  The diffusion view exposes the same
initial times as ``(initial_time, lead)`` pairs and lets diffusion learn the
residual left by the frozen regression U-Net.
"""

from __future__ import annotations

import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
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
LEAD_CHANNEL_NAMES = ("lead_sin", "lead_cos")


def as_path_list(value: Union[PathLike, Sequence[PathLike], None]) -> List[Path]:
    """Normalize a scalar/list path configuration to an ordered path list."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [Path(value)]
    return [Path(item) for item in value]


def parse_hourly_times(values: Iterable[Any]) -> pd.DatetimeIndex:
    """Parse ICON fractional-day or ISO-like time coordinates at exact hours."""
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
                parsed = pd.DatetimeIndex(pd.to_datetime(strings, utc=True))
            except Exception as exc:  # pragma: no cover - defensive error path
                bad = next(
                    (
                        value
                        for value in strings
                        if pd.isna(pd.to_datetime(value, errors="coerce"))
                    ),
                    strings[0],
                )
                raise ValueError(f"Unsupported time value {bad!r}") from exc
            aligned = parsed.floor("h")
            if np.any(parsed != aligned):
                first = parsed[np.flatnonzero(parsed != aligned)[0]]
                raise ValueError(
                    f"Time coordinate {first!r} is not aligned to an exact hour"
                )
            return aligned.tz_convert(None)
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
        raise KeyError(f"Required coordinate {name!r} is missing")
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


def _regrid_2d(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected lat/lon dimensions, got {da.dims}")
    old_lat = np.asarray(da.lat.values)
    old_lon = np.asarray(da.lon.values)
    same_grid = da.sizes["lat"] == len(lat) and da.sizes["lon"] == len(lon)
    same_grid = same_grid and np.allclose(old_lat, lat, rtol=0.0, atol=1e-6)
    same_grid = same_grid and np.allclose(old_lon, lon, rtol=0.0, atol=1e-6)
    if not same_grid:
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
    """Lazy, sorted index over one or more NetCDF files sharing one grid."""

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

    def read_variables_many(
        self,
        variables: Sequence[str],
        positions: Sequence[int],
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        engine: str = "netcdf4",
    ) -> np.ndarray:
        """Read variables and times while opening each source file only once."""
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
        return (
            np.zeros(n_in, np.float32),
            np.ones(n_in, np.float32),
            np.zeros(n_out, np.float32),
            np.ones(n_out, np.float32),
        )
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
        degenerate = hi == lo
        if np.any(degenerate):
            hi = hi.copy()
            hi[degenerate] = lo[degenerate] + 1.0
        return lo, hi

    in_min, in_max = collect(in_stats, input_variables)
    out_min, out_max = collect(out_stats, output_variables)
    return in_min, in_max, out_min, out_max


class WindForecastDataset(DownscalingDataset):
    """Six-hour low-resolution history to 72-hour high-resolution regression.

    ``__getitem__`` returns ``(y, x)``.  ``y`` is flattened in lead-major
    ``(72 * 2, H, W)`` order.  ``x`` contains 12 history channels, two HR
    static fields, and four coordinate channels.
    """

    def __init__(
        self,
        low_res_files: Union[PathLike, Sequence[PathLike]],
        high_res_files: Union[PathLike, Sequence[PathLike]],
        static_high_file: PathLike,
        low_res_variables: Sequence[str] = ("u10", "v10"),
        high_res_variables: Sequence[str] = ("u_10m", "v_10m"),
        output_variables: Sequence[str] = ("u10", "v10"),
        invariant_variables: Sequence[str] = ("elevation", "land_sea_mask"),
        stats_path: Optional[PathLike] = None,
        history_hours: int = 6,
        forecast_horizon_hours: int = 72,
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
        if forecast_horizon_hours < 1:
            raise ValueError("forecast_horizon_hours must be >= 1")
        if grid_channels not in (0, 4):
            raise ValueError("grid_channels must be 0 or 4")
        if preload_dtype not in ("float16", "float32"):
            raise ValueError("preload_dtype must be float16 or float32")

        self.low_res_variables = list(low_res_variables)
        self.high_res_variables = list(high_res_variables)
        self.output_variables = list(output_variables)
        self.invariant_variables = list(invariant_variables)
        if not (
            len(self.low_res_variables)
            == len(self.high_res_variables)
            == len(self.output_variables)
            == 2
        ):
            raise ValueError("WindForecastDataset expects two wind components")
        self.history_hours = int(history_hours)
        self.forecast_horizon_hours = int(forecast_horizon_hours)
        self.grid_channels = int(grid_channels)
        self.normalize = bool(normalize)
        self.nan_policy = nan_policy
        self.engine = engine
        self.preload_low = bool(preload_low)

        self.low_index = FileTimeIndex(as_path_list(low_res_files), engine=engine)
        self.high_index = FileTimeIndex(as_path_list(high_res_files), engine=engine)
        if (
            self.high_index.lat.min() < self.low_index.lat.min()
            or self.high_index.lat.max() > self.low_index.lat.max()
            or self.high_index.lon.min() < self.low_index.lon.min()
            or self.high_index.lon.max() > self.low_index.lon.max()
        ):
            raise ValueError("The high-resolution grid must lie within the low-resolution domain")

        self._lat = self.high_index.lat.copy()
        self._lon = self.high_index.lon.copy()
        self.grid = (
            _grid_from_coords(self._lat, self._lon)
            if self.grid_channels
            else np.zeros((0, len(self._lat), len(self._lon)), np.float32)
        )
        with xr.open_dataset(static_high_file, engine=engine) as static_ds:
            static = np.stack(
                [
                    _regrid_2d(static_ds[var], self._lat, self._lon)
                    for var in self.invariant_variables
                ],
                axis=0,
            )
        self.static = _sanitize(static, self.nan_policy, "static_high_file")
        self.physical_input_names = [
            name
            for _ in range(self.history_hours)
            for name in self.low_res_variables
        ] + self.invariant_variables
        self.channel_names = self.physical_input_names + (
            list(GRID_CHANNEL_NAMES[: self.grid_channels]) if self.grid_channels else []
        )

        selected_years = set(int(year) for year in years) if years is not None else None
        self.samples: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        skipped_targets = 0
        times = self.low_index.times
        for current in range(self.history_hours - 1, len(times) - self.forecast_horizon_hours):
            history = tuple(range(current - self.history_hours + 1, current + 1))
            history_times = [times[position] for position in history]
            initial_time = history_times[-1]
            if selected_years is not None and initial_time.year not in selected_years:
                continue
            if any(
                (history_times[index + 1] - history_times[index]) != pd.Timedelta(hours=1)
                for index in range(len(history_times) - 1)
            ):
                continue
            target_times = [
                initial_time + pd.Timedelta(hours=lead)
                for lead in range(1, self.forecast_horizon_hours + 1)
            ]
            if len({timestamp.year for timestamp in (*history_times, *target_times)}) != 1:
                continue
            target_positions = tuple(
                self.high_index.time_to_position.get(timestamp, -1)
                for timestamp in target_times
            )
            if any(position < 0 for position in target_positions):
                skipped_targets += 1
                continue
            self.samples.append((history, target_positions))
        if not self.samples:
            raise ValueError("No complete same-year 6-to-72-hour wind samples remain")
        if skipped_targets:
            warnings.warn(
                f"Skipped {skipped_targets} initial times without a complete high-resolution target window",
                RuntimeWarning,
                stacklevel=2,
            )

        self.input_min, self.input_max, self.output_min, self.output_max = load_channel_stats(
            stats_path, self.physical_input_names, self.output_variables
        )
        self._low_data: Optional[np.ndarray] = None
        if self.preload_low:
            dtype = np.float16 if preload_dtype == "float16" else np.float32
            self._low_data = np.empty(
                (
                    len(self.low_index),
                    len(self.low_res_variables),
                    len(self.low_index.lat),
                    len(self.low_index.lon),
                ),
                dtype=dtype,
            )
            grouped: Dict[Path, List[Tuple[int, int]]] = {}
            for global_pos, entry in enumerate(self.low_index.entries):
                grouped.setdefault(entry.path, []).append((global_pos, entry.local_index))
            for path, members in grouped.items():
                with xr.open_dataset(path, engine=engine) as ds:
                    for offset in range(0, len(members), 1024):
                        chunk = members[offset : offset + 1024]
                        local_indices = [local for _, local in chunk]
                        global_positions = [global_pos for global_pos, _ in chunk]
                        for channel, name in enumerate(self.low_res_variables):
                            values = _regrid_batch(
                                ds[name].isel(time=local_indices),
                                self.low_index.lat,
                                self.low_index.lon,
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
        return (len(self.low_index.lat), len(self.low_index.lon))

    def input_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=name) for name in self.physical_input_names]

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=name) for name in self.output_variables]

    def time(self) -> List[pd.Timestamp]:
        return [self.low_index.times[history[-1]] for history, _ in self.samples]

    def sample_metadata(self, index: int) -> Dict[str, Any]:
        history, targets = self.samples[index]
        return {
            "history_times": [self.low_index.times[position] for position in history],
            "initial_time": self.low_index.times[history[-1]],
            "target_times": [self.high_index.times[position] for position in targets],
            "lead_hours": list(range(1, self.forecast_horizon_hours + 1)),
        }

    def read_physical_history(self, index: int) -> np.ndarray:
        history, _ = self.samples[index]
        if self._low_data is not None:
            values = self._low_data[np.asarray(history, dtype=np.int64)].astype(np.float32)
        else:
            values = self.low_index.read_variables_many(
                self.low_res_variables,
                history,
                self.low_index.lat,
                self.low_index.lon,
                self.engine,
            )
        values = _sanitize(values, self.nan_policy, "low-res wind history")
        fields = values.reshape(-1, *self.low_res_shape())
        da = xr.DataArray(
            fields,
            dims=("field", "lat", "lon"),
            coords={"lat": self.low_index.lat, "lon": self.low_index.lon},
        )
        regridded = da.interp(lat=self._lat, lon=self._lon, method="linear")
        return np.asarray(regridded.values, dtype=np.float32).reshape(
            self.history_hours,
            len(self.low_res_variables),
            *self.image_shape(),
        )

    def read_physical_targets(self, index: int) -> np.ndarray:
        _, targets = self.samples[index]
        values = self.high_index.read_variables_many(
            self.high_res_variables,
            targets,
            self._lat,
            self._lon,
            self.engine,
        )
        return _sanitize(values, self.nan_policy, "high-res wind target")

    def read_physical_target(self, index: int, lead_index: int) -> np.ndarray:
        if not 0 <= int(lead_index) < self.forecast_horizon_hours:
            raise IndexError("lead_index is outside the forecast horizon")
        _, targets = self.samples[index]
        values = self.high_index.read_variables_many(
            self.high_res_variables,
            (targets[int(lead_index)],),
            self._lat,
            self._lon,
            self.engine,
        )[0]
        return _sanitize(values, self.nan_policy, "high-res wind target")

    def build_input_from_history(self, history_fields: np.ndarray) -> np.ndarray:
        fields = np.asarray(history_fields, dtype=np.float32)
        expected = (
            self.history_hours,
            len(self.low_res_variables),
            len(self._lat),
            len(self._lon),
        )
        if fields.shape != expected:
            raise ValueError(f"Expected history fields {expected}, got {fields.shape}")
        x = np.concatenate([fields.reshape(-1, *fields.shape[-2:]), self.static], axis=0)
        if self.grid_channels:
            x = np.concatenate([x, self.grid], axis=0)
        return x

    def condition(self, index: int) -> np.ndarray:
        history = self.read_physical_history(index)
        return self.normalize_input(self.build_input_from_history(history))

    def target(self, index: int) -> np.ndarray:
        targets = self.read_physical_targets(index)
        normalized = self.normalize_output(targets)
        return normalized.reshape(
            self.forecast_horizon_hours * len(self.output_variables),
            *self.image_shape(),
        )

    def target_at(self, index: int, lead_index: int) -> np.ndarray:
        return self.normalize_output(self.read_physical_target(index, lead_index))

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(x, dtype=np.float32)
        values = np.asarray(x, dtype=np.float32)
        physical = len(self.input_min)
        expected = physical + self.grid_channels
        if values.shape[-3] != expected:
            raise ValueError(
                f"Expected {expected} input channels, got {values.shape[-3]}"
            )
        stats_shape = (1,) * (values.ndim - 3) + (physical, 1, 1)
        normalized = values.copy()
        normalized[..., :physical, :, :] = (
            values[..., :physical, :, :] - self.input_min.reshape(stats_shape)
        ) / (
            self.input_max - self.input_min
        ).reshape(stats_shape)
        return normalized

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(x, dtype=np.float32)
        out = np.asarray(x, dtype=np.float32).copy()
        physical = len(self.input_min)
        stats_shape = (1,) * (out.ndim - 3) + (physical, 1, 1)
        out[..., :physical, :, :] = out[..., :physical, :, :] * (
            self.input_max - self.input_min
        ).reshape(stats_shape) + self.input_min.reshape(stats_shape)
        return out

    def normalize_output(self, y: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(y, dtype=np.float32)
        values = np.asarray(y, dtype=np.float32)
        if values.shape[-3] != len(self.output_min):
            raise ValueError(
                f"Expected {len(self.output_min)} output channels, got {values.shape[-3]}"
            )
        stats_shape = (1,) * (values.ndim - 3) + (len(self.output_min), 1, 1)
        return (values - self.output_min.reshape(stats_shape)) / (
            self.output_max - self.output_min
        ).reshape(stats_shape)

    def denormalize_output(self, y: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return np.asarray(y, dtype=np.float32)
        values = np.asarray(y, dtype=np.float32)
        if values.shape[-3] != len(self.output_min):
            raise ValueError(
                f"Expected {len(self.output_min)} output channels, got {values.shape[-3]}"
            )
        stats_shape = (1,) * (values.ndim - 3) + (len(self.output_min), 1, 1)
        return values * (self.output_max - self.output_min).reshape(stats_shape) + self.output_min.reshape(
            stats_shape
        )

    def forecast_fields(self, flat_prediction: np.ndarray) -> np.ndarray:
        values = np.asarray(flat_prediction, dtype=np.float32)
        channels = self.forecast_horizon_hours * len(self.output_variables)
        if values.shape[-3] != channels:
            raise ValueError(f"Expected {channels} forecast channels, got {values.shape[-3]}")
        return values.reshape(
            *values.shape[:-3],
            self.forecast_horizon_hours,
            len(self.output_variables),
            *self.image_shape(),
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.target(index)
        x = self.condition(index)
        return torch.from_numpy(y.copy()).float(), torch.from_numpy(x.copy()).float()


class WindDiffusionDataset(DownscalingDataset):
    """Lead-expanded view used to train residual diffusion."""

    def __init__(self, dataset: WindForecastDataset, leads: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.dataset = dataset
        selected = (
            list(range(1, dataset.forecast_horizon_hours + 1))
            if leads is None
            else [int(lead) for lead in leads]
        )
        if not selected or any(lead < 1 or lead > dataset.forecast_horizon_hours for lead in selected):
            raise ValueError("Diffusion leads must be within the forecast horizon")
        if len(set(selected)) != len(selected):
            raise ValueError("Diffusion leads must be unique")
        self.leads = tuple(selected)

    def __len__(self) -> int:
        return len(self.dataset) * len(self.leads)

    def latitude(self) -> np.ndarray:
        return self.dataset.latitude()

    def longitude(self) -> np.ndarray:
        return self.dataset.longitude()

    def image_shape(self) -> Tuple[int, int]:
        return self.dataset.image_shape()

    def input_channels(self) -> List[ChannelMetadata]:
        return self.dataset.input_channels() + [
            ChannelMetadata(name=name) for name in LEAD_CHANNEL_NAMES
        ]

    def output_channels(self) -> List[ChannelMetadata]:
        return self.dataset.output_channels()

    def time(self) -> List[pd.Timestamp]:
        result: List[pd.Timestamp] = []
        for timestamp in self.dataset.time():
            result.extend([timestamp] * len(self.leads))
        return result

    def sample_metadata(self, index: int) -> Dict[str, Any]:
        base_index, lead_position = divmod(int(index), len(self.leads))
        metadata = dict(self.dataset.sample_metadata(base_index))
        lead = self.leads[lead_position]
        metadata.update(
            {
                "base_index": base_index,
                "lead_hour": lead,
                "valid_time": metadata["initial_time"] + pd.Timedelta(hours=lead),
            }
        )
        return metadata

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_index, lead_position = divmod(int(index), len(self.leads))
        lead = self.leads[lead_position]
        y = self.dataset.target_at(base_index, lead - 1)
        x = self.dataset.condition(base_index)
        phase = 2.0 * np.pi * float(lead - 1) / float(self.dataset.forecast_horizon_hours)
        lead_encoding = np.asarray(
            [
                np.full(self.dataset.image_shape(), np.sin(phase), np.float32),
                np.full(self.dataset.image_shape(), np.cos(phase), np.float32),
            ]
        )
        x = np.concatenate([x, lead_encoding], axis=0)
        return (
            torch.from_numpy(y.copy()).float(),
            torch.from_numpy(x.copy()).float(),
            torch.tensor(lead, dtype=torch.long),
        )


__all__ = [
    "WindForecastDataset",
    "WindDiffusionDataset",
    "FileTimeIndex",
    "parse_hourly_times",
    "load_channel_stats",
    "GRID_CHANNEL_NAMES",
    "LEAD_CHANNEL_NAMES",
]
