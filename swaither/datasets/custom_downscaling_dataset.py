import json
import xarray as xr
import numpy as np
import torch
from typing import List, Tuple, Optional

import pandas as pd  # for time indexing and timedelta operations

import sys
import os
# --- Hack: add 'datasets' and 'helpers' dirs to sys.path to import base and helpers ---
CURRENT_DIR = os.path.dirname(__file__)
DATASETS_DIR = CURRENT_DIR
HELPERS_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'helpers'))
if DATASETS_DIR not in sys.path:
    sys.path.insert(0, DATASETS_DIR)
if HELPERS_DIR not in sys.path:
    sys.path.insert(0, HELPERS_DIR)

# Now imports resolve without hyphens or package names
try:
    from base import DownscalingDataset, ChannelMetadata
    from generate_helpers import convert_datetime_to_cftime
except ImportError as e:
    raise ImportError(f"Could not import dataset base classes: {e}")

def _nan_to_num_inplace(a: np.ndarray, clip_abs: float = 1e6):
    """Replace NaN->0, ±Inf->±clip_abs, in-place."""
    np.nan_to_num(a, copy=False, nan=0.0, posinf=clip_abs, neginf=-clip_abs)
    return a

def _per_channel_clip(a: np.ndarray, hi: float = None):
    """Optional: clip extreme outliers per-tensor. `a` is (C,H,W)."""
    if hi is not None:
        np.clip(a, a_min=None, a_max=hi, out=a)
    return a

# With extra time index storing! Useul for temporal regularization (all leads should have a similar prediction) 
class CustomDownscalingDatasetWithLeadFast_Time(DownscalingDataset):
    """
    Fast CorrDiff-compatible dataset with optional low-res→high-res interpolation
    and optional invariant (static) variables such as altitude.

    Expected dims in NetCDF:
      low_res:        time, prediction_delta, lat, lon   (for forecast channels)
      high_res:       time, prediction_delta?, lat, lon  (target may or may not have lead)
      static vars:    lat, lon                           (e.g. altitude), possibly only on one grid

    Behaviour:
      - interp_low_res = False:
          Assumes low_res grid is already same as high_res grid.
      - interp_low_res = True:
          Interpolates entire low_res dataset onto high_res lat/lon once in __init__,
          then preloads into NumPy arrays.

    Internally:
      low_data  : (T, L, C_dyn,  H, W)   dynamic low-res inputs (per-lead)
      high_data : (T, L, C_out,  H, W)   targets (per-lead)
      inv_data  : (C_inv,        H, W)   invariant (static) inputs (no time/lead)

    __getitem__(idx) returns:
      y_t        : (C_out,            H, W)  float32 (normalized if normalize=True)
      x_t        : (C_dyn + C_inv,    H, W)  float32 (normalized if normalize=True)
      lead_idx   : (1,) torch.long — discrete index of the lead (0..L-1)
      time_id    : (1,) torch.long — discrete ID of the *valid time group*

    Notes on time_id:
      - All samples that share the same valid time (same target precip time) will share
        the same integer time_id, regardless of lead.
      - time_id ranges from 0 .. (n_unique_times - 1).
    """

    def __init__(
        self,
        low_res_file: str,
        high_res_file: str,
        input_variables: List[str],
        output_variables: List[str],
        stats_path: Optional[str] = None,
        engine: str = "netcdf4",
        data_path: Optional[str] = None,
        log_precip: bool = False,
        crop_mult: int = 16,
        normalize: bool = True,
        interp_low_res: bool = False,
        # NEW: static fields such as altitude, snow fraction, land/sea mask, etc.
        invariant_variables: Optional[List[str]] = None,
        **kwargs,
    ):
        # ----------------- flags & config -----------------
        self.log_precip     = log_precip
        self.normalize      = normalize
        self.interp_low_res = interp_low_res

        # dynamic vs invariant inputs
        self.dynamic_input_variables = input_variables
        self.invariant_variables     = invariant_variables or []
        # All input channels as seen by the model & stats
        self.input_variables  = self.dynamic_input_variables + self.invariant_variables
        self.output_variables = output_variables

        # ----------------- 1) open datasets -----------------
        self.low_ds  = xr.open_dataset(low_res_file,  engine=engine, decode_timedelta=True)
        self.high_ds = xr.open_dataset(high_res_file, engine=engine, decode_timedelta=True)

        # ----------------- 2) time & lead info -----------------
        self.low_times   = pd.DatetimeIndex(self.low_ds.time.values)
        self.valid_times = pd.DatetimeIndex(self.high_ds.time.values)
        times_common     = sorted(set(self.low_times).intersection(set(self.valid_times)))

        if len(times_common) == 0:
            raise ValueError("No overlapping times between low_res and high_res datasets!")

        # store the common times in a fixed order
        self.times_common = pd.DatetimeIndex(times_common)  # length T_common

        # prediction_delta / lead values from *low* ds
        self.lead_values = self.low_ds.prediction_delta.values  # (L,) timedelta64
        self.lead_hours  = (self.lead_values / np.timedelta64(1, "h")).astype(np.float32)

        # ----------------- 3) interpolate low-res to high-res grid if requested -----------------
        if self.interp_low_res:
            self.low_ds_interp = (
                self.low_ds
                .interp(
                    lat=self.high_ds.lat,
                    lon=self.high_ds.lon,
                    method="linear",
                )
            )
            src_low = self.low_ds_interp
        else:
            self.low_ds_interp = None
            src_low = self.low_ds

        # ----------------- 4) spatial overlap & cropping on high-res grid -----------------
        lr_lats = src_low.lat.values
        lr_lons = src_low.lon.values
        hr_lats = self.high_ds.lat.values
        hr_lons = self.high_ds.lon.values

        self.hr_lats = hr_lats
        self.hr_lons = hr_lons

        lat_mask = (hr_lats >= lr_lats.min()) & (hr_lats <= lr_lats.max())
        lon_mask = (hr_lons >= lr_lons.min()) & (hr_lons <= lr_lons.max())
        lat_inds = np.where(lat_mask)[0]
        lon_inds = np.where(lon_mask)[0]
        if lat_inds.size == 0 or lon_inds.size == 0:
            raise ValueError("No overlap between low-res and high-res grids!")

        def _trim_to_mult_from_bottom_left(idx_array: np.ndarray, mult: int) -> Tuple[int, int]:
            start_full = idx_array.min()
            end_full   = idx_array.max() + 1
            length     = end_full - start_full
            length_trim = (length // mult) * mult
            if length_trim <= 0:
                return start_full, end_full
            new_start = end_full - length_trim
            new_end   = end_full
            return new_start, new_end

        if crop_mult > 1:
            self.lat_start, self.lat_end = _trim_to_mult_from_bottom_left(lat_inds, crop_mult)
            self.lon_start, self.lon_end = _trim_to_mult_from_bottom_left(lon_inds, crop_mult)
        else:
            self.lat_start, self.lat_end = lat_inds.min(), lat_inds.max() + 1
            self.lon_start, self.lon_end = lon_inds.min(), lon_inds.max() + 1

        # keep *original* low-res grid shape for metadata
        self.lr_lats = self.low_ds.lat.values
        self.lr_lons = self.low_ds.lon.values

        # ----------------- 5) map times to indices & sample list -----------------
        time_to_idx_low  = {t: i for i, t in enumerate(self.low_times)}
        time_to_idx_high = {t: i for i, t in enumerate(self.valid_times)}

        self.samples_idx: List[Tuple[int, int, int]] = []
        # NEW: per-sample time IDs, one integer per sample
        self.sample_time_id: List[int] = []

        for time_id, t in enumerate(self.times_common):
            ti_low  = time_to_idx_low[t]
            ti_high = time_to_idx_high[t]
            for li in range(len(self.lead_values)):
                self.samples_idx.append((ti_low, ti_high, li))   # (time_low, time_high, lead_idx)
                self.sample_time_id.append(time_id)              # <-- same ID for all leads at this valid time

        self.sample_time_id = np.array(self.sample_time_id, dtype=np.int64)

        # ----------------- 6) load min/max stats -----------------
        num_in  = len(self.input_variables)   # dynamic + invariants
        num_out = len(self.output_variables)

        if stats_path and stats_path != "???":
            with open(stats_path, "r") as f:
                stats = json.load(f)
            inp_stats = stats.get("input", {})
            out_stats = stats.get("output", {})
            self.input_min = np.array(
                [inp_stats.get(v, {}).get("min", 0.0) for v in self.input_variables],
                dtype=np.float32,
            )
            self.input_max = np.array(
                [inp_stats.get(v, {}).get("max", 1.0) for v in self.input_variables],
                dtype=np.float32,
            )
            self.output_min = np.array(
                [out_stats.get(v, {}).get("min", 0.0) for v in self.output_variables],
                dtype=np.float32,
            )
            self.output_max = np.array(
                [out_stats.get(v, {}).get("max", 1.0) for v in self.output_variables],
                dtype=np.float32,
            )
        else:
            self.input_min  = np.zeros(num_in,  dtype=np.float32)
            self.input_max  = np.ones( num_in,  dtype=np.float32)
            self.output_min = np.zeros(num_out, dtype=np.float32)
            self.output_max = np.ones( num_out, dtype=np.float32)

        # ----------------- 7) preload dynamic data into NumPy arrays -----------------
        T = len(self.low_ds.time)
        L = len(self.lead_values)
        H = self.lat_end - self.lat_start
        W = self.lon_end - self.lon_start
        Cdyn  = len(self.dynamic_input_variables)
        Cout  = len(self.output_variables)

        low_data  = np.zeros((T, L, Cdyn,  H, W), dtype=np.float32)
        high_data = np.zeros((T, L, Cout, H, W), dtype=np.float32)

        # helper to broadcast arrays to (T, L, H, W)
        def _broadcast_to_TLHW(v: np.ndarray, name: str) -> np.ndarray:
            """
            Accepts v with shape:
              (T, L, H, W) or (T, H, W) or (H, W)
            and broadcasts to (T, L, H, W).
            """
            v = np.array(v, copy=False)
            if v.ndim == 4:
                if v.shape[0] != T:
                    raise ValueError(f"{name}: time dim mismatch: {v.shape[0]} vs {T}")
                if v.shape[1] != L:
                    if v.shape[1] == 1:
                        v = np.repeat(v, L, axis=1)
                    else:
                        raise ValueError(f"{name}: lead dim mismatch: {v.shape[1]} vs {L}")
                return v.astype(np.float32)
            elif v.ndim == 3:
                if v.shape[0] != T:
                    raise ValueError(f"{name}: time dim mismatch: {v.shape[0]} vs {T}")
                v = v[:, None, :, :]  # (T,1,H,W) → (T,L,H,W)
                v = np.repeat(v, L, axis=1)
                return v.astype(np.float32)
            elif v.ndim == 2:
                v = v[None, None, :, :]           # (1,1,H,W) → (T,L,H,W)
                v = np.broadcast_to(v, (T, L, H, W))
                return v.astype(np.float32)
            else:
                raise ValueError(f"{name}: unsupported ndim {v.ndim}, shape {v.shape}")

        # fill dynamic input channels from src_low (possibly interpolated)
        for ci, var in enumerate(self.dynamic_input_variables):
            da = src_low[var].isel(
                lat=slice(self.lat_start, self.lat_end),
                lon=slice(self.lon_start, self.lon_end),
            )
            v = _broadcast_to_TLHW(da.values, name=f"input var {var}")
            low_data[:, :, ci, :, :] = v

        # fill output channels from high_ds (target), which may or may not have prediction_delta
        for co, var in enumerate(self.output_variables):
            da = self.high_ds[var].isel(
                lat=slice(self.lat_start, self.lat_end),
                lon=slice(self.lon_start, self.lon_end),
            )
            v = _broadcast_to_TLHW(da.values, name=f"output var {var}")
            high_data[:, :, co, :, :] = v

        self.low_data  = low_data          # (T, L, Cdyn,  H, W)
        self.high_data = high_data         # (T, L, Cout, H, W)

        # ----------------- 8) load invariant variables (static) to HR grid -----------------
        self.inv_data = None
        if len(self.invariant_variables) > 0:
            Cinv = len(self.invariant_variables)
            inv = np.zeros((Cinv, H, W), dtype=np.float32)

            for ci, var in enumerate(self.invariant_variables):
                if var in self.high_ds:
                    # static on high-res grid
                    da_inv = self.high_ds[var].isel(
                        lat=slice(self.lat_start, self.lat_end),
                        lon=slice(self.lon_start, self.lon_end),
                    )
                elif var in self.low_ds:
                    # static on low-res grid: optionally interpolate to high-res grid
                    if self.interp_low_res:
                        da_low = self.low_ds[var]
                        da_interp = da_low.interp(
                            lat=self.high_ds.lat,
                            lon=self.high_ds.lon,
                            method="linear",
                        )
                        da_inv = da_interp.isel(
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )
                    else:
                        da_inv = self.low_ds[var].isel(
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )
                else:
                    raise ValueError(
                        f"Invariant variable '{var}' not found in low_res or high_res datasets."
                    )

                v = da_inv.values.astype(np.float32)
                # Accept (H, W) or (T, H, W) or (L, H, W) and reduce to (H, W)
                if v.ndim == 2:
                    inv[ci] = v
                elif v.ndim == 3:
                    # take first slice along the leading axis (time or lead)
                    inv[ci] = v[0]
                else:
                    raise ValueError(f"Invariant '{var}' has unsupported shape {v.shape}.")

            self.inv_data = inv  # (Cinv, H, W)

        # ----------------- 9) close xarray datasets -----------------
        self.low_ds.close()
        self.high_ds.close()
        if self.low_ds_interp is not None:
            self.low_ds_interp.close()

    # --------- metadata helpers ---------

    def longitude(self):
        return self.hr_lons[self.lon_start:self.lon_end]

    def latitude(self):
        return self.hr_lats[self.lat_start:self.lat_end]

    def input_channels(self):
        # include both dynamic + invariant
        return [ChannelMetadata(name=v) for v in self.input_variables]

    def output_channels(self):
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self):
        """
        Return per-sample valid times, in the same order as __getitem__ indexing.
        (kept for backward compatibility; used by your weekly split logic)
        """
        times = []
        for ti_low, _, _ in self.samples_idx:
            times.append(convert_datetime_to_cftime(self.low_times[ti_low]))
        return times

    def time_ids(self):
        """
        Return the integer time_id for each sample (same length as dataset).
        Samples that share the same valid time share the same time_id.
        """
        return self.sample_time_id.copy()

    def image_shape(self):
        H = self.lat_end - self.lat_start
        W = self.lon_end - self.lon_start
        return H, W

    def low_res_shape(self):
        """Return original low-res grid size (before any interp)."""
        return (len(self.lr_lats), len(self.lr_lons))

    def __len__(self):
        return len(self.samples_idx)

    def get_prob_channel_index(self) -> List[int]:
        # precip (channel 0 among dynamic inputs) as probabilistic channel if needed
        # invariants are appended AFTER dynamic channels, so index 0 remains precip
        return [0]

    # --------------- core: __getitem__ ---------------

    def __getitem__(self, idx: int):
        ti_low, ti_high, li = self.samples_idx[idx]
        time_id = self.sample_time_id[idx]   # scalar int

        # dynamic part: (Cdyn, H, W)
        x_dyn = self.low_data[ti_low, li].copy()
        # target: (Cout, H, W)
        y_arr = self.high_data[ti_high, li].copy()

        # append invariants if present
        if self.inv_data is not None:
            # inv_data: (Cinv, H, W)
            x_arr = np.concatenate([x_dyn, self.inv_data], axis=0)
        else:
            x_arr = x_dyn

        # physical sanitization
        np.maximum(x_arr, 0.0, out=x_arr)
        np.maximum(y_arr, 0.0, out=y_arr)
        _nan_to_num_inplace(x_arr, clip_abs=1e6)
        _nan_to_num_inplace(y_arr, clip_abs=1e6)
        _per_channel_clip(x_arr, hi=None)
        _per_channel_clip(y_arr, hi=None)

        # optional log-transform on precip channel 0 (dynamic precip)
        if self.log_precip:
            x_arr[0] = np.log1p(x_arr[0])
            y_arr[0] = np.log1p(y_arr[0])
            _nan_to_num_inplace(x_arr[0:1], clip_abs=30.0)
            _nan_to_num_inplace(y_arr[0:1], clip_abs=30.0)

        # lead index as LongTensor (no normalization)
        lead_t = torch.tensor([li], dtype=torch.long)
        time_id_t = torch.tensor([time_id], dtype=torch.long)

        if not self.normalize:
            return (
                torch.from_numpy(y_arr).float(),
                torch.from_numpy(x_arr).float(),
                lead_t,
                time_id_t,
            )

        # --- Min–Max normalization over ALL input channels (dynamic + invariants) ---
        inp_min = self.input_min[:, None, None]
        inp_max = self.input_max[:, None, None]
        out_min = self.output_min[:, None, None]
        out_max = self.output_max[:, None, None]

        inp_min = np.nan_to_num(inp_min, nan=0.0, posinf=0.0, neginf=0.0)
        out_min = np.nan_to_num(out_min, nan=0.0, posinf=0.0, neginf=0.0)
        inp_max = np.nan_to_num(inp_max, nan=1.0, posinf=1.0, neginf=1.0)
        out_max = np.nan_to_num(out_max, nan=1.0, posinf=1.0, neginf=1.0)

        inp_range = np.maximum(inp_max - inp_min, 1e-6)
        out_range = np.maximum(out_max - out_min, 1e-6)

        x_norm = (x_arr - inp_min) / inp_range
        y_norm = (y_arr - out_min) / out_range

        _nan_to_num_inplace(x_norm, clip_abs=1e6)
        _nan_to_num_inplace(y_norm, clip_abs=1e6)

        x_t = torch.from_numpy(x_norm).float()
        y_t = torch.from_numpy(y_norm).float()

        return y_t, x_t, lead_t, time_id_t

# NoLead Fast, with added elevation! and the possibility to patch 
class CustomDownscalingDatasetNoLeadFastLite_2(DownscalingDataset):
    """
    Memory-aware no-lead dataset for CorrDiff step 2 (regression/super-resolution).

      Inputs:
         - dynamic input_variables: low-res fields (time, lat, lon), e.g. precip
         - invariant_variables: static fields (lat, lon), e.g. altitude

      Outputs:
         - high_res fields (time, lat, lon), e.g. precip at fine grid

    Features:
      - Optional low_res -> high_res interpolation per variable.
      - Optional preloading of *low-res* only (in float16 or float32).
      - High-res is never fully preloaded to avoid multi-GB memory use.
      - Optional patch-based training: random crops (patch_shape_y, patch_shape_x).
      - No lead_time; __getitem__ returns (y_t, x_t).

    __getitem__(idx) returns:
      y_t : (C_out, H, W) torch.float32
      x_t : (C_in_dyn + C_in_inv,  H, W) torch.float32
    """

    def __init__(
        self,
        low_res_file: str,
        high_res_file: str,
        input_variables: List[str],
        output_variables: List[str],
        stats_path: Optional[str] = None,
        engine: str = "netcdf4",
        data_path: Optional[str] = None,
        log_precip: bool = False,
        crop_mult: int = 16,
        normalize: bool = True,
        interp_low_res: bool = True,
        preload_low: bool = True,
        preload_dtype: str = "float16",  # 'float16' or 'float32'
        # NEW: static fields like altitude
        invariant_variables: Optional[List[str]] = None,
        # NEW: patch options
        use_patches: bool = False,
        patch_shape_y: Optional[int] = None,
        patch_shape_x: Optional[int] = None,
        **kwargs,
    ):
        # ---- store configuration ----
        self.dynamic_input_variables = input_variables
        self.invariant_variables     = invariant_variables or []
        # All input channels as seen by the model & stats
        self.input_variables  = self.dynamic_input_variables + self.invariant_variables
        self.output_variables = output_variables

        self.log_precip       = log_precip
        self.normalize        = normalize
        self.interp_low_res   = interp_low_res
        self.preload_low      = preload_low

        self.use_patches   = use_patches
        self.patch_shape_y = patch_shape_y
        self.patch_shape_x = patch_shape_x

        if preload_dtype not in ("float16", "float32"):
            raise ValueError("preload_dtype must be 'float16' or 'float32'")
        self._storage_dtype = np.float16 if preload_dtype == "float16" else np.float32

        # 1) open datasets
        self.low_ds  = xr.open_dataset(low_res_file,  engine=engine)
        self.high_ds = xr.open_dataset(high_res_file, engine=engine)

        # 2) time alignment
        self.low_times  = pd.DatetimeIndex(self.low_ds.time.values)
        self.high_times = pd.DatetimeIndex(self.high_ds.time.values)
        times_common    = sorted(set(self.low_times).intersection(set(self.high_times)))
        if len(times_common) == 0:
            raise ValueError("No overlapping times between low_res and high_res datasets!")

        # 3) spatial overlap on high-res grid
        hr_lats = self.high_ds.lat.values
        hr_lons = self.high_ds.lon.values
        self.hr_lats = hr_lats
        self.hr_lons = hr_lons

        lr_lats = self.low_ds.lat.values
        lr_lons = self.low_ds.lon.values

        lat_mask = (hr_lats >= lr_lats.min()) & (hr_lats <= lr_lats.max())
        lon_mask = (hr_lons >= lr_lons.min()) & (hr_lons <= lr_lons.max())
        lat_inds = np.where(lat_mask)[0]
        lon_inds = np.where(lon_mask)[0]
        if lat_inds.size == 0 or lon_inds.size == 0:
            raise ValueError("No overlap between low-res and high-res grids!")

        def _trim_to_mult_from_bottom_left(idx_array: np.ndarray, mult: int) -> Tuple[int, int]:
            start_full = idx_array.min()
            end_full   = idx_array.max() + 1
            length     = end_full - start_full
            length_trim = (length // mult) * mult
            if length_trim <= 0:
                return start_full, end_full
            new_start = end_full - length_trim
            new_end   = end_full
            return new_start, new_end

        if crop_mult > 1:
            self.lat_start, self.lat_end = _trim_to_mult_from_bottom_left(lat_inds, crop_mult)
            self.lon_start, self.lon_end = _trim_to_mult_from_bottom_left(lon_inds, crop_mult)
        else:
            self.lat_start, self.lat_end = lat_inds.min(), lat_inds.max() + 1
            self.lon_start, self.lon_end = lon_inds.min(), lon_inds.max() + 1

        self.H_full = self.lat_end - self.lat_start
        self.W_full = self.lon_end - self.lon_start

        # 4) patch sanity checks
        if self.use_patches:
            if self.patch_shape_y is None or self.patch_shape_x is None:
                raise ValueError("use_patches=True but patch_shape_y/x not provided.")
            if self.patch_shape_y > self.H_full or self.patch_shape_x > self.W_full:
                raise ValueError(
                    f"Patch shape {self.patch_shape_y}x{self.patch_shape_x} exceeds full size "
                    f"{self.H_full}x{self.W_full}."
                )
            # best if multiples of 16, but we don't enforce

        # keep original low-res grid for metadata
        self.lr_lats = self.low_ds.lat.values
        self.lr_lons = self.low_ds.lon.values

        # 5) map times to indices & sample list
        time_to_idx_low  = {t: i for i, t in enumerate(self.low_times)}
        time_to_idx_high = {t: i for i, t in enumerate(self.high_times)}

        self.samples_idx: List[Tuple[int, int]] = []
        for t in times_common:
            ti_low  = time_to_idx_low[t]
            ti_high = time_to_idx_high[t]
            self.samples_idx.append((ti_low, ti_high))

        # 6) load min/max stats (for *all* input channels: dynamic + invariants)
        num_in  = len(self.input_variables)
        num_out = len(self.output_variables)
        if stats_path and stats_path not in (None, "", "???"):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            inp_stats = stats.get("input", {})
            out_stats = stats.get("output", {})
            self.input_min = np.array(
                [inp_stats.get(v, {}).get("min", 0.0) for v in self.input_variables],
                dtype=np.float32,
            )
            self.input_max = np.array(
                [inp_stats.get(v, {}).get("max", 1.0) for v in self.input_variables],
                dtype=np.float32,
            )
            self.output_min = np.array(
                [out_stats.get(v, {}).get("min", 0.0) for v in self.output_variables],
                dtype=np.float32,
            )
            self.output_max = np.array(
                [out_stats.get(v, {}).get("max", 1.0) for v in self.output_variables],
                dtype=np.float32,
            )
        else:
            self.input_min  = np.zeros(num_in,  dtype=np.float32)
            self.input_max  = np.ones( num_in,  dtype=np.float32)
            self.output_min = np.zeros(num_out, dtype=np.float32)
            self.output_max = np.ones( num_out, dtype=np.float32)

        # 7) helper to broadcast to (T, H, W)
        def _broadcast_to_THW(v: np.ndarray, T: int, name: str) -> np.ndarray:
            v = np.array(v, copy=False)
            if v.ndim == 3:
                if v.shape[0] != T:
                    raise ValueError(f"{name}: time dim mismatch: {v.shape[0]} vs {T}")
                return v.astype(np.float32)
            elif v.ndim == 2:
                v = v[None, :, :]  # (1, H, W)
                v = np.broadcast_to(v, (T, v.shape[1], v.shape[2]))
                return v.astype(np.float32)
            else:
                raise ValueError(f"{name}: unsupported ndim {v.ndim} for {name}, shape {v.shape}")

        # 8) load invariant variables (static) to HR grid
        self.inv_data = None
        if len(self.invariant_variables) > 0:
            Cinv = len(self.invariant_variables)
            inv = np.zeros((Cinv, self.H_full, self.W_full), dtype=np.float32)

            for ci, var in enumerate(self.invariant_variables):
                if var in self.high_ds:
                    da_inv = self.high_ds[var].isel(
                        lat=slice(self.lat_start, self.lat_end),
                        lon=slice(self.lon_start, self.lon_end),
                    )
                elif var in self.low_ds:
                    # altitude or other invariant on low grid: interpolate if requested
                    if self.interp_low_res:
                        da_low = self.low_ds[var]
                        da_interp = da_low.interp(
                            lat=self.high_ds.lat,
                            lon=self.high_ds.lon,
                            method="linear",
                        )
                        da_inv = da_interp.isel(
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )
                    else:
                        da_inv = self.low_ds[var].isel(
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )
                else:
                    raise ValueError(f"Invariant variable '{var}' not found in low_res or high_res datasets.")

                v = da_inv.values.astype(np.float32)
                if v.ndim == 2:
                    inv[ci] = v
                elif v.ndim == 3:
                    # if (T, H, W) by mistake, take first time slice
                    inv[ci] = v[0]
                else:
                    raise ValueError(f"Invariant '{var}' has unsupported shape {v.shape}.")

            self.inv_data = inv  # (Cinv, H_full, W_full)

        # 9) optionally preload low-res dynamic variables
        self.low_data = None
        T_low = len(self.low_ds.time)
        Cd    = len(self.dynamic_input_variables)

        if self.preload_low:
            low_data = np.zeros((T_low, Cd, self.H_full, self.W_full), dtype=self._storage_dtype)

            for ci, var in enumerate(self.dynamic_input_variables):
                if self.interp_low_res:
                    da_low = self.low_ds[var]
                    da_interp = da_low.interp(
                        lat=self.high_ds.lat,
                        lon=self.high_ds.lon,
                        method="linear",
                    )
                    da = da_interp.isel(
                        lat=slice(self.lat_start, self.lat_end),
                        lon=slice(self.lon_start, self.lon_end),
                    )
                else:
                    da = self.low_ds[var].isel(
                        lat=slice(self.lat_start, self.lat_end),
                        lon=slice(self.lon_start, self.lon_end),
                    )
                v = _broadcast_to_THW(da.values, T_low, name=f"input var {var}")
                low_data[:, ci, :, :] = v.astype(self._storage_dtype)

            self.low_data = low_data
            # we can close low_ds now; invariants already loaded
            self.low_ds.close()
        else:
            # keep low_ds open for per-sample on-the-fly access
            pass

        # high_ds stays open; we read targets per-sample

    # ---------- metadata helpers ----------

    def longitude(self):
        return self.hr_lons[self.lon_start:self.lon_end]

    def latitude(self):
        return self.hr_lats[self.lat_start:self.lat_end]

    def input_channels(self):
        # include both dynamic + invariant
        return [ChannelMetadata(name=v) for v in self.input_variables]

    def output_channels(self):
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self):
        times = []
        for ti_low, _ in self.samples_idx:
            times.append(convert_datetime_to_cftime(self.low_times[ti_low]))
        return times

    def image_shape(self):
        """
        Shape seen by the model (H, W):
          - full cropped size if use_patches=False
          - patch size if use_patches=True
        """
        if self.use_patches:
            return self.patch_shape_y, self.patch_shape_x
        else:
            return self.H_full, self.W_full

    def low_res_shape(self):
        """Return original low-res grid size (before any interp)."""
        return (len(self.lr_lats), len(self.lr_lons))

    def __len__(self):
        return len(self.samples_idx)

    def get_prob_channel_index(self) -> List[int]:
        # precip (channel 0) as probabilistic channel if needed
        return [0]

    # -------------- core: __getitem__ --------------

    def __getitem__(self, idx: int):
        ti_low, ti_high = self.samples_idx[idx]

        Cd   = len(self.dynamic_input_variables)
        Cinv = len(self.invariant_variables)
        Cout = len(self.output_variables)

        # ---- dynamic low-res inputs ----
        if self.low_data is not None:
            x_dyn_full = self.low_data[ti_low].astype(np.float32)  # (Cd, H_full, W_full)
        else:
            x_dyn_full = np.zeros((Cd, self.H_full, self.W_full), dtype=np.float32)
            for ci, var in enumerate(self.dynamic_input_variables):
                if self.interp_low_res:
                    da_low = self.low_ds[var]
                    da_interp = da_low.interp(
                        lat=self.high_ds.lat,
                        lon=self.high_ds.lon,
                        method="linear",
                    )
                    da = da_interp.isel(
                        time=ti_low,
                        lat=slice(self.lat_start, self.lat_end),
                        lon=slice(self.lon_start, self.lon_end),
                    )
                else:
                    da = self.low_ds[var].isel(
                        time=ti_low,
                        lat=slice(self.lat_start, self.lat_end),
                        lon=slice(self.lon_start, self.lon_end),
                    )
                v = da.values.astype(np.float32)
                if v.ndim == 2:
                    v = v[None, :, :]
                x_dyn_full[ci] = v

        # ---- high-res target ----
        y_full = np.zeros((Cout, self.H_full, self.W_full), dtype=np.float32)
        for co, var in enumerate(self.output_variables):
            da = self.high_ds[var].isel(
                time=ti_high,
                lat=slice(self.lat_start, self.lat_end),
                lon=slice(self.lon_start, self.lon_end),
            )
            v = da.values.astype(np.float32)
            if v.ndim == 2:
                v = v[None, :, :]
            y_full[co] = v

        # ---- combine dynamic + invariants into x_full ----
        if self.inv_data is not None:
            # inv_data: (Cinv, H_full, W_full)
            x_full = np.concatenate([x_dyn_full, self.inv_data], axis=0)
        else:
            x_full = x_dyn_full

        # ---- patching (same crop for x & y & invariants) ----
        if self.use_patches:
            ph = self.patch_shape_y
            pw = self.patch_shape_x

            max_y0 = self.H_full - ph
            max_x0 = self.W_full - pw

            y0 = np.random.randint(0, max_y0 + 1) if max_y0 > 0 else 0
            x0 = np.random.randint(0, max_x0 + 1) if max_x0 > 0 else 0

            y1 = y0 + ph
            x1 = x0 + pw

            x_arr = x_full[:, y0:y1, x0:x1].copy()
            y_arr = y_full[:, y0:y1, x0:x1].copy()
        else:
            x_arr = x_full.copy()
            y_arr = y_full.copy()

        # ---- physical sanitization ----
        np.maximum(x_arr, 0.0, out=x_arr)
        np.maximum(y_arr, 0.0, out=y_arr)
        _nan_to_num_inplace(x_arr, clip_abs=1e6)
        _nan_to_num_inplace(y_arr, clip_abs=1e6)
        _per_channel_clip(x_arr, hi=None)
        _per_channel_clip(y_arr, hi=None)

        # ---- log-transform on precip (assumed channel 0 input & output) ----
        if self.log_precip:
            x_arr[0] = np.log1p(x_arr[0])
            y_arr[0] = np.log1p(y_arr[0])
            _nan_to_num_inplace(x_arr[0:1], clip_abs=30.0)
            _nan_to_num_inplace(y_arr[0:1], clip_abs=30.0)

        if not self.normalize:
            return (
                torch.from_numpy(y_arr).float(),
                torch.from_numpy(x_arr).float(),
            )

        # ---- Min–Max normalization over all input channels (dynamic + invariants) ----
        inp_min = self.input_min[:, None, None]
        inp_max = self.input_max[:, None, None]
        out_min = self.output_min[:, None, None]
        out_max = self.output_max[:, None, None]

        inp_min = np.nan_to_num(inp_min, nan=0.0, posinf=0.0, neginf=0.0)
        out_min = np.nan_to_num(out_min, nan=0.0, posinf=0.0, neginf=0.0)
        inp_max = np.nan_to_num(inp_max, nan=1.0, posinf=1.0, neginf=1.0)
        out_max = np.nan_to_num(out_max, nan=1.0, posinf=1.0, neginf=1.0)

        inp_range = np.maximum(inp_max - inp_min, 1e-6)
        out_range = np.maximum(out_max - out_min, 1e-6)

        x_norm = (x_arr - inp_min) / inp_range
        y_norm = (y_arr - out_min) / out_range

        _nan_to_num_inplace(x_norm, clip_abs=1e6)
        _nan_to_num_inplace(y_norm, clip_abs=1e6)

        x_t = torch.from_numpy(x_norm).float()
        y_t = torch.from_numpy(y_norm).float()

        return y_t, x_t

# NoLead Fast, with added elevation! and the possibility to patch
# NOW: can also handle multi-horizon LR files with prediction_delta during inference
class CustomDownscalingDatasetNoLeadFastLite_3(DownscalingDataset):
    """
    Memory-aware dataset for CorrDiff step 2 (regression/super-resolution).

    TRAINING MODE (classic Step-2):
      - low_res is (time, lat, lon) -> behaves like before
      - __getitem__ returns (y_t, x_t)

    INFERENCE MODE (multi-horizon bias-corrected LR):
      - low_res is (time, prediction_delta, lat, lon)
      - dataset expands to time×lead samples
      - __getitem__ returns (y_t, x_t, lead_label, time_id)
        so generate.py can pass lead_time_label to the models.
    """

    def __init__(
        self,
        low_res_file: str,
        high_res_file: str,
        input_variables: List[str],
        output_variables: List[str],
        stats_path: Optional[str] = None,
        engine: str = "netcdf4",
        data_path: Optional[str] = None,
        log_precip: bool = False,
        crop_mult: int = 16,
        normalize: bool = True,
        interp_low_res: bool = True,
        preload_low: bool = True,
        preload_dtype: str = "float16",  # 'float16' or 'float32'
        invariant_variables: Optional[List[str]] = None,
        use_patches: bool = False,
        patch_shape_y: Optional[int] = None,
        patch_shape_x: Optional[int] = None,

        # NEW: lead handling for inference
        pred_delta_dim_name: str = "prediction_delta",
        use_pred_delta_as_lead: Optional[bool] = None,
        preload_low_with_leads: bool = False,  # to protect you from accidental OOM
        **kwargs,
    ):
        # ---- store configuration ----
        self.dynamic_input_variables = input_variables
        self.invariant_variables     = invariant_variables or []
        self.input_variables  = self.dynamic_input_variables + self.invariant_variables
        self.output_variables = output_variables

        self.log_precip       = log_precip
        self.normalize        = normalize
        self.interp_low_res   = interp_low_res
        self.preload_low      = preload_low

        self.use_patches   = use_patches
        self.patch_shape_y = patch_shape_y
        self.patch_shape_x = patch_shape_x

        self.pred_delta_dim_name = pred_delta_dim_name

        if preload_dtype not in ("float16", "float32"):
            raise ValueError("preload_dtype must be 'float16' or 'float32'")
        self._storage_dtype = np.float16 if preload_dtype == "float16" else np.float32

        # 1) open datasets
        self.low_ds  = xr.open_dataset(low_res_file,  engine=engine)
        self.high_ds = xr.open_dataset(high_res_file, engine=engine)

        # -----------------------------
        # NEW: detect multi-horizon LR
        # -----------------------------
        # We consider "multi-horizon inference mode" if ANY input variable has prediction_delta dim.
        self._lr_has_leads = False
        for v in self.dynamic_input_variables:
            if v in self.low_ds and (self.pred_delta_dim_name in self.low_ds[v].dims):
                self._lr_has_leads = True
                break

        if use_pred_delta_as_lead is None:
            self.use_leads = self._lr_has_leads
        else:
            self.use_leads = bool(use_pred_delta_as_lead) and self._lr_has_leads

        if self.use_leads:
            # lead coordinate values (timedelta64 most likely)
            # take from first dynamic var that has the dim
            v0 = next(v for v in self.dynamic_input_variables if self.pred_delta_dim_name in self.low_ds[v].dims)
            self.lead_values = self.low_ds[v0][self.pred_delta_dim_name].values
            self.num_leads = int(len(self.lead_values))
            # safety: preloading becomes T×L×... which can explode
            if self.preload_low and (not preload_low_with_leads):
                print(
                    f"[WARN] low_res has '{self.pred_delta_dim_name}' (L={self.num_leads}). "
                    f"Disabling preload_low to avoid OOM. "
                    f"Set preload_low_with_leads=True if you really want it."
                )
                self.preload_low = False
        else:
            self.lead_values = None
            self.num_leads = 1

        # 2) time alignment
        self.low_times  = pd.DatetimeIndex(self.low_ds.time.values)
        self.high_times = pd.DatetimeIndex(self.high_ds.time.values)
        times_common    = sorted(set(self.low_times).intersection(set(self.high_times)))
        if len(times_common) == 0:
            raise ValueError("No overlapping times between low_res and high_res datasets!")

        # 3) spatial overlap on high-res grid
        hr_lats = self.high_ds.lat.values
        hr_lons = self.high_ds.lon.values
        self.hr_lats = hr_lats
        self.hr_lons = hr_lons

        lr_lats = self.low_ds.lat.values
        lr_lons = self.low_ds.lon.values

        lat_mask = (hr_lats >= lr_lats.min()) & (hr_lats <= lr_lats.max())
        lon_mask = (hr_lons >= lr_lons.min()) & (hr_lons <= lr_lons.max())
        lat_inds = np.where(lat_mask)[0]
        lon_inds = np.where(lon_mask)[0]
        if lat_inds.size == 0 or lon_inds.size == 0:
            raise ValueError("No overlap between low-res and high-res grids!")

        def _trim_to_mult_from_bottom_left(idx_array: np.ndarray, mult: int) -> Tuple[int, int]:
            start_full = idx_array.min()
            end_full   = idx_array.max() + 1
            length     = end_full - start_full
            length_trim = (length // mult) * mult
            if length_trim <= 0:
                return start_full, end_full
            new_start = end_full - length_trim
            new_end   = end_full
            return new_start, new_end

        if crop_mult > 1:
            self.lat_start, self.lat_end = _trim_to_mult_from_bottom_left(lat_inds, crop_mult)
            self.lon_start, self.lon_end = _trim_to_mult_from_bottom_left(lon_inds, crop_mult)
        else:
            self.lat_start, self.lat_end = lat_inds.min(), lat_inds.max() + 1
            self.lon_start, self.lon_end = lon_inds.min(), lon_inds.max() + 1

        self.H_full = self.lat_end - self.lat_start
        self.W_full = self.lon_end - self.lon_start

        # 4) patch sanity checks
        if self.use_patches:
            if self.patch_shape_y is None or self.patch_shape_x is None:
                raise ValueError("use_patches=True but patch_shape_y/x not provided.")
            if self.patch_shape_y > self.H_full or self.patch_shape_x > self.W_full:
                raise ValueError(
                    f"Patch shape {self.patch_shape_y}x{self.patch_shape_x} exceeds full size "
                    f"{self.H_full}x{self.W_full}."
                )

        # keep original low-res grid for metadata
        self.lr_lats = self.low_ds.lat.values
        self.lr_lons = self.low_ds.lon.values

        # 5) map times to indices & sample list
        time_to_idx_low  = {t: i for i, t in enumerate(self.low_times)}
        time_to_idx_high = {t: i for i, t in enumerate(self.high_times)}

        # -----------------------------
        # NEW: samples_idx holds lead too if use_leads
        # -----------------------------
        self.samples_idx = []
        for t in times_common:
            ti_low  = time_to_idx_low[t]
            ti_high = time_to_idx_high[t]
            if self.use_leads:
                for li in range(self.num_leads):
                    self.samples_idx.append((ti_low, ti_high, li))
            else:
                self.samples_idx.append((ti_low, ti_high))

        # 6) load min/max stats
        num_in  = len(self.input_variables)
        num_out = len(self.output_variables)
        if stats_path and stats_path not in (None, "", "???"):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            inp_stats = stats.get("input", {})
            out_stats = stats.get("output", {})
            self.input_min = np.array(
                [inp_stats.get(v, {}).get("min", 0.0) for v in self.input_variables],
                dtype=np.float32,
            )
            self.input_max = np.array(
                [inp_stats.get(v, {}).get("max", 1.0) for v in self.input_variables],
                dtype=np.float32,
            )
            self.output_min = np.array(
                [out_stats.get(v, {}).get("min", 0.0) for v in self.output_variables],
                dtype=np.float32,
            )
            self.output_max = np.array(
                [out_stats.get(v, {}).get("max", 1.0) for v in self.output_variables],
                dtype=np.float32,
            )
        else:
            self.input_min  = np.zeros(num_in,  dtype=np.float32)
            self.input_max  = np.ones( num_in,  dtype=np.float32)
            self.output_min = np.zeros(num_out, dtype=np.float32)
            self.output_max = np.ones( num_out, dtype=np.float32)

        # 7) helper to broadcast to (T, H, W)
        def _broadcast_to_THW(v: np.ndarray, T: int, name: str) -> np.ndarray:
            v = np.array(v, copy=False)
            if v.ndim == 3:
                if v.shape[0] != T:
                    raise ValueError(f"{name}: time dim mismatch: {v.shape[0]} vs {T}")
                return v.astype(np.float32)
            elif v.ndim == 2:
                v = v[None, :, :]  # (1, H, W)
                v = np.broadcast_to(v, (T, v.shape[1], v.shape[2]))
                return v.astype(np.float32)
            else:
                raise ValueError(f"{name}: unsupported ndim {v.ndim} for {name}, shape {v.shape}")

        # 8) load invariant variables (static) to HR grid
        self.inv_data = None
        if len(self.invariant_variables) > 0:
            Cinv = len(self.invariant_variables)
            inv = np.zeros((Cinv, self.H_full, self.W_full), dtype=np.float32)

            for ci, var in enumerate(self.invariant_variables):
                if var in self.high_ds:
                    da_inv = self.high_ds[var].isel(
                        lat=slice(self.lat_start, self.lat_end),
                        lon=slice(self.lon_start, self.lon_end),
                    )
                elif var in self.low_ds:
                    if self.interp_low_res:
                        da_low = self.low_ds[var]
                        da_interp = da_low.interp(
                            lat=self.high_ds.lat,
                            lon=self.high_ds.lon,
                            method="linear",
                        )
                        da_inv = da_interp.isel(
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )
                    else:
                        da_inv = self.low_ds[var].isel(
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )
                else:
                    raise ValueError(f"Invariant variable '{var}' not found in low_res or high_res datasets.")

                v = da_inv.values.astype(np.float32)
                if v.ndim == 2:
                    inv[ci] = v
                elif v.ndim == 3:
                    inv[ci] = v[0]
                else:
                    raise ValueError(f"Invariant '{var}' has unsupported shape {v.shape}.")

            self.inv_data = inv  # (Cinv, H_full, W_full)

        # 9) optionally preload low-res dynamic variables
        self.low_data = None
        T_low = len(self.low_ds.time)
        Cd    = len(self.dynamic_input_variables)

        if self.preload_low:
            if self.use_leads:
                low_data = np.zeros((T_low, self.num_leads, Cd, self.H_full, self.W_full), dtype=self._storage_dtype)
            else:
                low_data = np.zeros((T_low, Cd, self.H_full, self.W_full), dtype=self._storage_dtype)

            for ci, var in enumerate(self.dynamic_input_variables):
                if var not in self.low_ds:
                    raise ValueError(f"Input var '{var}' not found in low_res dataset.")

                if not self.use_leads:
                    # original behavior
                    if self.interp_low_res:
                        da_low = self.low_ds[var]
                        da_interp = da_low.interp(lat=self.high_ds.lat, lon=self.high_ds.lon, method="linear")
                        da = da_interp.isel(
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )
                    else:
                        da = self.low_ds[var].isel(
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )

                    v = _broadcast_to_THW(da.values, T_low, name=f"input var {var}")
                    low_data[:, ci, :, :] = v.astype(self._storage_dtype)

                else:
                    # lead-aware preload: loop leads (can be expensive; only allowed if you forced preload_low_with_leads=True)
                    for li in range(self.num_leads):
                        if self.interp_low_res:
                            da_sel = self.low_ds[var].isel({self.pred_delta_dim_name: li})
                            da_interp = da_sel.interp(lat=self.high_ds.lat, lon=self.high_ds.lon, method="linear")
                            da = da_interp.isel(
                                lat=slice(self.lat_start, self.lat_end),
                                lon=slice(self.lon_start, self.lon_end),
                            )
                        else:
                            da = self.low_ds[var].isel(
                                {self.pred_delta_dim_name: li},
                                lat=slice(self.lat_start, self.lat_end),
                                lon=slice(self.lon_start, self.lon_end),
                            )
                        v = _broadcast_to_THW(da.values, T_low, name=f"input var {var} lead={li}")
                        low_data[:, li, ci, :, :] = v.astype(self._storage_dtype)

            self.low_data = low_data
            self.low_ds.close()
        else:
            # keep low_ds open for per-sample access
            pass

        # high_ds stays open

    # ---------- metadata helpers ----------

    def longitude(self):
        return self.hr_lons[self.lon_start:self.lon_end]

    def latitude(self):
        return self.hr_lats[self.lat_start:self.lat_end]

    def input_channels(self):
        return [ChannelMetadata(name=v) for v in self.input_variables]

    def output_channels(self):
        return [ChannelMetadata(name=v) for v in self.output_variables]

    # -----------------------------
    # NEW: repeat times per lead
    # -----------------------------
    def time(self):
        times = []
        if not self.use_leads:
            for ti_low, _ in self.samples_idx:
                times.append(convert_datetime_to_cftime(self.low_times[ti_low]))
        else:
            for ti_low, _, _li in self.samples_idx:
                times.append(convert_datetime_to_cftime(self.low_times[ti_low]))
        return times

    def image_shape(self):
        if self.use_patches:
            return self.patch_shape_y, self.patch_shape_x
        else:
            return self.H_full, self.W_full

    def low_res_shape(self):
        return (len(self.lr_lats), len(self.lr_lons))

    def __len__(self):
        return len(self.samples_idx)

    def get_prob_channel_index(self) -> List[int]:
        return [0]

    # -------------- core: __getitem__ --------------

    def __getitem__(self, idx: int):
        if not self.use_leads:
            ti_low, ti_high = self.samples_idx[idx]
            li = None
        else:
            ti_low, ti_high, li = self.samples_idx[idx]

        Cd   = len(self.dynamic_input_variables)
        Cinv = len(self.invariant_variables)
        Cout = len(self.output_variables)

        # ---- dynamic low-res inputs ----
        if self.low_data is not None:
            if not self.use_leads:
                x_dyn_full = self.low_data[ti_low].astype(np.float32)  # (Cd, H_full, W_full)
            else:
                x_dyn_full = self.low_data[ti_low, li].astype(np.float32)  # (Cd, H_full, W_full)
        else:
            x_dyn_full = np.zeros((Cd, self.H_full, self.W_full), dtype=np.float32)

            for ci, var in enumerate(self.dynamic_input_variables):
                if self.interp_low_res:
                    # KEY CHANGE: select (time, lead) BEFORE interp -> much faster
                    if self.use_leads:
                        da_sel = self.low_ds[var].isel(time=ti_low, **{self.pred_delta_dim_name: li})
                    else:
                        da_sel = self.low_ds[var].isel(time=ti_low)

                    da_interp = da_sel.interp(
                        lat=self.high_ds.lat,
                        lon=self.high_ds.lon,
                        method="linear",
                    )
                    da = da_interp.isel(
                        lat=slice(self.lat_start, self.lat_end),
                        lon=slice(self.lon_start, self.lon_end),
                    )
                else:
                    if self.use_leads:
                        da = self.low_ds[var].isel(
                            time=ti_low,
                            **{self.pred_delta_dim_name: li},
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )
                    else:
                        da = self.low_ds[var].isel(
                            time=ti_low,
                            lat=slice(self.lat_start, self.lat_end),
                            lon=slice(self.lon_start, self.lon_end),
                        )

                v = da.values.astype(np.float32)
                if v.ndim == 2:
                    v = v[None, :, :]
                x_dyn_full[ci] = v

        # ---- high-res target ----
        y_full = np.zeros((Cout, self.H_full, self.W_full), dtype=np.float32)
        for co, var in enumerate(self.output_variables):
            da = self.high_ds[var].isel(
                time=ti_high,
                lat=slice(self.lat_start, self.lat_end),
                lon=slice(self.lon_start, self.lon_end),
            )
            v = da.values.astype(np.float32)
            if v.ndim == 2:
                v = v[None, :, :]
            y_full[co] = v

        # ---- combine dynamic + invariants into x_full ----
        if self.inv_data is not None:
            x_full = np.concatenate([x_dyn_full, self.inv_data], axis=0)
        else:
            x_full = x_dyn_full

        # ---- patching ----
        if self.use_patches:
            ph = self.patch_shape_y
            pw = self.patch_shape_x
            max_y0 = self.H_full - ph
            max_x0 = self.W_full - pw
            y0 = np.random.randint(0, max_y0 + 1) if max_y0 > 0 else 0
            x0 = np.random.randint(0, max_x0 + 1) if max_x0 > 0 else 0
            y1 = y0 + ph
            x1 = x0 + pw
            x_arr = x_full[:, y0:y1, x0:x1].copy()
            y_arr = y_full[:, y0:y1, x0:x1].copy()
        else:
            x_arr = x_full.copy()
            y_arr = y_full.copy()

        # ---- physical sanitization ----
        np.maximum(x_arr, 0.0, out=x_arr)
        np.maximum(y_arr, 0.0, out=y_arr)
        _nan_to_num_inplace(x_arr, clip_abs=1e6)
        _nan_to_num_inplace(y_arr, clip_abs=1e6)
        _per_channel_clip(x_arr, hi=None)
        _per_channel_clip(y_arr, hi=None)

        # ---- log-transform on precip (assumed channel 0 input & output) ----
        if self.log_precip:
            x_arr[0] = np.log1p(x_arr[0])
            y_arr[0] = np.log1p(y_arr[0])
            _nan_to_num_inplace(x_arr[0:1], clip_abs=30.0)
            _nan_to_num_inplace(y_arr[0:1], clip_abs=30.0)

        if not self.normalize:
            y_t = torch.from_numpy(y_arr).float()
            x_t = torch.from_numpy(x_arr).float()
            if not self.use_leads:
                return y_t, x_t
            else:
                lead_label = torch.tensor(li, dtype=torch.long)
                time_id    = torch.tensor(ti_low, dtype=torch.long)
                return y_t, x_t, lead_label, time_id

        # ---- Min–Max normalization ----
        inp_min = self.input_min[:, None, None]
        inp_max = self.input_max[:, None, None]
        out_min = self.output_min[:, None, None]
        out_max = self.output_max[:, None, None]

        inp_min = np.nan_to_num(inp_min, nan=0.0, posinf=0.0, neginf=0.0)
        out_min = np.nan_to_num(out_min, nan=0.0, posinf=0.0, neginf=0.0)
        inp_max = np.nan_to_num(inp_max, nan=1.0, posinf=1.0, neginf=1.0)
        out_max = np.nan_to_num(out_max, nan=1.0, posinf=1.0, neginf=1.0)

        inp_range = np.maximum(inp_max - inp_min, 1e-6)
        out_range = np.maximum(out_max - out_min, 1e-6)

        x_norm = (x_arr - inp_min) / inp_range
        y_norm = (y_arr - out_min) / out_range

        _nan_to_num_inplace(x_norm, clip_abs=1e6)
        _nan_to_num_inplace(y_norm, clip_abs=1e6)

        x_t = torch.from_numpy(x_norm).float()
        y_t = torch.from_numpy(y_norm).float()

        if not self.use_leads:
            return y_t, x_t
        else:
            lead_label = torch.tensor(li, dtype=torch.long)   # passed to diffusion/regression
            time_id    = torch.tensor(ti_low, dtype=torch.long)
            return y_t, x_t, lead_label, time_id
