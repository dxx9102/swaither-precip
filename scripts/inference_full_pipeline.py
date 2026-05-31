"""
End-to-end inference pipeline for precipitation downscaling over Switzerland.

Pipeline:
  STEP 1  – Bias correction:   AIFS low-res preds  → bias-corrected low-res
  STEP 1b – Interpolation:     low-res grid         → high-res grid
  STEP 2  – Super-resolution:  high-res (CorrDiff)  → ensemble of high-res samples

Required user inputs:
  (i)   AIFS low-res predictions   (.nc with dims: time, prediction_delta, lat, lon)
  (ii)  Checkpoint paths:          bias_correction, regression (CorrDiff), diffusion (CorrDiff)
  (iii) Stats JSON files:          one for STEP 1, one for STEP 2
  (iv)  Altitude / invariant data  (.nc with 'altitude' on the high-res grid)

No Hydra / YAML config files needed.  Run with:
    python inference_pipeline.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import contextlib
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import netCDF4 as nc
import torch
import xarray as xr
from numpy.random import default_rng
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# PhysicsNeMo imports  (adjust if your installation differs)
# ---------------------------------------------------------------------------
from physicsnemo.distributed import DistributedManager
from physicsnemo import Module
from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler
from physicsnemo.utils.corrdiff import regression_step, diffusion_step
from physicsnemo.utils.patching import GridPatching2D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RNG = default_rng()
CLIP_ABS = 1e6
LOG_CLIP = 30.0

DEFAULT_BC_INPUT_VARIABLES = [
    "total_precipitation_NoNeg",
    "convective_precipitation_NoNeg",
    "total_cloud_cover",
    "wind_10m_u",
    "wind_10m_v",
    "specific_humidity_500hPa",
    "specific_humidity_850hPa",
    "temperature_500hPa",
    "temperature_850hPa",
    "geopotential_500hPa",
    "geopotential_850hPa",
]

# ===================================================================
#  Helper dataclass – groups every knob the user can set
# ===================================================================
@dataclass
class PipelineConfig:
    """All tuneable knobs for the inference pipeline."""

    # --- data paths ---
    aifs_pred_file: str = ""          # (i) low-res AIFS predictions
    altitude_file: str = ""           # (iv) high-res altitude / invariant NetCDF

    # --- checkpoint paths ---
    bc_ckpt: str = ""                 # bias-correction model  (.mdlus)
    reg_ckpt: str = ""                # CorrDiff regression    (.mdlus)
    dif_ckpt: str = ""                # CorrDiff diffusion     (.mdlus)

    # --- stats files ---
    bc_stats: str = ""                # STEP 1 stats JSON
    sr_stats: str = ""                # STEP 2 stats JSON

    # --- output ---
    output_file: str = "pipeline_output.nc"
    save_bc: bool = False                 # save bias-corrected intermediate as separate file

    # --- input variable names inside AIFS prediction NetCDF ---
    bc_input_variables: List[str] = field(
        default_factory=lambda: list(DEFAULT_BC_INPUT_VARIABLES)
    )
    bc_output_variable: str = "precip6h_cumulative_capped"
    bc_invariant_variables: List[str] = field(
        default_factory=lambda: ["altitude"]
    )

    sr_input_variable: str = "precip6h_cumulative_capped"
    sr_output_variable: str = "precip6h_cumulative_capped"
    invariant_variable: str = "altitude"

    # --- generation settings ---
    num_ensembles: int = 12
    seed_batch_size: int = 12
    horizons: Optional[List[int]] = None   # None → all
    time_stride: int = 1
    max_times: Optional[int] = None

    # --- model behaviour ---
    inference_mode: str = "all"       # "regression", "diffusion", "all"
    hr_mean_conditioning: bool = True
    crop_mult: int = 16               # spatial crop to this multiple
    log_precip: bool = True           # log1p / expm1 transform on precip

    # --- sampler ---
    sampler_type: str = "deterministic"   # "deterministic" or "stochastic"
    sampler_num_steps: int = 18
    sampler_solver: str = "euler"

    # --- performance ---
    bc_batch_size: int = 256
    force_fp16: bool = False


# ===================================================================
#  Spatial helpers
# ===================================================================
def trim_to_mult(idx_array: np.ndarray, mult: int) -> Tuple[int, int]:
    """Trim an index range so its length is a multiple of *mult*, anchored at the top."""
    start = idx_array.min()
    end = idx_array.max() + 1
    length = end - start
    trimmed = (length // mult) * mult
    if trimmed <= 0:
        return start, end
    return end - trimmed, end


def compute_crop_slices(
    lr_lats: np.ndarray,
    lr_lons: np.ndarray,
    hr_lats: np.ndarray,
    hr_lons: np.ndarray,
    crop_mult: int,
) -> Tuple[int, int, int, int]:
    """Return (lat_start, lat_end, lon_start, lon_end) on the high-res grid."""
    lat_mask = (hr_lats >= lr_lats.min()) & (hr_lats <= lr_lats.max())
    lon_mask = (hr_lons >= lr_lons.min()) & (hr_lons <= lr_lons.max())
    lat_inds = np.where(lat_mask)[0]
    lon_inds = np.where(lon_mask)[0]
    if lat_inds.size == 0 or lon_inds.size == 0:
        raise ValueError("No spatial overlap between low-res and high-res grids!")
    if crop_mult > 1:
        lat_s, lat_e = trim_to_mult(lat_inds, crop_mult)
        lon_s, lon_e = trim_to_mult(lon_inds, crop_mult)
    else:
        lat_s, lat_e = lat_inds.min(), lat_inds.max() + 1
        lon_s, lon_e = lon_inds.min(), lon_inds.max() + 1
    return lat_s, lat_e, lon_s, lon_e


# ===================================================================
#  Normalization / processing helpers (matching training pipeline)
# ===================================================================
def load_stats(stats_path: str, input_vars: List[str], output_vars: List[str]):
    """Load min/max from a JSON stats file.  Returns (inp_min, inp_max, out_min, out_max)."""
    with open(stats_path, "r") as f:
        stats = json.load(f)
    inp_stats = stats.get("input", {})
    out_stats = stats.get("output", {})
    inp_min = np.array(
        [inp_stats.get(v, {}).get("min", 0.0) for v in input_vars], dtype=np.float32
    )
    inp_max = np.array(
        [inp_stats.get(v, {}).get("max", 1.0) for v in input_vars], dtype=np.float32
    )
    out_min = np.array(
        [out_stats.get(v, {}).get("min", 0.0) for v in output_vars], dtype=np.float32
    )
    out_max = np.array(
        [out_stats.get(v, {}).get("max", 1.0) for v in output_vars], dtype=np.float32
    )
    return inp_min, inp_max, out_min, out_max


def sanitize(arr: np.ndarray, clip_abs: float = CLIP_ABS) -> np.ndarray:
    """Clamp negatives to 0 and replace NaN/Inf."""
    np.maximum(arr, 0.0, out=arr)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=clip_abs, neginf=-clip_abs)
    return arr


def normalize_minmax(arr: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Min-max normalize along channel axis (channel-first arrays)."""
    vmin = np.nan_to_num(vmin[:, None, None], nan=0.0, posinf=0.0, neginf=0.0)
    vmax = np.nan_to_num(vmax[:, None, None], nan=1.0, posinf=1.0, neginf=1.0)
    rng = np.maximum(vmax - vmin, 1e-6)
    out = (arr - vmin) / rng
    np.nan_to_num(out, copy=False, nan=0.0, posinf=CLIP_ABS, neginf=-CLIP_ABS)
    return out


def denorm_minmax(arr: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """Inverse of normalize_minmax."""
    vmin = np.nan_to_num(vmin[:, None, None], nan=0.0, posinf=0.0, neginf=0.0)
    vmax = np.nan_to_num(vmax[:, None, None], nan=1.0, posinf=1.0, neginf=1.0)
    rng = np.maximum(vmax - vmin, 1e-6)
    return arr * rng + vmin


def denorm_and_delog_tensor(
    y_norm: torch.Tensor,
    out_min: np.ndarray,
    out_max: np.ndarray,
) -> torch.Tensor:
    """Undo min-max norm (in log-space) then expm1.  y_norm: (B, C, H, W)."""
    omin = torch.as_tensor(out_min, device=y_norm.device, dtype=y_norm.dtype)[:, None, None]
    omax = torch.as_tensor(out_max, device=y_norm.device, dtype=y_norm.dtype)[:, None, None]
    rng = torch.clamp(omax - omin, min=1e-6)
    y_log = y_norm * rng + omin
    y_lin = torch.expm1(y_log)
    return torch.clamp(y_lin, min=0.0)


# ===================================================================
#  STEP 1 – Bias correction
# ===================================================================
def step1_bias_correction(
    cfg: PipelineConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the bias-correction UNet on every (time, lead) slice of the AIFS
    low-res predictions.

    Returns
    -------
    bc_precip : (T, L, H_lr, W_lr) – bias-corrected precipitation in mm
    times     : np.ndarray of datetime64
    lead_vals : np.ndarray of timedelta64
    lr_lats, lr_lons : 1-D coordinate arrays
    """
    print("\n" + "=" * 70)
    print("  STEP 1 – Bias Correction")
    print("=" * 70)

    # --- Load AIFS predictions ---
    ds = xr.open_dataset(cfg.aifs_pred_file, engine="netcdf4")
    times_all = ds.time.values
    lead_values = ds.prediction_delta.values
    T, L = len(times_all), len(lead_values)

    # --- Determine crop on STEP-1 (low-res) grid ---
    # STEP 1 operates at native low-res; crop_mult=8 matches training config
    lr_lats = ds.lat.values
    lr_lons = ds.lon.values
    crop_mult_step1 = 8  # must match STEP 1 training config

    H_raw, W_raw = len(lr_lats), len(lr_lons)
    H_crop = (H_raw // crop_mult_step1) * crop_mult_step1
    W_crop = (W_raw // crop_mult_step1) * crop_mult_step1
    lat_s = H_raw - H_crop
    lat_e = H_raw
    lon_s = W_raw - W_crop
    lon_e = W_raw
    H, W = lat_e - lat_s, lon_e - lon_s
    print(f"  LR grid: {H_raw}x{W_raw} → cropped to {H}x{W}  (crop_mult={crop_mult_step1})")

    # --- Load stats for STEP 1 (dynamic + invariant variables) ---
    all_bc_input_vars = cfg.bc_input_variables + cfg.bc_invariant_variables
    bc_inp_min, bc_inp_max, bc_out_min, bc_out_max = load_stats(
        cfg.bc_stats, all_bc_input_vars, [cfg.bc_output_variable]
    )
    n_dyn = len(cfg.bc_input_variables)
    n_inv = len(cfg.bc_invariant_variables)
    n_total = n_dyn + n_inv
    print(f"  Input channels: {n_dyn} dynamic + {n_inv} invariant = {n_total} total")

    # --- Load invariant data (e.g. altitude) on the LR grid ---
    inv_lr = np.zeros((n_inv, H, W), dtype=np.float32)
    for ci, var in enumerate(cfg.bc_invariant_variables):
        if var in ds:
            # Altitude is in the AIFS prediction file itself
            da_inv = ds[var].isel(lat=slice(lat_s, lat_e), lon=slice(lon_s, lon_e))
            v = da_inv.values.astype(np.float32)
            if v.ndim == 3:
                v = v[0]  # take first time slice if (T, H, W)
            elif v.ndim == 4:
                v = v[0, 0]  # (T, L, H, W) → (H, W)
            inv_lr[ci] = v
            print(f"  Loaded invariant '{var}' from AIFS prediction file (LR grid)")
        else:
            # Not in AIFS file → load from altitude_file, interpolate to LR grid
            ds_alt = xr.open_dataset(cfg.altitude_file, engine="netcdf4")
            da_hr = ds_alt[var]
            v_hr = da_hr.values.astype(np.float32)
            if v_hr.ndim == 3:
                v_hr = v_hr[0]
            # Build xr.DataArray on HR grid, then interp to LR grid
            da_tmp = xr.DataArray(
                v_hr,
                dims=("lat", "lon"),
                coords={"lat": ds_alt.lat.values, "lon": ds_alt.lon.values},
            )
            da_lr = da_tmp.interp(
                lat=lr_lats[lat_s:lat_e], lon=lr_lons[lon_s:lon_e], method="linear"
            )
            inv_lr[ci] = np.nan_to_num(da_lr.values.astype(np.float32), nan=0.0)
            ds_alt.close()
            print(f"  Loaded invariant '{var}' from altitude file (interpolated HR→LR)")

    # Sanitize invariants (same as training: clamp≥0, nan→0)
    sanitize(inv_lr)

    # --- Load model ---
    print(f"  Loading bias-correction checkpoint: {os.path.basename(cfg.bc_ckpt)}")
    bc_model = Module.from_checkpoint(cfg.bc_ckpt, {})
    bc_model = bc_model.to(device).eval().to(memory_format=torch.channels_last)
    if hasattr(bc_model, "amp_mode"):
        bc_model.amp_mode = False

    c_out = bc_model.img_out_channels if hasattr(bc_model, "img_out_channels") else 1

    # --- Prepare input tensor per (time, lead) and run ---
    bc_precip = np.full((T, L, H, W), np.nan, dtype=np.float32)

    total = T * L
    print(f"  Running bias correction on {T} times × {L} leads = {total} samples")

    with torch.no_grad():
        pbar = tqdm(total=total, desc="  BC inference")
        for ti in range(T):
            # Build batch: (L, n_total, H, W) = dynamic channels + invariant channels
            batch_x = np.zeros((L, n_total, H, W), dtype=np.float32)

            # Fill dynamic channels
            for ci, var in enumerate(cfg.bc_input_variables):
                da = ds[var].isel(time=ti, lat=slice(lat_s, lat_e), lon=slice(lon_s, lon_e))
                v = da.values.astype(np.float32)  # (L, H, W) or (H, W)
                if v.ndim == 2:
                    v = np.broadcast_to(v[None], (L, H, W))
                batch_x[:, ci, :, :] = v

            # Append invariant channels (same for all leads)
            for li in range(L):
                batch_x[li, n_dyn:, :, :] = inv_lr

            # Sanitize & log-transform on precip (channel 0 only)
            sanitize(batch_x)
            if cfg.log_precip:
                batch_x[:, 0] = np.log1p(batch_x[:, 0])
                np.nan_to_num(batch_x[:, 0], copy=False, nan=0.0, posinf=LOG_CLIP, neginf=-LOG_CLIP)

            # Min-max normalize (all channels: dynamic + invariant)
            for b in range(L):
                batch_x[b] = normalize_minmax(batch_x[b], bc_inp_min, bc_inp_max)

            # Lead labels
            lead_labels = torch.arange(L, dtype=torch.long, device=device)

            # Process in sub-batches
            for b_start in range(0, L, cfg.bc_batch_size):
                b_end = min(b_start + cfg.bc_batch_size, L)
                x_batch = torch.from_numpy(batch_x[b_start:b_end]).to(
                    device=device, dtype=torch.float32, memory_format=torch.channels_last
                )
                leads_batch = lead_labels[b_start:b_end]

                B_sub = x_batch.shape[0]
                x0 = torch.zeros((B_sub, c_out, H, W), device=device)
                y_pred_norm = bc_model(x0, x_batch, lead_time_label=leads_batch)

                if isinstance(y_pred_norm, dict):
                    for key in ("output", "y", "pred"):
                        if key in y_pred_norm:
                            y_pred_norm = y_pred_norm[key]
                            break

                if y_pred_norm.dim() == 3:
                    y_pred_norm = y_pred_norm[:, None, :, :]
                if y_pred_norm.size(1) > 1:
                    y_pred_norm = y_pred_norm[:, 0:1]

                # Denorm + delog → mm
                y_mm = denorm_and_delog_tensor(y_pred_norm, bc_out_min, bc_out_max)
                bc_precip[ti, b_start:b_end, :, :] = y_mm.squeeze(1).cpu().numpy()

            pbar.update(L)
        pbar.close()

    ds.close()
    del bc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    filled = np.isfinite(bc_precip).mean() * 100
    print(f"  Bias-corrected array: shape={bc_precip.shape}, filled={filled:.1f}%")
    return bc_precip, times_all, lead_values, lr_lats[lat_s:lat_e], lr_lons[lon_s:lon_e]


# ===================================================================
#  STEP 1b – Interpolation to high-res grid
# ===================================================================
def step1b_interpolate(
    bc_precip: np.ndarray,
    times: np.ndarray,
    lead_values: np.ndarray,
    lr_lats: np.ndarray,
    lr_lons: np.ndarray,
    hr_lats: np.ndarray,
    hr_lons: np.ndarray,
) -> np.ndarray:
    """
    Bilinearly interpolate bias-corrected low-res precip onto the high-res grid.

    Returns
    -------
    bc_hr : (T, L, H_hr, W_hr) float32
    """
    print("\n" + "=" * 70)
    print("  STEP 1b – Interpolation  LR → HR grid")
    print("=" * 70)

    T, L, _, _ = bc_precip.shape
    H_hr, W_hr = len(hr_lats), len(hr_lons)
    print(f"  LR shape: {bc_precip.shape[2]}x{bc_precip.shape[3]}  →  HR shape: {H_hr}x{W_hr}")

    bc_hr = np.zeros((T, L, H_hr, W_hr), dtype=np.float32)

    # Interpolate horizon-by-horizon to keep memory bounded
    for li in tqdm(range(L), desc="  Interpolating leads"):
        da = xr.DataArray(
            bc_precip[:, li, :, :],
            dims=("time", "lat", "lon"),
            coords={"time": times, "lat": lr_lats, "lon": lr_lons},
        )
        da_hr = da.interp(lat=hr_lats, lon=hr_lons, method="linear")
        bc_hr[:, li, :, :] = da_hr.values.astype(np.float32)

    # Replace any NaN from extrapolation with 0
    np.nan_to_num(bc_hr, copy=False, nan=0.0)
    np.maximum(bc_hr, 0.0, out=bc_hr)
    print(f"  Interpolated shape: {bc_hr.shape}")
    return bc_hr


# ===================================================================
#  Save bias-corrected intermediate as xarray Dataset
# ===================================================================
def save_bc_intermediate(
    bc_hr: np.ndarray,
    times: np.ndarray,
    lead_values: np.ndarray,
    hr_lats_full: np.ndarray,
    hr_lons_full: np.ndarray,
    lat_s: int,
    lat_e: int,
    lon_s: int,
    lon_e: int,
    cfg: PipelineConfig,
):
    """
    Save the bias-corrected, interpolated precipitation as an xarray Dataset.

    Format:  (time, prediction_delta, lat, lon)  matching existing BC files.
    Only the overlap-cropped region is saved.
    """
    print("\n" + "=" * 70)
    print("  Saving bias-corrected intermediate")
    print("=" * 70)

    hr_lats = hr_lats_full[lat_s:lat_e]
    hr_lons = hr_lons_full[lon_s:lon_e]

    # Crop bc_hr to overlap region: (T, L, H_crop, W_crop)
    bc_cropped = bc_hr[:, :, lat_s:lat_e, lon_s:lon_e]

    da = xr.DataArray(
        bc_cropped.astype(np.float64),
        dims=("time", "prediction_delta", "lat", "lon"),
        coords={
            "time": times,
            "prediction_delta": lead_values,
            "lat": hr_lats,
            "lon": hr_lons,
        },
        name="precip6h_cumulative_capped",
    )
    ds_bc = xr.Dataset({"precip6h_cumulative_capped": da})

    # Build output path: same folder as main output, with _bc suffix
    output_dir = os.path.dirname(os.path.abspath(cfg.output_file)) or "."
    base_name = os.path.splitext(os.path.basename(cfg.output_file))[0]
    bc_path = os.path.join(output_dir, f"{base_name}_bias_corrected.nc")

    os.makedirs(output_dir, exist_ok=True)
    ds_bc.to_netcdf(
        bc_path,
        engine="netcdf4",
        encoding={
            "precip6h_cumulative_capped": {
                "zlib": True,
                "complevel": 4,
                "chunksizes": (1, 1, lat_e - lat_s, lon_e - lon_s),
            }
        },
    )
    print(f"  Saved: {bc_path}")
    print(f"  Shape: {bc_cropped.shape}  (time, prediction_delta, lat, lon)")


# ===================================================================
#  STEP 2 – CorrDiff super-resolution
# ===================================================================
def step2_superres(
    cfg: PipelineConfig,
    bc_hr: np.ndarray,
    times: np.ndarray,
    lead_values: np.ndarray,
    hr_lats_full: np.ndarray,
    hr_lons_full: np.ndarray,
    lat_s: int,
    lat_e: int,
    lon_s: int,
    lon_e: int,
    altitude: np.ndarray,
    device: torch.device,
):
    """
    Run CorrDiff regression + diffusion on the bias-corrected, interpolated
    high-res input for every requested (time, horizon) pair.

    Writes results directly to a NetCDF file.
    """
    print("\n" + "=" * 70)
    print("  STEP 2 – CorrDiff Super-Resolution")
    print("=" * 70)

    T_all, L_all = bc_hr.shape[0], bc_hr.shape[1]

    # --- Apply time filtering ---
    time_indices = list(range(0, T_all, cfg.time_stride))
    if cfg.max_times is not None:
        time_indices = time_indices[: cfg.max_times]
    T = len(time_indices)
    print(f"  Using {T} times (stride={cfg.time_stride}, max={cfg.max_times})")

    # --- Determine horizons ---
    if cfg.horizons is None:
        horizons = list(range(L_all))
    else:
        horizons = [h for h in cfg.horizons if 0 <= h < L_all]
    n_leads = len(horizons)
    print(f"  Horizons: {horizons} ({n_leads} total)")

    # --- Use pre-computed overlap crop slices ---
    H = lat_e - lat_s
    W = lon_e - lon_s
    print(f"  HR grid: {len(hr_lats_full)}x{len(hr_lons_full)} → "
          f"overlap-cropped to {H}x{W}  (crop_mult={cfg.crop_mult})")

    hr_lats = hr_lats_full[lat_s:lat_e]
    hr_lons = hr_lons_full[lon_s:lon_e]

    # --- Crop altitude to match ---
    alt_cropped = altitude[lat_s:lat_e, lon_s:lon_e].astype(np.float32)

    # --- Load STEP 2 stats ---
    sr_input_vars = [cfg.sr_input_variable, cfg.invariant_variable]
    sr_output_vars = [cfg.sr_output_variable]
    sr_inp_min, sr_inp_max, sr_out_min, sr_out_max = load_stats(
        cfg.sr_stats, sr_input_vars, sr_output_vars
    )

    # --- Load models ---
    load_reg = cfg.inference_mode in ("regression", "all")
    load_dif = cfg.inference_mode in ("diffusion", "all")

    net_reg, net_res = None, None
    if load_reg:
        print(f"  Loading regression checkpoint: {os.path.basename(cfg.reg_ckpt)}")
        net_reg = Module.from_checkpoint(cfg.reg_ckpt)
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.force_fp16:
            net_reg.use_fp16 = True
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False

    if load_dif:
        print(f"  Loading diffusion checkpoint: {os.path.basename(cfg.dif_ckpt)}")
        net_res = Module.from_checkpoint(cfg.dif_ckpt)
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.force_fp16:
            net_res.use_fp16 = True
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False

    img_shape = (H, W)
    img_out_channels = 1  # single precip channel

    # --- Sampler function ---
    if cfg.sampler_type == "deterministic":
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=cfg.sampler_num_steps,
            solver=cfg.sampler_solver,
        )
    elif cfg.sampler_type == "stochastic":
        sampler_fn = partial(stochastic_sampler, patching=None)
    else:
        raise ValueError(f"Unknown sampler type: {cfg.sampler_type}")

    # --- Pre-process invariant (altitude) ---
    # Sanitize + log + normalize (matching training)
    alt_processed = alt_cropped.copy()
    sanitize(alt_processed.reshape(1, H, W))
    # altitude is NOT log-transformed (only precip channel 0 is)

    # --- Create output NetCDF ---
    output_path = os.path.abspath(cfg.output_file)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    times_used = times[time_indices]
    times_iso = []
    for t in times_used:
        if isinstance(t, np.datetime64):
            times_iso.append(np.datetime_as_string(t, unit="s"))
        else:
            times_iso.append(str(t))

    print(f"  Writing output to: {output_path}")
    f = nc.Dataset(output_path, "w", format="NETCDF4")
    f.createDimension("ensemble", cfg.num_ensembles)
    f.createDimension("time", T)
    f.createDimension("lead", n_leads)
    f.createDimension("y", H)
    f.createDimension("x", W)

    time_var = f.createVariable("time", str, ("time",))
    time_var[:] = np.array(times_iso, dtype=object)

    lead_var = f.createVariable("lead", "i4", ("lead",))
    lead_var[:] = np.array(horizons, dtype=np.int32)

    lat_var = f.createVariable("lat", "f4", ("y",))
    lon_var = f.createVariable("lon", "f4", ("x",))
    lat_var[:] = hr_lats.astype(np.float32)
    lon_var[:] = hr_lons.astype(np.float32)

    # pred = ensemble of downscaled samples
    pred_var = f.createVariable("pred", "f4", ("ensemble", "time", "lead", "y", "x"), zlib=True, complevel=2)

    # --- Generation function (same logic as original generate script) ---
    def generate_fn(
        image_lr: torch.Tensor,
        lead_time_label: torch.Tensor | None,
        rank_batches: list,
    ) -> torch.Tensor:
        img_lr = image_lr.to(memory_format=torch.channels_last)

        image_reg = None
        if net_reg is not None:
            image_reg = regression_step(
                net=net_reg,
                img_lr=img_lr,
                latents_shape=(cfg.seed_batch_size, img_out_channels, H, W),
                lead_time_label=lead_time_label,
            )

        image_res = None
        if net_res is not None:
            mean_hr = image_reg[0:1] if (image_reg is not None and cfg.hr_mean_conditioning) else None
            image_res = diffusion_step(
                net=net_res,
                sampler_fn=sampler_fn,
                img_shape=img_shape,
                img_out_channels=img_out_channels,
                rank_batches=rank_batches,
                img_lr=img_lr.expand(cfg.seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                rank=0,
                device=device,
                mean_hr=mean_hr,
                lead_time_label=lead_time_label,
            )

        if cfg.inference_mode == "regression":
            return image_reg
        if cfg.inference_mode == "diffusion":
            return image_res

        # "all" mode: regression + residual
        B_reg = image_reg.shape[0]
        B_res = image_res.shape[0]
        repeats = B_res // B_reg
        return image_reg.repeat(repeats, 1, 1, 1) + image_res

    # --- Main inference loops ---
    total_samples = T * n_leads
    pbar = tqdm(total=total_samples, desc="  CorrDiff generation")

    with torch.no_grad():
        for lead_pos, hori_idx in enumerate(horizons):
            for t_pos, t_global in enumerate(time_indices):

                # --- Prepare input: precip + altitude → (1, 2, H, W) ---
                precip_slice = bc_hr[t_global, hori_idx, lat_s:lat_e, lon_s:lon_e].copy()

                # Build x_arr: channel 0 = precip, channel 1 = altitude
                x_arr = np.stack([precip_slice, alt_processed], axis=0)  # (2, H, W)

                # Sanitize
                sanitize(x_arr)

                # Log-transform on precip (channel 0 only)
                if cfg.log_precip:
                    x_arr[0] = np.log1p(x_arr[0])
                    np.nan_to_num(x_arr[0:1], copy=False, nan=0.0, posinf=LOG_CLIP, neginf=-LOG_CLIP)

                # Min-max normalize
                x_norm = normalize_minmax(x_arr, sr_inp_min, sr_inp_max)

                # To tensor (1, C, H, W)
                x_t = torch.from_numpy(x_norm[None]).to(
                    device=device, dtype=torch.float32, memory_format=torch.channels_last
                )

                # Skip if all-NaN input
                if torch.isnan(x_t).all():
                    pbar.update(1)
                    continue

                # Seeds for ensembles
                seeds = RNG.integers(low=0, high=2**31 - 1, size=cfg.num_ensembles, dtype=np.int64).tolist()
                num_batches = max(1, (len(seeds) - 1) // cfg.seed_batch_size + 1)
                all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
                rank_batches = list(all_batches)  # single-GPU: all batches go to rank 0

                # Run generation
                image_out = generate_fn(
                    image_lr=x_t,
                    lead_time_label=None,  # STEP 2 model is not lead-time-aware
                    rank_batches=rank_batches,
                )

                # Denorm + delog → mm
                out_mm = denorm_and_delog_tensor(image_out, sr_out_min, sr_out_max)
                out_np = out_mm[:, 0, :, :].cpu().numpy()  # (n_ens, H, W)

                # Write to NetCDF
                n_written = min(out_np.shape[0], cfg.num_ensembles)
                pred_var[:n_written, t_pos, lead_pos, :, :] = out_np[:n_written]

                pbar.update(1)

    pbar.close()
    f.close()

    del net_reg, net_res
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Done! Output saved to: {output_path}")


# ===================================================================
#  Main orchestrator
# ===================================================================
def run_pipeline(cfg: PipelineConfig):
    """Run the full STEP 1 → 1b → 2 pipeline."""

    # --- Distributed / device ---
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    print(f"Device: {device}")

    # ---- STEP 1: bias correction ----
    bc_precip, times, lead_values, lr_lats, lr_lons = step1_bias_correction(cfg, device)

    # ---- Load original (uncropped) LR grid for step2 overlap computation ----
    ds_aifs_tmp = xr.open_dataset(cfg.aifs_pred_file, engine="netcdf4")
    lr_lats_orig = ds_aifs_tmp.lat.values
    lr_lons_orig = ds_aifs_tmp.lon.values
    ds_aifs_tmp.close()

    # ---- Determine high-res grid from altitude file ----
    ds_alt = xr.open_dataset(cfg.altitude_file, engine="netcdf4")
    hr_lats_full = ds_alt.lat.values
    hr_lons_full = ds_alt.lon.values
    altitude_2d = ds_alt[cfg.invariant_variable].values.astype(np.float32)
    if altitude_2d.ndim == 3:
        altitude_2d = altitude_2d[0]  # take first time slice if present
    ds_alt.close()

    # ---- STEP 1b: interpolation LR → HR ----
    bc_hr = step1b_interpolate(
        bc_precip, times, lead_values, lr_lats, lr_lons, hr_lats_full, hr_lons_full
    )
    del bc_precip  # free memory

    # ---- Compute overlap crop (needed for BC save and step2) ----
    lat_s, lat_e, lon_s, lon_e = compute_crop_slices(
        lr_lats_orig, lr_lons_orig, hr_lats_full, hr_lons_full, cfg.crop_mult
    )

    # ---- Optionally save bias-corrected data ----
    if cfg.save_bc:
        save_bc_intermediate(
            bc_hr, times, lead_values,
            hr_lats_full, hr_lons_full,
            lat_s, lat_e, lon_s, lon_e,
            cfg,
        )

    # ---- STEP 2: CorrDiff super-resolution ----
    step2_superres(
        cfg, bc_hr, times, lead_values,
        hr_lats_full, hr_lons_full,
        lat_s, lat_e, lon_s, lon_e,
        altitude_2d, device
    )


# ===================================================================
#  CLI
# ===================================================================
def parse_args() -> PipelineConfig:
    p = argparse.ArgumentParser(
        description="End-to-end precipitation downscaling: bias-correction → interpolation → CorrDiff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Required ---
    p.add_argument("--aifs-pred-file", required=True, help="AIFS low-res predictions (.nc)")
    p.add_argument("--altitude-file", required=True, help="High-res altitude NetCDF (.nc)")
    p.add_argument("--bc-ckpt", required=True, help="Bias-correction model checkpoint (.mdlus)")
    p.add_argument("--reg-ckpt", required=True, help="CorrDiff regression checkpoint (.mdlus)")
    p.add_argument("--dif-ckpt", required=True, help="CorrDiff diffusion checkpoint (.mdlus)")
    p.add_argument("--bc-stats", required=True, help="STEP 1 stats JSON")
    p.add_argument("--sr-stats", required=True, help="STEP 2 stats JSON")

    # --- Output ---
    p.add_argument("--output-file", default="pipeline_output.nc", help="Output NetCDF path")
    p.add_argument("--save-bc", action="store_true",
                   help="Save bias-corrected intermediate as a separate NetCDF (<output>_bias_corrected.nc)")

    # --- Variable names ---
    p.add_argument(
        "--bc-input-variables",
        nargs="+",
        default=DEFAULT_BC_INPUT_VARIABLES,
        help="Input variable names for bias correction model",
    )
    p.add_argument(
        "--bc-invariant-variables",
        nargs="+",
        default=["altitude"],
        help="Invariant (static) variable names for bias correction model (e.g. altitude)",
    )
    p.add_argument("--sr-input-variable", default="precip6h_cumulative_capped")
    p.add_argument("--sr-output-variable", default="precip6h_cumulative_capped")
    p.add_argument("--invariant-variable", default="altitude")

    # --- Generation ---
    p.add_argument("--num-ensembles", type=int, default=12)
    p.add_argument("--seed-batch-size", type=int, default=12)
    p.add_argument("--horizons", type=int, nargs="*", default=None, help="Horizon indices (default: all)")
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--max-times", type=int, default=None)

    # --- Model behaviour ---
    p.add_argument("--inference-mode", choices=["regression", "diffusion", "all"], default="all")
    p.add_argument("--no-hr-mean-conditioning", action="store_true")
    p.add_argument("--crop-mult", type=int, default=16)
    p.add_argument("--no-log-precip", action="store_true")

    # --- Sampler ---
    p.add_argument("--sampler-type", choices=["deterministic", "stochastic"], default="deterministic")
    p.add_argument("--sampler-num-steps", type=int, default=18)
    p.add_argument("--sampler-solver", default="euler")

    # --- Performance ---
    p.add_argument("--bc-batch-size", type=int, default=256)
    p.add_argument("--force-fp16", action="store_true")

    args = p.parse_args()

    return PipelineConfig(
        aifs_pred_file=args.aifs_pred_file,
        altitude_file=args.altitude_file,
        bc_ckpt=args.bc_ckpt,
        reg_ckpt=args.reg_ckpt,
        dif_ckpt=args.dif_ckpt,
        bc_stats=args.bc_stats,
        sr_stats=args.sr_stats,
        output_file=args.output_file,
        save_bc=args.save_bc,
        bc_input_variables=args.bc_input_variables,
        bc_invariant_variables=args.bc_invariant_variables,
        sr_input_variable=args.sr_input_variable,
        sr_output_variable=args.sr_output_variable,
        invariant_variable=args.invariant_variable,
        num_ensembles=args.num_ensembles,
        seed_batch_size=args.seed_batch_size,
        horizons=args.horizons,
        time_stride=args.time_stride,
        max_times=args.max_times,
        inference_mode=args.inference_mode,
        hr_mean_conditioning=not args.no_hr_mean_conditioning,
        crop_mult=args.crop_mult,
        log_precip=not args.no_log_precip,
        sampler_type=args.sampler_type,
        sampler_num_steps=args.sampler_num_steps,
        sampler_solver=args.sampler_solver,
        bc_batch_size=args.bc_batch_size,
        force_fp16=args.force_fp16,
    )


# ===================================================================
#  Entry point
# ===================================================================
if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)