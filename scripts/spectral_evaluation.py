"""
Spectral Analysis for CorrDiff Super-Resolution Pipeline (v4)
==============================================================

Key insight: The spectral analysis must separate TWO questions:
  (a) Does the model predict rain in the right places? (spatial coverage)
  (b) Where it rains, does it have the right spatial structure? (texture/resolution)

The effective resolution question is (b). To answer it, we compute spectra
ONLY within the common wet area (where both pred AND truth have precipitation).
This removes the mask-geometry signal that was dominating all previous analyses.

Approach:
  1. For each timestep, define a common analysis region where BOTH pred and
     truth have precipitation (optionally with a buffer/dilation)
  2. Set both fields to zero outside this region
  3. Apply log-transform: log(P + epsilon)
  4. Compute spectra of these "co-masked" fields
  5. Compare spectra → effective resolution

This is analogous to computing spectra "conditional on rain occurrence" —
a well-established approach in precipitation verification.

Data format:
  - Each .nc file contains "pred" (sample, time, horizon, y, x)
    and "truth" (time, y, x)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import warnings
from scipy.signal.windows import tukey
from scipy.ndimage import binary_dilation
import pickle

_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path("output/generations/")

# Model files: dict of {model_name: filename}
# Each .nc file contains "pred" and "truth".
#
# For MULTI-HORIZON models: pred has shape (sample, time, horizon, y, x)
# For SINGLE-HORIZON models: pred has shape (sample, time, 1, y, x)
#   → specify which global horizon it corresponds to in MODEL_HORIZON_MAP
MODEL_FILES = {
    "UNet_lam2": "UNet_lam2_ValSampleStride4/UNet_lam2_ValSampleStride4_samples_clean.nc",
    "MultiFT2": "UNet_MultiStepFineTuning2_ValSampleStride4/UNet_MultiStepFineTuning2_ValSampleStride4_samples_clean.nc",
    "B0: AIFS": "AIFSOrigPreds_ValSampleStride4/AIFSOrigPreds_ValSampleStride4_samples_clean.nc",
    "B4: Pixel LR, MultiVar ": "BaselineB4_ValSampleStride4/BaselineB4_ValSampleStride4_samples_clean.nc",
    "UNet Single 6h": "SingleLead_6h_ValSampleStride4/SingleLead_6h_ValSampleStride4_samples_clean.nc",
    "UNet Single 3days": "SingleLead_3days_ValSampleStride4/SingleLead_3days_ValSampleStride4_samples_clean.nc",
    "UNet Single 6days": "SingleLead_6days_ValSampleStride4/SingleLead_6days_ValSampleStride4_samples_clean.nc",
}

# For single-horizon models, map model_name -> horizon label(s) they cover.
# Models not listed here are assumed to have ALL horizons (multi-horizon).
# The value should be a list of horizon labels matching HORIZON_LABELS.
MODEL_HORIZON_MAP = {
    "UNet Single 6h":    ["6h"],
    "UNet Single 3days": ["3days"],
    "UNet Single 6days": ["6days"],
}
# Set to {} or None if all models are multi-horizon.

PRED_VAR = "pred"
TRUTH_VAR = "truth"

DX = 1.0  # km
DY = 1.0  # km

HORIZON_LABELS = ["6h", "12h", "18h", "24h", "2days", "3days", "4days", "5days", "6days"]

# Select specific horizons to compute (for fast testing).
# Set to None to compute all, or a list of indices, e.g. [0, 4, 8]
HORIZONS_TO_COMPUTE = [0,5,8]  # fast test: first, middle, last

# Spectral parameters
N_RADIAL_BINS = None
WINDOW_TYPE = "tukey"
TUKEY_ALPHA = 0.1

# Log-transform offset (same unit as your precip data)
LOG_OFFSET = 0.1  # Using 0.1 to match your threshold

# Wet threshold for defining the analysis mask
WET_THRESHOLD = 0.1  # mm/6h (match your threshold)

# Buffer (dilation) around wet areas in pixels.
# Smooths mask boundary to reduce edge artifacts in the FFT.
# For scattered precipitation, keep at 0 — the log-transform already
# softens boundaries. Only increase if your precipitation is concentrated
# in a few large connected regions.
MASK_DILATION_PIXELS = 0

# Masking strategy:
#   "intersection" : only where BOTH pred and truth are wet (strictest)
#   "union"        : where EITHER pred or truth is wet (most inclusive)
#   "truth_only"   : where truth is wet (doesn't use model info for mask)
MASK_STRATEGY = "intersection"

# Minimum fraction of domain that must be in the common wet area
MIN_WET_FRACTION = 0.03

# Wavelength bands (km)
SCALE_BANDS = {
    "large":  (100, 600),
    "meso":   (20, 100),
    "small":  (2, 20),
}

EFF_RES_THRESHOLD = 0.5

OUTPUT_DIR = Path("output/generations/spectral_results_v4_AllModels_Final")

# =============================================================================
# Core spectral functions
# =============================================================================

def apply_window_2d(field, window_type="tukey", alpha=0.1):
    ny, nx = field.shape
    if window_type == "none":
        return field
    elif window_type == "tukey":
        wy = tukey(ny, alpha=alpha)
        wx = tukey(nx, alpha=alpha)
    elif window_type == "hann":
        wy = np.hanning(ny)
        wx = np.hanning(nx)
    else:
        raise ValueError(f"Unknown window: {window_type}")
    return field * np.outer(wy, wx)


def compute_radial_psd(field, dx, dy=None, n_bins=None,
                       window="tukey", alpha=0.1):
    """Compute radially-averaged PSD from a preprocessed, detrended 2D field."""
    if dy is None:
        dy = dx
    ny, nx = field.shape
    f = np.nan_to_num(field.copy().astype(np.float64), nan=0.0)
    f = f - np.mean(f)
    f = apply_window_2d(f, window_type=window, alpha=alpha)

    F = np.fft.fftshift(np.fft.fft2(f))
    psd_2d = (np.abs(F) ** 2) * (dx * dy) / (nx * ny)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    k_max = min(np.abs(kx).max(), np.abs(ky).max())

    if n_bins is None:
        n_bins = min(ny, nx) // 2
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    psd_radial = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
        if mask.sum() > 0:
            psd_radial[i] = np.mean(psd_2d[mask])

    valid = (~np.isnan(psd_radial)) & (k_centers > 0)
    k_centers = k_centers[valid]
    psd_radial = psd_radial[valid]
    wavelength_km = 1.0 / k_centers
    idx = np.argsort(wavelength_km)[::-1]
    return wavelength_km[idx], psd_radial[idx]


# =============================================================================
# Common-mask construction
# =============================================================================

def build_common_mask(truth_field, pred_field, strategy="intersection",
                      wet_threshold=0.1, dilation_pixels=0):
    """
    Build a common analysis mask where both/either fields have precipitation.

    Parameters
    ----------
    truth_field : (ny, nx) - single truth timestep
    pred_field : (ny, nx) - single pred sample/timestep
    strategy : "intersection", "union", or "truth_only"
    wet_threshold : threshold to define "wet"
    dilation_pixels : dilate the mask by this many pixels

    Returns
    -------
    mask : (ny, nx) boolean array, True where analysis should be done
    """
    truth_wet = truth_field > wet_threshold
    pred_wet = pred_field > wet_threshold

    if strategy == "intersection":
        mask = truth_wet & pred_wet
    elif strategy == "union":
        mask = truth_wet | pred_wet
    elif strategy == "truth_only":
        mask = truth_wet
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Dilate to include surrounding area (smoother mask boundary)
    if dilation_pixels > 0:
        struct = np.ones((2*dilation_pixels+1, 2*dilation_pixels+1))
        mask = binary_dilation(mask, structure=struct)

    return mask


def apply_mask_and_log(field, mask, log_offset=0.1):
    """
    Apply common mask and log-transform.
    Pixels outside mask get log(offset) — same as zero precipitation.
    Pixels inside mask get log(P + offset).
    """
    result = np.full_like(field, np.log(log_offset), dtype=np.float64)
    result[mask] = np.log(np.maximum(field[mask], 0) + log_offset)
    return result


# =============================================================================
# Batch computation
# =============================================================================

def compute_comasked_spectra(
    truth_all, pred_all_samples, dx, dy=None,
    wet_threshold=0.1, mask_strategy="intersection",
    dilation_pixels=0, log_offset=0.1,
    min_wet_fraction=0.03,
    n_bins=None, window="tukey", alpha=0.1,
):
    """
    Compute time-averaged spectra for truth and pred using co-masking.

    For each timestep:
      1. For each sample, build a common mask (truth ∩ pred_sample)
      2. Apply mask + log-transform to both truth and pred_sample
      3. Compute PSD of each

    Then average over time, and over samples (with std across samples).

    Parameters
    ----------
    truth_all : (n_time, ny, nx)
    pred_all_samples : (n_samples, n_time, ny, nx)

    Returns
    -------
    wavelength_km, psd_truth_mean, psd_pred_mean, psd_pred_std, n_valid, wet_fractions
    """
    n_samples, n_time, ny, nx = pred_all_samples.shape
    n_pixels = ny * nx

    # Collect per-sample time-averaged spectra
    truth_spectra_per_sample = []  # one truth spectrum per sample (masked differently)
    pred_spectra_per_sample = []
    wet_fractions = []
    n_valid_total = 0

    for s in range(n_samples):
        truth_spectra_t = []
        pred_spectra_t = []

        for t in range(n_time):
            truth_t = truth_all[t]
            pred_t = pred_all_samples[s, t]

            # Build common mask
            mask = build_common_mask(
                truth_t, pred_t, strategy=mask_strategy,
                wet_threshold=wet_threshold,
                dilation_pixels=dilation_pixels
            )

            wet_frac = mask.sum() / n_pixels
            if wet_frac < min_wet_fraction:
                continue

            wet_fractions.append(wet_frac)
            n_valid_total += 1

            # Apply mask + log
            truth_masked = apply_mask_and_log(truth_t, mask, log_offset)
            pred_masked = apply_mask_and_log(pred_t, mask, log_offset)

            # Compute PSD
            wl_t, psd_t = compute_radial_psd(
                truth_masked, dx, dy, n_bins=n_bins, window=window, alpha=alpha)
            wl_p, psd_p = compute_radial_psd(
                pred_masked, dx, dy, n_bins=n_bins, window=window, alpha=alpha)

            if len(wl_t) > 0 and len(wl_p) > 0:
                truth_spectra_t.append(psd_t)
                pred_spectra_t.append(psd_p)

        if len(truth_spectra_t) > 0:
            truth_spectra_per_sample.append(np.mean(truth_spectra_t, axis=0))
            pred_spectra_per_sample.append(np.mean(pred_spectra_t, axis=0))

    if len(pred_spectra_per_sample) == 0:
        return (np.array([]),) * 4 + (0, [])

    truth_spectra = np.array(truth_spectra_per_sample)
    pred_spectra = np.array(pred_spectra_per_sample)

    return (
        wl_t,  # wavelength grid (same for all, since same field size)
        np.mean(truth_spectra, axis=0),   # truth PSD (averaged over samples)
        np.mean(pred_spectra, axis=0),    # pred PSD mean
        np.std(pred_spectra, axis=0),     # pred PSD std across samples
        n_valid_total,
        wet_fractions,
    )


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(wavelength_km, psd_model, psd_observed,
                    scale_bands, threshold=0.5):
    """Compute all spectral metrics."""
    ratio = psd_model / (psd_observed + 1e-30)
    log_ratio_db = 10 * np.log10(ratio + 1e-30)

    # Band-averaged ratios
    band_ratios = {}
    for bname, (wl_min, wl_max) in scale_bands.items():
        mask = (wavelength_km >= wl_min) & (wavelength_km <= wl_max)
        if mask.sum() > 0:
            band_ratios[bname] = (
                np.mean(psd_model[mask]) / (np.mean(psd_observed[mask]) + 1e-30)
            )
        else:
            band_ratios[bname] = np.nan

    # Spectral bias
    log_ratio = np.log10(psd_model / (psd_observed + 1e-30) + 1e-30)
    log_wl = np.log10(wavelength_km)
    deficit_signed = _trapz(log_ratio, log_wl)

    # Effective resolution
    below = ratio < threshold
    if not np.any(below):
        eff_res = 2 * wavelength_km[-1]
    else:
        idx = np.where(below)[0][0]
        if idx == 0:
            eff_res = wavelength_km[0]
        else:
            w1, w2 = wavelength_km[idx-1], wavelength_km[idx]
            r1, r2 = ratio[idx-1], ratio[idx]
            if r1 != r2:
                frac = (threshold - r1) / (r2 - r1)
                eff_res = w1 + frac * (w2 - w1)
            else:
                eff_res = w1

    return {
        "ratio": ratio,
        "log_ratio_db": log_ratio_db,
        "band_ratios": band_ratios,
        "spectral_bias": deficit_signed,
        "eff_res_km": eff_res,
    }


# =============================================================================
# Data loading
# =============================================================================

def load_all_data(data_dir, model_files, pred_var="pred", truth_var="truth"):
    preds = {}
    truth = None
    for name, fname in model_files.items():
        filepath = Path(data_dir) / fname
        print(f"Loading '{name}' from {fname}...")
        ds = xr.open_dataset(filepath)
        pred_da = ds[pred_var]
        print(f"  pred shape: {pred_da.shape}, dims: {pred_da.dims}")
        preds[name] = pred_da
        if truth is None:
            truth = ds[truth_var]
            print(f"  truth shape: {truth.shape}, dims: {truth.dims}")
        else:
            other_truth = ds[truth_var]
            if not np.allclose(truth.values[:3], other_truth.values[:3], equal_nan=True):
                warnings.warn(f"Truth in '{fname}' differs!")
    return preds, truth


# =============================================================================
# Main analysis
# =============================================================================

def run_spectral_analysis(
    preds, truth,
    dx=1.0, dy=1.0,
    horizon_labels=None,
    horizons_to_compute=None,
    model_horizon_map=None,
    scale_bands=None,
    eff_res_threshold=0.5,
    log_offset=0.1,
    wet_threshold=0.1,
    mask_strategy="intersection",
    dilation_pixels=5,
    min_wet_fraction=0.03,
    output_dir=Path("./spectral_results_v4"),
    n_bins=None, window="tukey", alpha=0.1,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if scale_bands is None:
        scale_bands = SCALE_BANDS

    truth_np = truth.values
    n_time = truth_np.shape[0]

    first_pred = list(preds.values())[0]
    n_horizons = first_pred.shape[2]
    if horizon_labels is None:
        horizon_labels = [f"h{i}" for i in range(n_horizons)]

    # Filter horizons if requested
    if horizons_to_compute is not None:
        horizon_indices = horizons_to_compute
        horizon_labels = [horizon_labels[i] for i in horizon_indices]
    else:
        horizon_indices = list(range(n_horizons))

    print(f"\nGrid: {dx} km, Domain: {truth_np.shape[1]} x {truth_np.shape[2]}")
    print(f"Time steps: {n_time}, Horizons: {n_horizons}")
    print(f"Models: {list(preds.keys())}")
    print(f"Mask strategy: {mask_strategy}, dilation: {dilation_pixels}px")
    print(f"Log offset: {log_offset}, Wet threshold: {wet_threshold}")

    if model_horizon_map is None:
        model_horizon_map = {}

    # Build per-model mapping: model_name -> list of (hi, h) pairs
    # where hi = index into horizon_labels, h = index into pred array
    model_horizon_indices = {}
    for model_name in preds:
        if model_name in model_horizon_map:
            # Single-horizon model: find which horizon_labels match
            allowed_labels = model_horizon_map[model_name]
            pairs = []
            for hi, hlabel in enumerate(horizon_labels):
                if hlabel in allowed_labels:
                    # The pred array has only 1 horizon (or as many as allowed_labels)
                    # Map to the index within the pred array
                    pred_h_idx = allowed_labels.index(hlabel)
                    pairs.append((hi, pred_h_idx))
            model_horizon_indices[model_name] = pairs
            if pairs:
                print(f"  {model_name}: single-horizon, mapped to {allowed_labels}")
            else:
                print(f"  WARNING: {model_name} horizons {allowed_labels} "
                      f"not found in {horizon_labels}")
        else:
            # Multi-horizon model: compute all requested horizons
            # hi = sequential index into horizon_labels
            # horizon_indices[hi] = actual index into pred array
            model_horizon_indices[model_name] = [
                (hi, horizon_indices[hi])
                for hi in range(len(horizon_labels))
            ]

    results = {
        "models": {},
        "horizon_labels": horizon_labels,
        "dx": dx,
        "scale_bands": scale_bands,
    }

    for model_name, pred_da in preds.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        pred_np = pred_da.values  # (sample, time, horizon, y, x)

        model_results = {}

        for hi, pred_h_idx in model_horizon_indices[model_name]:
            print(f"\n  Horizon {horizon_labels[hi]}:")

            pred_h = pred_np[:, :, pred_h_idx, :, :]  # (n_samples, n_time, ny, nx)

            wl, psd_truth, psd_pred, psd_pred_std, n_valid, wet_fracs = \
                compute_comasked_spectra(
                    truth_np, pred_h, dx, dy,
                    wet_threshold=wet_threshold,
                    mask_strategy=mask_strategy,
                    dilation_pixels=dilation_pixels,
                    log_offset=log_offset,
                    min_wet_fraction=min_wet_fraction,
                    n_bins=n_bins, window=window, alpha=alpha,
                )

            if len(wl) == 0:
                print("    SKIPPED (no valid timesteps)")
                model_results[hi] = None
                continue

            mean_wet_frac = np.mean(wet_fracs) if wet_fracs else 0
            print(f"    Valid pairs: {n_valid}, "
                  f"Mean wet fraction: {mean_wet_frac:.1%}")

            metrics = compute_metrics(wl, psd_pred, psd_truth,
                                       scale_bands, eff_res_threshold)

            model_results[hi] = {
                "wavelength": wl,
                "psd_truth": psd_truth,
                "psd_pred_mean": psd_pred,
                "psd_pred_std": psd_pred_std,
                "metrics": metrics,
                "n_valid": n_valid,
                "mean_wet_fraction": mean_wet_frac,
            }

            br = metrics["band_ratios"]
            print(f"    Eff. resolution: {metrics['eff_res_km']:.1f} km")
            print(f"    Band ratios — "
                  + " | ".join(f"{k}: {v:.3f}" for k, v in br.items()))
            print(f"    Spectral bias: {metrics['spectral_bias']:.3f}")

        results["models"][model_name] = model_results

    # --- Plots ---
    print(f"\n{'='*60}")
    print("Generating plots...")
    _plot_spectra_grid(results, output_dir)
    _plot_ratio_grid(results, output_dir)
    _plot_metrics_summary(results, output_dir)
    _plot_detailed_per_horizon(results, output_dir)
    _plot_composite_figure(results, output_dir)
    _plot_composite_figure_subpanels(results, output_dir)  # optional

    _save_metrics_csv(results, output_dir)

    save_results(results, output_dir / "spectral_results.pkl")

    print(f"All outputs saved to: {output_dir.resolve()}")
    return results


# =============================================================================
# Plotting
# =============================================================================

def _get_colors(names):
    cmap = plt.cm.tab10
    return {n: cmap(i) for i, n in enumerate(names)}


def _plot_spectra_grid(results, output_dir):
    """Spectra grid: co-masked truth vs pred per horizon."""
    hl = results["horizon_labels"]
    n_h = len(hl)
    ncols = min(3, n_h)
    nrows = int(np.ceil(n_h / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4*nrows), squeeze=False)
    colors = _get_colors(results["models"].keys())

    for h, hlabel in enumerate(hl):
        r, c = divmod(h, ncols)
        ax = axes[r][c]

        first_drawn = False
        for mname, mdata in results["models"].items():
            if mdata.get(h) is None:
                continue
            d = mdata[h]
            col = colors[mname]

            # Plot co-masked truth (may vary slightly per model due to intersection)
            if not first_drawn:
                ax.loglog(d["wavelength"], d["psd_truth"], "k-", lw=2.5,
                          label="Observed (co-masked)", zorder=10)
                first_drawn = True

            ax.loglog(d["wavelength"], d["psd_pred_mean"], color=col, lw=1.5,
                      label=mname)
            ax.fill_between(d["wavelength"],
                            d["psd_pred_mean"] - d["psd_pred_std"],
                            d["psd_pred_mean"] + d["psd_pred_std"],
                            alpha=0.12, color=col)

        ax.set_title(f"{hlabel}", fontsize=11)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.2, which="both")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        if r == nrows-1: ax.set_xlabel("Wavelength (km)")
        if c == 0: ax.set_ylabel("PSD [co-masked log(P+ε)]")
        if h == 0: ax.legend(fontsize=7, loc="lower left")

    for h in range(n_h, nrows*ncols):
        axes[*divmod(h, ncols)].set_visible(False)

    fig.suptitle("Co-Masked Log-Precip Spectra — All Horizons", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "spectra_comasked_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: spectra_comasked_grid.png")


def _plot_ratio_grid(results, output_dir):
    """Spectral ratio grid with scale bands."""
    hl = results["horizon_labels"]
    n_h = len(hl)
    ncols = min(3, n_h)
    nrows = int(np.ceil(n_h / ncols))
    scale_bands = results["scale_bands"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 3.5*nrows), squeeze=False)
    colors = _get_colors(results["models"].keys())
    band_colors = {"large": "#fee0d2", "meso": "#e5f5e0", "small": "#deebf7"}

    for h, hlabel in enumerate(hl):
        r, c = divmod(h, ncols)
        ax = axes[r][c]

        for mname, mdata in results["models"].items():
            if mdata.get(h) is None:
                continue
            d = mdata[h]
            col = colors[mname]
            eff = d["metrics"]["eff_res_km"]
            ax.semilogx(d["wavelength"], d["metrics"]["ratio"],
                        color=col, lw=1.5,
                        label=f"{mname} (eff={eff:.1f}km)")

        ax.axhline(1.0, color="gray", alpha=0.4)
        ax.axhline(0.5, color="black", ls=":", alpha=0.5)

        for bname, (wl_min, wl_max) in scale_bands.items():
            bc = band_colors.get(bname, "#f0f0f0")
            ax.axvspan(wl_min, wl_max, alpha=0.2, color=bc, zorder=0)
            ax.text(np.sqrt(wl_min*wl_max), 0.97, bname, ha="center",
                    va="top", fontsize=7, alpha=0.5, style="italic",
                    transform=ax.get_xaxis_transform())

        ax.set_title(hlabel, fontsize=11)
        ax.invert_xaxis()
        ax.set_ylim(0, 2.0)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        if r == nrows-1: ax.set_xlabel("Wavelength (km)")
        if c == 0: ax.set_ylabel("PSD ratio (pred/obs)")
        if h == 0: ax.legend(fontsize=7)

    for h in range(n_h, nrows*ncols):
        axes[*divmod(h, ncols)].set_visible(False)

    fig.suptitle("Spectral Ratio (co-masked) — All Horizons", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "ratio_comasked_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: ratio_comasked_grid.png")


def _plot_metrics_summary(results, output_dir):
    """Summary: eff res + band ratios + bias across horizons."""
    hl = results["horizon_labels"]
    model_names = list(results["models"].keys())
    colors = _get_colors(model_names)
    scale_bands = results["scale_bands"]
    band_names = list(scale_bands.keys())
    x = np.arange(len(hl))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Top: Effective resolution
    ax = axes[0]
    for mn in model_names:
        md = results["models"][mn]
        vals = [md[h]["metrics"]["eff_res_km"] if md.get(h) else np.nan
                for h in range(len(hl))]
        ax.plot(x, vals, "o-", color=colors[mn], lw=2, ms=6, label=mn)
    ax.axhline(2*results["dx"], color="red", ls=":", alpha=0.5,
               label=f"Nyquist ({2*results['dx']} km)")
    ax.set_ylabel("Effective Resolution (km)")
    ax.set_title("Effective Resolution (co-masked log-precip)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Middle: Spectral bias
    ax = axes[1]
    for mn in model_names:
        md = results["models"][mn]
        vals = [md[h]["metrics"]["spectral_bias"] if md.get(h) else np.nan
                for h in range(len(hl))]
        ax.plot(x, vals, "s-", color=colors[mn], lw=2, ms=6, label=mn)
    ax.axhline(0, color="black", alpha=0.3)
    ax.set_ylabel("Spectral Bias")
    ax.set_title("Integrated Spectral Bias (negative = under-dispersive)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom: Band ratios
    ax = axes[2]
    ls_map = {"large": "-", "meso": "--", "small": ":"}
    for mn in model_names:
        md = results["models"][mn]
        for bn in band_names:
            vals = [md[h]["metrics"]["band_ratios"].get(bn, np.nan)
                    if md.get(h) else np.nan
                    for h in range(len(hl))]
            ax.plot(x, vals, marker="o", ls=ls_map.get(bn, "-"),
                    color=colors[mn], lw=1.5, ms=4,
                    label=f"{mn} ({bn})")
    ax.axhline(1.0, color="black", alpha=0.3)
    ax.set_ylabel("Band-avg PSD ratio")
    ax.set_title("Scale-Dependent Spectral Ratio (1.0 = perfect)")
    ax.set_xticks(x)
    ax.set_xticklabels(hl)
    ax.set_xlabel("Forecast Horizon")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 2.0)

    plt.tight_layout()
    fig.savefig(output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: metrics_summary.png")


def _plot_detailed_per_horizon(results, output_dir):
    """Per-horizon: spectra + ratio + dB ratio."""
    hl = results["horizon_labels"]
    colors = _get_colors(results["models"].keys())
    scale_bands = results["scale_bands"]
    band_colors = {"large": "#fee0d2", "meso": "#e5f5e0", "small": "#deebf7"}

    for h, hlabel in enumerate(hl):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        for mname, mdata in results["models"].items():
            if mdata.get(h) is None:
                continue
            d = mdata[h]
            col = colors[mname]
            eff = d["metrics"]["eff_res_km"]

            # Left: Spectra
            ax1.loglog(d["wavelength"], d["psd_truth"], "k-", lw=2.5,
                       label="Observed (co-masked)")
            ax1.loglog(d["wavelength"], d["psd_pred_mean"], color=col, lw=2,
                       label=mname)
            ax1.fill_between(d["wavelength"],
                             d["psd_pred_mean"] - d["psd_pred_std"],
                             d["psd_pred_mean"] + d["psd_pred_std"],
                             alpha=0.12, color=col)

            # Middle: Linear ratio
            ax2.semilogx(d["wavelength"], d["metrics"]["ratio"],
                         color=col, lw=2,
                         label=f"{mname} (eff={eff:.1f}km)")

            # Right: dB ratio
            ax3.semilogx(d["wavelength"], d["metrics"]["log_ratio_db"],
                         color=col, lw=2, label=mname)

        ax1.set_xlabel("Wavelength (km)")
        ax1.set_ylabel("PSD")
        ax1.set_title(f"Co-masked Spectra — {hlabel}")
        ax1.legend(fontsize=9)
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3, which="both")
        ax1.xaxis.set_major_formatter(ScalarFormatter())

        ax2.axhline(1.0, color="gray", alpha=0.3)
        ax2.axhline(0.5, color="black", ls=":", alpha=0.5, label="Threshold")
        for bn, (lo, hi) in scale_bands.items():
            ax2.axvspan(lo, hi, alpha=0.15, color=band_colors.get(bn, "#f0f0f0"))
        ax2.set_xlabel("Wavelength (km)")
        ax2.set_ylabel("PSD ratio")
        ax2.set_title(f"Spectral Ratio — {hlabel}")
        ax2.legend(fontsize=8)
        ax2.invert_xaxis()
        ax2.set_ylim(0, 2.0)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(ScalarFormatter())

        ax3.axhline(0, color="black", lw=1, alpha=0.8)
        for bn, (lo, hi) in scale_bands.items():
            ax3.axvspan(lo, hi, alpha=0.15, color=band_colors.get(bn, "#f0f0f0"))
        ax3.set_xlabel("Wavelength (km)")
        ax3.set_ylabel("PSD ratio (dB)")
        ax3.set_title(f"Log Spectral Ratio — {hlabel}")
        ax3.legend(fontsize=9)
        ax3.invert_xaxis()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(ScalarFormatter())

        plt.tight_layout()
        fig.savefig(output_dir / f"detailed_{hlabel}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved: detailed_*.png ({len(hl)} files)")


# ── Colour palette ──────────────────────────────────────────────────────────
# Adjust or extend as needed.  Designed for ≤8 models with good contrast.
MODEL_COLORS = {
    "UNet_lam2":                  "#1f77b4",   # blue
    "MultiFT2":                   "#d62728",   # red
    "B0: AIFS":                   "#7f7f7f",   # gray
    "B4: Pixel LR, MultiVar ":   "#ff7f0e",   # orange
    "UNet Single 6h":             "#2ca02c",   # green
    "UNet Single 3days":          "#9467bd",   # purple
    "UNet Single 6days":          "#8c564b",   # brown
}
 
MODEL_LINESTYLES = {
    "UNet_lam2":                  "-",
    "MultiFT2":                   "-",
    "B0: AIFS":                   "--",
    "B4: Pixel LR, MultiVar ":   "--",
    "UNet Single 6h":             "-.",
    "UNet Single 3days":          "-.",
    "UNet Single 6days":          "-.",
}
 
# Scale-band shading
BAND_COLORS = {"large": "#fee0d2", "meso": "#e5f5e0", "small": "#deebf7"}
 
 
def _color_for(model_name):
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    # fallback: cycle through tab10
    idx = hash(model_name) % 10
    return plt.cm.tab10(idx)
 
 
def _ls_for(model_name):
    return MODEL_LINESTYLES.get(model_name, "-")
 
 
# ── Main composite figure ──────────────────────────────────────────────────
 
def _plot_composite_figure(
    results,
    output_dir,
    fname="spectral_composite.pdf",
    figsize_per_panel=(4.5, 3.5),
    ratio_ylim=(0, 1.8),
    show_band_shading=True,
    truth_envelope_color="#bdbdbd",
    truth_envelope_alpha=0.35,
    truth_median_color="black",
    dpi=200,
):
    """
    Create a 2-row × N_hz composite spectral figure with a single
    unified legend anchored below all panels.
 
    Parameters
    ----------
    results : dict
        Output of run_spectral_analysis().
    output_dir : Path
        Where to save the figure.
    fname : str
        Output filename (supports .pdf, .png, .svg).
    figsize_per_panel : tuple
        (width, height) per subplot panel.
    ratio_ylim : tuple
        y-axis limits for the ratio panels.
    show_band_shading : bool
        Shade large / meso / small wavelength bands.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    hl = results["horizon_labels"]
    n_hz = len(hl)
    scale_bands = results["scale_bands"]
    model_names = list(results["models"].keys())
 
    fw = figsize_per_panel[0] * n_hz
    fh = figsize_per_panel[1] * 2
    fig = plt.figure(figsize=(fw, fh))
 
    # Extra bottom margin for the legend
    gs = gridspec.GridSpec(
        2, n_hz, figure=fig,
        hspace=0.38, wspace=0.30,
        left=0.07, right=0.97, bottom=0.15, top=0.92,
    )
 
    # ── Top row: spectral ratio ─────────────────────────────────────────
    ratio_axes = []
    for col, hlabel in enumerate(hl):
        ax = fig.add_subplot(gs[0, col])
        ratio_axes.append(ax)
        h = col
 
        for mname in model_names:
            mdata = results["models"][mname]
            if mdata.get(h) is None:
                continue
            d = mdata[h]
            ax.semilogx(
                d["wavelength"], d["metrics"]["ratio"],
                color=_color_for(mname), ls=_ls_for(mname),
                lw=1.6,
            )
 
        # Reference lines
        ax.axhline(1.0, color="gray", alpha=0.4, lw=0.8)
        ax.axhline(0.5, color="black", ls=":", alpha=0.5, lw=0.8)
 
        # Band shading
        if show_band_shading:
            for bname, (wl_lo, wl_hi) in scale_bands.items():
                ax.axvspan(wl_lo, wl_hi, alpha=0.15,
                           color=BAND_COLORS.get(bname, "#f0f0f0"), zorder=0)
                ax.text(
                    np.sqrt(wl_lo * wl_hi), 0.97, bname,
                    ha="center", va="top", fontsize=7, alpha=0.45,
                    style="italic",
                    transform=ax.get_xaxis_transform(),
                )
 
        ax.set_title(hlabel, fontsize=11, fontweight="bold")
        ax.invert_xaxis()
        ax.set_ylim(ratio_ylim)
        ax.grid(True, alpha=0.15, which="both")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(labelsize=8)
 
        if col == 0:
            ax.set_ylabel("PSD ratio (pred / obs)", fontsize=9)
        else:
            ax.set_yticklabels([])
 
    # # Shared x-label via annotation
    # fig.text(0.52, 0.50, "Wavelength (km)", ha="center", fontsize=9)
 
    # ── Bottom row: co-masked spectra ───────────────────────────────────
    spec_axes = []
    for col, hlabel in enumerate(hl):
        ax = fig.add_subplot(gs[1, col])
        spec_axes.append(ax)
        h = col
 
        # Collect all co-masked truth spectra to form the envelope
        truth_spectra = []
        wl_ref = None
        for mname in model_names:
            mdata = results["models"][mname]
            if mdata.get(h) is None:
                continue
            d = mdata[h]
            truth_spectra.append(d["psd_truth"])
            if wl_ref is None:
                wl_ref = d["wavelength"]
 
        if wl_ref is None or len(truth_spectra) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
 
        truth_stack = np.array(truth_spectra)
        truth_lo = np.min(truth_stack, axis=0)
        truth_hi = np.max(truth_stack, axis=0)
        truth_med = np.median(truth_stack, axis=0)
 
        # Truth envelope
        ax.fill_between(
            wl_ref, truth_lo, truth_hi,
            color=truth_envelope_color, alpha=truth_envelope_alpha,
        )
        ax.loglog(
            wl_ref, truth_med,
            color=truth_median_color, lw=2.0, ls="-", zorder=8,
        )
 
        # Model predictions
        for mname in model_names:
            mdata = results["models"][mname]
            if mdata.get(h) is None:
                continue
            d = mdata[h]
            ax.loglog(
                d["wavelength"], d["psd_pred_mean"],
                color=_color_for(mname), ls=_ls_for(mname),
                lw=1.4,
            )
            if d["psd_pred_std"] is not None:
                ax.fill_between(
                    d["wavelength"],
                    np.maximum(d["psd_pred_mean"] - d["psd_pred_std"], 1e-15),
                    d["psd_pred_mean"] + d["psd_pred_std"],
                    alpha=0.08, color=_color_for(mname),
                )
 
        ax.set_title(hlabel, fontsize=11, fontweight="bold")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.15, which="both")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Wavelength (km)", fontsize=9)
 
        if col == 0:
            ax.set_ylabel("PSD [log(P + ε)]", fontsize=9)
        else:
            ax.set_yticklabels([])
 
    # ── Row labels on the left margin ───────────────────────────────────
    fig.text(
        0.01, 0.73, "Spectral\nRatio", rotation=90,
        ha="center", va="center", fontsize=10, fontweight="bold",
    )
    fig.text(
        0.01, 0.33, "Co-masked\nSpectra", rotation=90,
        ha="center", va="center", fontsize=10, fontweight="bold",
    )
 
    # ── Unified legend at the bottom ────────────────────────────────────
    legend_handles = []
    # Models
    for mname in model_names:
        legend_handles.append(
            Line2D([0], [0], color=_color_for(mname), ls=_ls_for(mname),
                   lw=2.0, label=mname)
        )
    # Observed (spectra row)
    legend_handles.append(
        Line2D([0], [0], color=truth_median_color, ls="-", lw=2.0,
               label="Observed (median)")
    )
    legend_handles.append(
        Patch(facecolor=truth_envelope_color, alpha=truth_envelope_alpha,
              edgecolor="none", label="Observed (co-mask envelope)")
    )
    # Reference lines (ratio row)
    legend_handles.append(
        Line2D([0], [0], color="black", ls=":", lw=0.8, alpha=0.6,
               label="0.5 threshold")
    )
 
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, -0.01),
        ncol=min(len(legend_handles), 5),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.5,
        columnspacing=1.5,
    )
 
    outpath = output_dir / fname
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite figure: {outpath}")
 
 
# ── Alternative: per-model subpanels for spectra ────────────────────────────
# Use this instead of the envelope approach if you want each model's
# co-masked truth shown explicitly.
 
def _plot_composite_figure_subpanels(
    results,
    output_dir,
    fname="spectral_composite_subpanels.pdf",
    dpi=200,
):
    """
    Alternative layout with per-model subpanels for the spectra row.
 
    Layout:
      Row 0 (height 1):  Spectral ratio — 1 panel per horizon, all models
      Rows 1..N (height 1 each): Spectra — 1 row per model, 1 col per horizon
          Each small panel shows that model's co-masked truth (black)
          and prediction (colored).
 
    A single unified legend is placed below all panels.
    """
    from matplotlib.lines import Line2D
 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    hl = results["horizon_labels"]
    n_hz = len(hl)
    scale_bands = results["scale_bands"]
    model_names = list(results["models"].keys())
 
    # Filter to models that have data for at least one horizon
    active_models = [
        mn for mn in model_names
        if any(results["models"][mn].get(h) is not None for h in range(n_hz))
    ]
    n_models = len(active_models)
 
    # Grid: 1 ratio row + n_models spectra rows
    nrows = 1 + n_models
    fig_w = 4.2 * n_hz
    fig_h = 2.8 * nrows
 
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Height ratios: ratio row slightly taller
    gs = gridspec.GridSpec(
        nrows, n_hz, figure=fig,
        height_ratios=[1.2] + [1.0] * n_models,
        hspace=0.35, wspace=0.25,
        left=0.10, right=0.97, bottom=0.10, top=0.94,
    )
 
    # ── Row 0: Spectral ratio ──────────────────────────────────────────
    for col in range(n_hz):
        ax = fig.add_subplot(gs[0, col])
        h = col
        for mname in active_models:
            mdata = results["models"][mname]
            if mdata.get(h) is None:
                continue
            d = mdata[h]
            ax.semilogx(
                d["wavelength"], d["metrics"]["ratio"],
                color=_color_for(mname), ls=_ls_for(mname),
                lw=1.5,
            )
        ax.axhline(1.0, color="gray", alpha=0.4, lw=0.8)
        ax.axhline(0.5, color="black", ls=":", alpha=0.5, lw=0.8)
 
        for bname, (wl_lo, wl_hi) in scale_bands.items():
            ax.axvspan(wl_lo, wl_hi, alpha=0.15,
                       color=BAND_COLORS.get(bname, "#f0f0f0"), zorder=0)
 
        ax.set_title(hl[col], fontsize=10, fontweight="bold")
        ax.invert_xaxis()
        ax.set_ylim(0, 1.8)
        ax.grid(True, alpha=0.15, which="both")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel("PSD ratio", fontsize=9)
 
    # Row label
    fig.text(0.02, 0.92, "Ratio", rotation=90, va="center",
             fontsize=9, fontweight="bold")
 
    # ── Rows 1..N: per-model spectra ───────────────────────────────────
    for mi, mname in enumerate(active_models):
        row = 1 + mi
        mdata = results["models"][mname]
 
        for col in range(n_hz):
            ax = fig.add_subplot(gs[row, col])
            h = col
            if mdata.get(h) is None:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, alpha=0.4)
                ax.set_visible(True)
                continue
 
            d = mdata[h]
            # Truth (co-masked for this specific model)
            ax.loglog(d["wavelength"], d["psd_truth"],
                      "k-", lw=2.0)
            # Prediction
            ax.loglog(d["wavelength"], d["psd_pred_mean"],
                      color=_color_for(mname), ls=_ls_for(mname),
                      lw=1.5)
            if d["psd_pred_std"] is not None:
                ax.fill_between(
                    d["wavelength"],
                    np.maximum(d["psd_pred_mean"] - d["psd_pred_std"], 1e-15),
                    d["psd_pred_mean"] + d["psd_pred_std"],
                    alpha=0.10, color=_color_for(mname),
                )
 
            ax.invert_xaxis()
            ax.grid(True, alpha=0.12, which="both")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.tick_params(labelsize=7)
 
            if col == 0:
                ax.set_ylabel("PSD", fontsize=8)
            if row == nrows - 1:
                ax.set_xlabel("Wavelength (km)", fontsize=8)
 
        # Model name as row label
        fig.text(
            0.02, 1.0 - (row + 0.5) / nrows,
            mname, rotation=90, va="center",
            fontsize=8, fontweight="bold", color=_color_for(mname),
        )
 
    # ── Unified legend at the bottom ────────────────────────────────────
    legend_handles = []
    # Models
    for mname in active_models:
        legend_handles.append(
            Line2D([0], [0], color=_color_for(mname), ls=_ls_for(mname),
                   lw=2.0, label=mname)
        )
    # Observed (spectra rows)
    legend_handles.append(
        Line2D([0], [0], color="black", ls="-", lw=2.0,
               label="Observed (co-masked)")
    )
    # Reference lines (ratio row)
    legend_handles.append(
        Line2D([0], [0], color="black", ls=":", lw=0.8, alpha=0.6,
               label="0.5 threshold")
    )
 
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, -0.01),
        ncol=min(len(legend_handles), 5),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        handlelength=2.5,
        columnspacing=1.5,
    )
 
    outpath = output_dir / fname
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite (subpanels): {outpath}")
 


# =============================================================================
# Save results and metrics
# =============================================================================

def save_results(results, path="spectral_results.pkl"):
    """Save spectral analysis results to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
 
    # Convert any numpy arrays to ensure pickle compatibility
    with open(path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {path} ({path.stat().st_size / 1024:.0f} KB)")
 
 
def load_results(path="spectral_results.pkl"):
    """Load spectral analysis results from disk."""
    path = Path(path)
    with open(path, "rb") as f:
        results = pickle.load(f)
    n_models = len(results["models"])
    n_horizons = len(results["horizon_labels"])
    print(f"Loaded results: {n_models} models, {n_horizons} horizons")
    return results



def _save_metrics_csv(results, output_dir):
    import csv
    hl = results["horizon_labels"]
    scale_bands = results["scale_bands"]
    bnames = list(scale_bands.keys())
    path = output_dir / "spectral_metrics.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["model", "horizon", "eff_res_km", "spectral_bias",
                  "n_valid_pairs", "mean_wet_fraction"]
        header += [f"band_ratio_{b}" for b in bnames]
        writer.writerow(header)

        for mn, md in results["models"].items():
            for h in range(len(hl)):
                if md.get(h) is None:
                    continue
                d = md[h]
                m = d["metrics"]
                row = [mn, hl[h],
                       f"{m['eff_res_km']:.2f}",
                       f"{m['spectral_bias']:.4f}",
                       d["n_valid"],
                       f"{d['mean_wet_fraction']:.4f}"]
                row += [f"{m['band_ratios'].get(b, np.nan):.4f}" for b in bnames]
                writer.writerow(row)

    print(f"  Saved: {path.name}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    preds, truth = load_all_data(
        DATA_DIR, MODEL_FILES,
        pred_var=PRED_VAR, truth_var=TRUTH_VAR,
    )

    results = run_spectral_analysis(
        preds=preds,
        truth=truth,
        dx=DX, dy=DY,
        horizon_labels=HORIZON_LABELS,
        horizons_to_compute=HORIZONS_TO_COMPUTE,
        model_horizon_map=MODEL_HORIZON_MAP,
        scale_bands=SCALE_BANDS,
        eff_res_threshold=EFF_RES_THRESHOLD,
        log_offset=LOG_OFFSET,
        wet_threshold=WET_THRESHOLD,
        mask_strategy=MASK_STRATEGY,
        dilation_pixels=MASK_DILATION_PIXELS,
        min_wet_fraction=MIN_WET_FRACTION,
        output_dir=OUTPUT_DIR,
        n_bins=N_RADIAL_BINS,
        window=WINDOW_TYPE,
        alpha=TUKEY_ALPHA,
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for mn, md in results["models"].items():
        print(f"\n{mn}:")
        hdr = f"  {'Hz':<6s} {'Eff.Res':<9s} {'WetFrac':<9s}"
        for b in SCALE_BANDS:
            hdr += f" {b:<9s}"
        hdr += f" {'Bias':<9s}"
        print(hdr)
        print("  " + "-" * (len(hdr)-2))
        for h, hlabel in enumerate(results["horizon_labels"]):
            if md.get(h) is None:
                continue
            d = md[h]
            m = d["metrics"]
            row = (f"  {hlabel:<6s} "
                   f"{m['eff_res_km']:<9.1f} "
                   f"{d['mean_wet_fraction']:<9.1%}")
            for b in SCALE_BANDS:
                row += f" {m['band_ratios'].get(b, np.nan):<9.3f}"
            row += f" {m['spectral_bias']:<9.3f}"
            print(row)

    return results


if __name__ == "__main__":
    results = main()
