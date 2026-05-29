"""
Final super-resolution evaluation metrics.

Contains helpers that are used within the Step2 evaluation notebook.
"""

import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from typing import Optional, Literal
import matplotlib.pyplot as plt

# =============================================================================
# basic Helpers
# =============================================================================
def _ensure_datetime64_time(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.coords:
        raise ValueError("Expected a 'time' coordinate.")
    if not np.issubdtype(da["time"].dtype, np.datetime64):
        da = da.assign_coords(time=da["time"].values.astype("datetime64[ns]"))
    return da


def crps_ensemble_np(ens: np.ndarray, obs: np.ndarray, axis: int = 0) -> np.ndarray:
    """CRPS = E|X-y| - 0.5 E|X-X'| for an ensemble forecast."""
    ens = np.asarray(ens)
    obs = np.asarray(obs)
    term1 = np.mean(np.abs(ens - obs), axis=axis)
    ens0 = np.moveaxis(ens, axis, 0)
    term2 = 0.5 * np.mean(np.abs(ens0[:, None, ...] - ens0[None, :, ...]), axis=(0, 1))
    return term1 - term2


def mask_bbox_from_latlon_window(window_latlon: xr.DataArray):
    """Get bounding box slices from lat/lon window with NaNs outside."""
    if window_latlon.ndim != 2:
        raise ValueError("window_latlon must be 2D.")
    lat_dim, lon_dim = window_latlon.dims
    m = window_latlon.notnull()
    m_np = m.data
    if hasattr(m_np, "compute"):
        m_np = m_np.compute()
    m_np = np.asarray(m_np)
    if not np.any(m_np):
        raise ValueError("Lat/Lon window mask has no non-NaN values.")
    i_lat, i_lon = np.where(m_np)
    lat_min, lat_max = int(i_lat.min()), int(i_lat.max())
    lon_min, lon_max = int(i_lon.min()), int(i_lon.max())
    return slice(lat_min, lat_max + 1), slice(lon_min, lon_max + 1), m


def _infer_spatial_dims(pred: xr.DataArray):
    core = {"ensemble", "time", "lead"}
    spatial = [d for d in pred.dims if d not in core]
    if len(spatial) != 2:
        raise ValueError(f"Could not infer 2 spatial dims from pred. Got {spatial}")
    return tuple(spatial)


# =============================================================================
# Ensemble-Aware Metric Functions
# =============================================================================

def _to_torch(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(np.asarray(arr)).float()


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return t.cpu().numpy()


# -----------------------------------------------------------------------------
# Binary Event Statistics (ensemble-aware)
# -----------------------------------------------------------------------------

def compute_ensemble_event_stats(
    ens: np.ndarray,
    obs: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """
    Compute event-based statistics for ensemble forecasts.
    
    Uses ensemble mean for deterministic categorical metrics,
    and also computes probabilistic metrics.
    
    Args:
        ens: Ensemble predictions (M, H, W) or (M, ..., H, W)
        obs: Observations (H, W) or (..., H, W)
        threshold: Precipitation threshold
    
    Returns:
        Dictionary with CSI, POD, FAR, Bias, and Brier Score
    """
    ens = np.asarray(ens)
    obs = np.asarray(obs)
    
    # Ensemble mean for deterministic metrics
    ens_mean = np.mean(ens, axis=0)
    
    # Binary fields
    obs_binary = (obs >= threshold)
    pred_binary = (ens_mean >= threshold)
    
    # Contingency table
    TP = np.sum(pred_binary & obs_binary)
    FP = np.sum(pred_binary & ~obs_binary)
    FN = np.sum(~pred_binary & obs_binary)
    TN = np.sum(~pred_binary & ~obs_binary)
    
    eps = 1e-8
    
    # Deterministic categorical metrics
    csi = TP / (TP + FP + FN + eps)
    pod = TP / (TP + FN + eps)
    far = FP / (TP + FP + eps)
    bias = (TP + FP) / (TP + FN + eps)
    
    # Probabilistic: Brier Score
    prob = np.mean(ens >= threshold, axis=0)  # Probability of exceedance
    brier_score = np.mean((prob - obs_binary.astype(float)) ** 2)
    
    return {
        "CSI": float(csi),
        "POD": float(pod),
        "FAR": float(far),
        "Bias": float(bias),
        "Brier_Score": float(brier_score),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
    }


# -----------------------------------------------------------------------------
# Conditional Error Metrics
# -----------------------------------------------------------------------------

def compute_conditional_errors(
    ens: np.ndarray,
    obs: np.ndarray,
    wet_threshold: float = 0.1,
    heavy_threshold: float = 1.0,
) -> dict:
    """
    Compute error metrics conditioned on precipitation intensity.
    
    Args:
        ens: Ensemble predictions (M, H, W)
        obs: Observations (H, W)
        wet_threshold: Threshold for "wet" pixels
        heavy_threshold: Threshold for "heavy" precipitation
    
    Returns:
        Dictionary with MSE_wet, MAE_wet, MSE_heavy, MAE_heavy
    """
    ens_mean = np.mean(ens, axis=0)
    
    diff = ens_mean - obs
    se = diff ** 2
    ae = np.abs(diff)
    
    wet_mask = obs >= wet_threshold
    heavy_mask = obs >= heavy_threshold
    
    results = {}
    
    # Wet pixels
    if np.any(wet_mask):
        results["MSE_wet"] = float(np.mean(se[wet_mask]))
        results["MAE_wet"] = float(np.mean(ae[wet_mask]))
    else:
        results["MSE_wet"] = np.nan
        results["MAE_wet"] = np.nan
    
    # Heavy pixels
    if np.any(heavy_mask):
        results["MSE_heavy"] = float(np.mean(se[heavy_mask]))
        results["MAE_heavy"] = float(np.mean(ae[heavy_mask]))
    else:
        results["MSE_heavy"] = np.nan
        results["MAE_heavy"] = np.nan
    
    results["wet_fraction"] = float(np.mean(wet_mask))
    results["heavy_fraction"] = float(np.mean(heavy_mask))
    
    return results


# -----------------------------------------------------------------------------
# Member-Averaged FSS (avFSS)
# -----------------------------------------------------------------------------

def compute_binary_probability_np(
    field: np.ndarray,
    threshold: float,
    neighborhood: int,
) -> np.ndarray:
    """
    Compute Binary Probability field using convolution.
    
    Args:
        field: Input field (H, W)
        threshold: Precipitation threshold
        neighborhood: Neighborhood size (odd integer)
    
    Returns:
        BP field (H, W)
    """
    binary = (field >= threshold).astype(np.float32)
    
    # Use torch for efficient convolution
    t = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    pad = neighborhood // 2
    t_padded = F.pad(t, (pad, pad, pad, pad), mode='reflect')
    bp = F.avg_pool2d(t_padded, kernel_size=neighborhood, stride=1)
    
    return bp.squeeze().numpy()


def compute_fss_deterministic_np(
    obs: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    neighborhood: int,
) -> float:
    """
    Compute deterministic FSS.
    
    Args:
        obs: Observations (H, W)
        pred: Predictions (H, W)
        threshold: Precipitation threshold
        neighborhood: Neighborhood size
    
    Returns:
        FSS value
    """
    bp_obs = compute_binary_probability_np(obs, threshold, neighborhood)
    bp_pred = compute_binary_probability_np(pred, threshold, neighborhood)
    
    fbs = np.mean((bp_pred - bp_obs) ** 2)
    fbs_ref = np.mean(bp_obs ** 2) + np.mean(bp_pred ** 2)
    
    if fbs_ref < 1e-10:
        return np.nan
    
    return float(1.0 - fbs / fbs_ref)


def compute_avfss_np(
    ens: np.ndarray,
    obs: np.ndarray,
    threshold: float,
    neighborhood: int,
) -> dict:
    """
    Compute Member-Averaged FSS (avFSS) following Necker et al. (2024).
    
    Args:
        ens: Ensemble predictions (M, H, W)
        obs: Observations (H, W)
        threshold: Precipitation threshold
        neighborhood: Neighborhood size
    
    Returns:
        Dictionary with avFSS and member statistics
    """
    M = ens.shape[0]
    
    # Compute BP for observation once
    bp_obs = compute_binary_probability_np(obs, threshold, neighborhood)
    
    fss_members = []
    for m in range(M):
        bp_member = compute_binary_probability_np(ens[m], threshold, neighborhood)
        
        fbs = np.mean((bp_member - bp_obs) ** 2)
        fbs_ref = np.mean(bp_obs ** 2) + np.mean(bp_member ** 2)
        
        if fbs_ref >= 1e-10:
            fss_m = 1.0 - fbs / fbs_ref
            fss_members.append(fss_m)
    
    if len(fss_members) == 0:
        return {"avFSS": np.nan, "avFSS_std": np.nan}
    
    fss_members = np.array(fss_members)
    
    return {
        "avFSS": float(np.mean(fss_members)),
        "avFSS_std": float(np.std(fss_members)),
        "avFSS_min": float(np.min(fss_members)),
        "avFSS_max": float(np.max(fss_members)),
    }


def compute_avfss_multi_scale_np(
    ens: np.ndarray,
    obs: np.ndarray,
    threshold: float = 0.1,
    neighborhoods: list[int] | None = None,
) -> dict:
    """
    Compute avFSS at multiple spatial scales.
    
    Args:
        ens: Ensemble predictions (M, H, W)
        obs: Observations (H, W)
        threshold: Precipitation threshold
        neighborhoods: List of neighborhood sizes
    
    Returns:
        Dictionary with avFSS at each scale
    """
    if neighborhoods is None:
        neighborhoods = [1, 5, 17]
    
    results = {}
    for n in neighborhoods:
        avfss_result = compute_avfss_np(ens, obs, threshold, n)
        results[f"avFSS_{n}"] = avfss_result["avFSS"]
        results[f"avFSS_{n}_std"] = avfss_result["avFSS_std"]
    
    return results


# -----------------------------------------------------------------------------
# Ensemble SAL (eSAL)
# -----------------------------------------------------------------------------

def compute_sal_single_np(
    obs: np.ndarray,
    pred: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """
    Compute SAL for a single deterministic forecast.
    
    Args:
        obs: Observations (H, W)
        pred: Predictions (H, W)
        threshold: Precipitation threshold
    
    Returns:
        Dictionary with S, A, L components
    """
    H, W = obs.shape
    
    # Mask below threshold
    obs_masked = np.where(obs >= threshold, obs, 0.0)
    pred_masked = np.where(pred >= threshold, pred, 0.0)
    
    R_obs = obs_masked.sum()
    R_pred = pred_masked.sum()
    
    if R_obs + R_pred < 1e-10:
        return {"S": np.nan, "A": np.nan, "L": np.nan}
    
    # A (Amplitude)
    A = (R_pred - R_obs) / (0.5 * (R_pred + R_obs))
    
    # Coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Centers of mass
    if R_obs > 1e-10:
        y_cm_obs = (y_coords * obs_masked).sum() / R_obs
        x_cm_obs = (x_coords * obs_masked).sum() / R_obs
    else:
        y_cm_obs, x_cm_obs = H / 2, W / 2
    
    if R_pred > 1e-10:
        y_cm_pred = (y_coords * pred_masked).sum() / R_pred
        x_cm_pred = (x_coords * pred_masked).sum() / R_pred
    else:
        y_cm_pred, x_cm_pred = H / 2, W / 2
    
    # L (Location)
    max_distance = np.sqrt(H**2 + W**2)
    cm_distance = np.sqrt((y_cm_obs - y_cm_pred)**2 + (x_cm_obs - x_cm_pred)**2)
    L1 = cm_distance / max_distance
    
    # Spread
    if R_obs > 1e-10:
        r_obs = (np.sqrt((y_coords - y_cm_obs)**2 + (x_coords - x_cm_obs)**2) * obs_masked).sum() / R_obs
    else:
        r_obs = 0.0
    
    if R_pred > 1e-10:
        r_pred = (np.sqrt((y_coords - y_cm_pred)**2 + (x_coords - x_cm_pred)**2) * pred_masked).sum() / R_pred
    else:
        r_pred = 0.0
    
    L2 = np.abs(r_pred - r_obs) / max_distance
    L = L1 + 2 * L2
    
    # S (Structure)
    obs_max = obs_masked.max()
    pred_max = pred_masked.max()
    
    V_obs = R_obs / max(obs_max, 1e-10) if obs_max > 0 else 0
    V_pred = R_pred / max(pred_max, 1e-10) if pred_max > 0 else 0
    
    if V_obs + V_pred < 1e-10:
        S = 0.0
    else:
        S = (V_pred - V_obs) / (0.5 * (V_pred + V_obs))
    
    return {"S": float(S), "A": float(A), "L": float(L)}


def compute_esal_np(
    ens: np.ndarray,
    obs: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """
    Compute ensemble SAL (eSAL).
    
    Computes SAL for each member and reports median and spread.
    
    Args:
        ens: Ensemble predictions (M, H, W)
        obs: Observations (H, W)
        threshold: Precipitation threshold
    
    Returns:
        Dictionary with eS, eA, eL and their spreads
    """
    M = ens.shape[0]
    
    S_members = []
    A_members = []
    L_members = []
    
    for m in range(M):
        sal_m = compute_sal_single_np(obs, ens[m], threshold)
        if not np.isnan(sal_m["S"]):
            S_members.append(sal_m["S"])
            A_members.append(sal_m["A"])
            L_members.append(sal_m["L"])
    
    if len(S_members) == 0:
        return {
            "eS": np.nan, "eA": np.nan, "eL": np.nan,
            "eS_spread": np.nan, "eA_spread": np.nan, "eL_spread": np.nan,
        }
    
    S_members = np.array(S_members)
    A_members = np.array(A_members)
    L_members = np.array(L_members)
    
    return {
        "eS": float(np.median(S_members)),
        "eA": float(np.median(A_members)),
        "eL": float(np.median(L_members)),
        "eS_spread": float(np.percentile(S_members, 75) - np.percentile(S_members, 25)),
        "eA_spread": float(np.percentile(A_members, 75) - np.percentile(A_members, 25)),
        "eL_spread": float(np.percentile(L_members, 75) - np.percentile(L_members, 25)),
        "eS_mean": float(np.mean(S_members)),
        "eA_mean": float(np.mean(A_members)),
        "eL_mean": float(np.mean(L_members)),
    }


# -----------------------------------------------------------------------------
# Rank Histogram
# -----------------------------------------------------------------------------

def compute_rank_histogram_np(
    ens: np.ndarray,
    obs: np.ndarray,
) -> dict:
    """
    Compute rank histogram for ensemble reliability assessment.
    
    Args:
        ens: Ensemble predictions (M, H, W)
        obs: Observations (H, W)
    
    Returns:
        Dictionary with histogram, reliability index, and delta
    """
    M = ens.shape[0]
    n_bins = M + 1
    
    # Flatten spatial dimensions
    ens_flat = ens.reshape(M, -1).T  # (N_pixels, M)
    obs_flat = obs.flatten()  # (N_pixels,)
    
    # Remove NaN pixels
    valid = ~np.isnan(obs_flat) & ~np.any(np.isnan(ens_flat), axis=1)
    ens_flat = ens_flat[valid]
    obs_flat = obs_flat[valid]
    
    if len(obs_flat) == 0:
        return {"histogram": np.full(n_bins, np.nan), "reliability_index": np.nan}
    
    # Compute ranks: number of ensemble members < observation
    ranks = np.sum(ens_flat < obs_flat[:, None], axis=1)
    
    # Build histogram
    hist, _ = np.histogram(ranks, bins=np.arange(n_bins + 1) - 0.5)
    hist = hist / hist.sum()  # Normalize
    
    # Reliability index (deviation from uniform)
    uniform = 1.0 / n_bins
    delta = np.mean(np.abs(hist - uniform))
    
    # Diagnose shape
    # U-shape: underdispersive, dome-shape: overdispersive
    edge_mean = (hist[0] + hist[-1]) / 2
    center_mean = np.mean(hist[n_bins//4 : 3*n_bins//4]) if n_bins > 4 else np.mean(hist[1:-1])
    
    if edge_mean > center_mean * 1.2:
        shape = "U-shaped (underdispersive)"
    elif center_mean > edge_mean * 1.2:
        shape = "dome-shaped (overdispersive)"
    else:
        shape = "approximately flat (well-calibrated)"
    
    return {
        "histogram": hist,
        "reliability_index": float(delta),
        "shape_diagnosis": shape,
    }


# -----------------------------------------------------------------------------
# Probability Integral Transform (PIT)
# Following Leinonen et al. (2023) and Gneiting et al. (2007)
# -----------------------------------------------------------------------------


#  Helper: KL divergence from uniform
def _kl_from_uniform(hist_norm: np.ndarray) -> float:
    """KL divergence of a normalized histogram from the uniform distribution."""
    n_bins = len(hist_norm)
    uniform = 1.0 / n_bins
    hist_safe = np.clip(hist_norm, 1e-10, None)
    return float(np.sum(hist_safe * np.log(hist_safe / uniform)))


def compute_pit_values_np(
    ens: np.ndarray,
    obs: np.ndarray,
    randomize: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Compute per-pixel Probability Integral Transform (PIT) values.

    For an ensemble forecast {x_1, ..., x_M} and observation y, the empirical
    CDF gives F(y) = #{x_i <= y} / M.  For precipitation, many members may
    share the same value (especially zero), so we use the *randomized* PIT:

        PIT = U[ F(y-), F(y) ]

    where F(y-) = #{x_i < y} / M.  If the ensemble correctly represents the
    forecast distribution, PIT values are uniformly distributed on [0, 1].

    References:
        Gneiting, Balabdaoui & Raftery (2007), JRSS-B.
        Leinonen et al. (2023), arXiv:2304.12891 (rank distributions).

    Args:
        ens: Ensemble predictions (M, H, W) or (M, N_pixels)
        obs: Observations (H, W) or (N_pixels,)
        randomize: If True, use randomized PIT (recommended for precip).
                   If False, use midpoint PIT = (F(y-) + F(y)) / 2.
        rng: numpy random Generator for reproducibility.

    Returns:
        pit_values: array of same shape as obs, values in [0, 1].
                    NaN where obs is NaN.
    """
    ens = np.asarray(ens, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    M = ens.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    # Flatten spatial dims for vectorized computation
    orig_shape = obs.shape
    obs_flat = obs.ravel()
    ens_flat = ens.reshape(M, -1)  # (M, N)
    N = obs_flat.shape[0]

    # F(y) = fraction of members <= obs
    F_upper = np.sum(ens_flat <= obs_flat[None, :], axis=0) / M  # (N,)
    # F(y-) = fraction of members strictly < obs
    F_lower = np.sum(ens_flat < obs_flat[None, :], axis=0) / M   # (N,)

    if randomize:
        # PIT ~ U[F(y-), F(y)]
        u = rng.uniform(size=N)
        pit = F_lower + u * (F_upper - F_lower)
    else:
        # Midpoint PIT
        pit = 0.5 * (F_lower + F_upper)

    # Preserve NaN
    nan_mask = np.isnan(obs_flat)
    pit[nan_mask] = np.nan

    return pit.reshape(orig_shape)


def compute_pit_histogram_np(
    ens: np.ndarray,
    obs: np.ndarray,
    n_bins: int = 20,
    randomize: bool = True,
    rng: np.random.Generator | None = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(42)

    pit_vals = compute_pit_values_np(ens, obs, randomize=randomize, rng=rng)

    pit_flat = pit_vals.ravel()
    pit_flat = pit_flat[np.isfinite(pit_flat)]

    if len(pit_flat) == 0:
        return {
            "pit_values": np.array([]),
            "histogram": np.full(n_bins, np.nan),
            "bin_edges": np.linspace(0, 1, n_bins + 1),
            "reliability_index": np.nan,
            "kl_divergence": np.nan,
            "ks_statistic": np.nan,
            "shape_diagnosis": "no valid data",
        }

    bin_edges = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(pit_flat, bins=bin_edges)
    hist_norm = hist / hist.sum()

    uniform = 1.0 / n_bins
    reliability_index = float(np.mean(np.abs(hist_norm - uniform)))
    kl_divergence = _kl_from_uniform(hist_norm)

    pit_sorted = np.sort(pit_flat)
    n = len(pit_sorted)
    ecdf = np.arange(1, n + 1) / n
    ks_stat = float(np.max(np.abs(ecdf - pit_sorted)))

    # Shape diagnosis
    edge_mean = (hist_norm[:2].mean() + hist_norm[-2:].mean()) / 2
    center_slice = slice(n_bins // 4, 3 * n_bins // 4)
    center_mean = hist_norm[center_slice].mean()
    left_mean = hist_norm[: n_bins // 3].mean()
    right_mean = hist_norm[2 * n_bins // 3 :].mean()

    parts = []
    if edge_mean > center_mean * 1.3:
        parts.append("U-shaped (underdispersive)")
    elif center_mean > edge_mean * 1.3:
        parts.append("dome-shaped (overdispersive)")
    if left_mean > right_mean * 1.3:
        parts.append("left-skewed (ensemble overestimates)")
    elif right_mean > left_mean * 1.3:
        parts.append("right-skewed (ensemble underestimates)")
    if not parts:
        shape_diagnosis = "approximately flat (well-calibrated)"
    else:
        shape_diagnosis = ", ".join(parts)

    return {
        "pit_values": pit_flat,
        "histogram": hist_norm,
        "bin_edges": bin_edges,
        "reliability_index": reliability_index,
        "kl_divergence": kl_divergence,
        "ks_statistic": ks_stat,
        "shape_diagnosis": shape_diagnosis,
    }


def compute_pit_by_intensity_np(
    ens: np.ndarray,
    obs: np.ndarray,
    thresholds: list[float] | None = None,
    n_bins: int = 20,
    randomize: bool = True,
    rng: np.random.Generator | None = None,
) -> dict:
    if thresholds is None:
        thresholds = [0.0, 0.1, 1.0, np.inf]
    if rng is None:
        rng = np.random.default_rng(42)

    pit_vals = compute_pit_values_np(ens, obs, randomize=randomize, rng=rng)

    obs_flat = obs.ravel()
    pit_flat = pit_vals.ravel()
    valid = np.isfinite(obs_flat) & np.isfinite(pit_flat)
    obs_v = obs_flat[valid]
    pit_v = pit_flat[valid]

    bin_edges = np.linspace(0, 1, n_bins + 1)

    results = {}
    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        if hi == np.inf:
            label = f">={lo:.1f}mm"
        elif lo == 0:
            label = f"<{hi:.1f}mm"
        else:
            label = f"{lo:.1f}-{hi:.1f}mm"

        mask = (obs_v >= lo) & (obs_v < hi)
        n_pix = int(mask.sum())

        if n_pix < 10:
            results[label] = {
                "histogram": np.full(n_bins, np.nan),
                "reliability_index": np.nan,
                "kl_divergence": np.nan,
                "n_pixels": n_pix,
            }
            continue

        hist, _ = np.histogram(pit_v[mask], bins=bin_edges)
        hist_norm = hist / hist.sum()
        uniform = 1.0 / n_bins
        rel_idx = float(np.mean(np.abs(hist_norm - uniform)))
        kl_div = _kl_from_uniform(hist_norm)

        results[label] = {
            "histogram": hist_norm,
            "reliability_index": rel_idx,
            "kl_divergence": kl_div,
            "n_pixels": n_pix,
        }

    return results


# -----------------------------------------------------------------------------
# PIT Plotting
# -----------------------------------------------------------------------------

def plot_pit_histogram(
    ens: np.ndarray,
    obs: np.ndarray,
    n_bins: int = 20,
    title: str | None = None,
    figsize: tuple = (7, 5),
    ax=None,
    color: str = "steelblue",
    label: str | None = None,
):
    pit_result = compute_pit_histogram_np(ens, obs, n_bins=n_bins)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("#F5F5F5")
        created_fig = True

    hist = pit_result["histogram"]
    edges = pit_result["bin_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax.bar(centers, hist, width=width * 0.9, color=color, edgecolor="white",
           linewidth=0.5, alpha=0.85, label=label)
    ax.axhline(y=1.0 / n_bins, color="red", linestyle="--", linewidth=2,
               label="Uniform (ideal)" if label is None else None)

    ax.set_xlabel("PIT value", fontsize=12)
    ax.set_ylabel("Relative frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, color="white", linewidth=1)

    diag = pit_result["shape_diagnosis"]
    ri = pit_result["reliability_index"]
    kl = pit_result["kl_divergence"]
    ks = pit_result["ks_statistic"]
    annot = f"RI = {ri:.4f}  |  KL = {kl:.4f}  |  KS = {ks:.4f}\n{diag}"
    ax.text(0.98, 0.95, annot, transform=ax.transAxes, fontsize=9,
            va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", alpha=0.8))

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    if label is not None:
        ax.legend(fontsize=9)

    if created_fig:
        plt.tight_layout()
        return fig, ax
    return ax


def plot_pit_by_intensity(
    ens: np.ndarray,
    obs: np.ndarray,
    thresholds: list[float] | None = None,
    n_bins: int = 20,
    figsize: tuple = (16, 4),
    title: str | None = None,
):
    """
    Plot PIT histograms conditioned on precipitation intensity.

    Args:
        ens: Ensemble predictions (M, H, W)
        obs: Observations (H, W)
        thresholds: Intensity regime boundaries
        n_bins: Number of bins
        figsize: Figure size
        title: Super-title

    Returns:
        fig, axes
    """
    import matplotlib.pyplot as plt

    results = compute_pit_by_intensity_np(ens, obs, thresholds=thresholds, n_bins=n_bins)
    n_regimes = len(results)

    fig, axes = plt.subplots(1, n_regimes, figsize=figsize, facecolor="white")
    if n_regimes == 1:
        axes = [axes]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]
    colors = ["#90CAF9", "#42A5F5", "#1565C0", "#0D47A1"]

    for idx, (regime_label, regime_data) in enumerate(results.items()):
        ax = axes[idx]
        ax.set_facecolor("#F5F5F5")

        hist = regime_data["histogram"]
        if np.all(np.isnan(hist)):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
        else:
            c = colors[idx % len(colors)]
            ax.bar(centers, hist, width=width * 0.9, color=c,
                   edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.axhline(y=1.0 / n_bins, color="red", linestyle="--", linewidth=2)

            ri = regime_data["reliability_index"]
            n_pix = regime_data["n_pixels"]
            ax.text(0.98, 0.95, f"RI={ri:.4f}\nn={n_pix:,}",
                    transform=ax.transAxes, fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_title(regime_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("PIT value", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, color="white", linewidth=1)
        if idx == 0:
            ax.set_ylabel("Relative frequency", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig, axes


def plot_pit_comparison(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bins: int = 20,
    figsize: tuple = (8, 5),
    title: str = "PIT Histogram Comparison",
):
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2", "#F9A825"]
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("#F5F5F5")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]
    n_models = len(models)
    bar_width = width * 0.85 / n_models

    for idx, (name, (ens, obs)) in enumerate(models.items()):
        pit_result = compute_pit_histogram_np(ens, obs, n_bins=n_bins)
        hist = pit_result["histogram"]
        kl = pit_result["kl_divergence"]
        c = colors[idx % len(colors)]
        offset = (idx - n_models / 2 + 0.5) * bar_width
        ax.bar(centers + offset, hist, width=bar_width, color=c,
               edgecolor="white", linewidth=0.3, alpha=0.8,
               label=f"{name} (KL={kl:.4f})")

    ax.axhline(y=1.0 / n_bins, color="red", linestyle="--", linewidth=2,
               label="Uniform (ideal)")
    ax.set_xlabel("PIT value", fontsize=12)
    ax.set_ylabel("Relative frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, color="white", linewidth=1)

    plt.tight_layout()
    return fig, ax



# -----------------------------------------------------------------------------
# Spread-Skill Relationship
# -----------------------------------------------------------------------------

def compute_spread_skill_np(
    ens: np.ndarray,
    obs: np.ndarray,
) -> dict:
    """
    Compute spread-skill relationship metrics.
    
    Args:
        ens: Ensemble predictions (M, H, W)
        obs: Observations (H, W)
    
    Returns:
        Dictionary with spread, RMSE, and spread-skill ratio
    """
    ens_mean = np.mean(ens, axis=0)
    ens_std = np.std(ens, axis=0)
    
    # RMSE of ensemble mean
    rmse = np.sqrt(np.nanmean((ens_mean - obs) ** 2))
    
    # Mean spread
    spread = np.nanmean(ens_std)
    
    # Spread-skill ratio
    ssr = spread / (rmse + 1e-10)
    
    return {
        "spread": float(spread),
        "rmse_mean": float(rmse),
        "spread_skill_ratio": float(ssr),
    }


# -----------------------------------------------------------------------------
# Spectral Analysis (lightweight)
# -----------------------------------------------------------------------------

def compute_radial_spectrum_np(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radially-averaged power spectrum.
    
    Args:
        field: 2D field (H, W)
    
    Returns:
        wavenumbers, power spectrum
    """
    H, W = field.shape
    
    # Remove NaNs
    field = np.nan_to_num(field, nan=0.0)
    
    # 2D FFT
    fft2 = np.fft.fft2(field)
    power = np.abs(np.fft.fftshift(fft2)) ** 2
    
    # Radial averaging
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    max_r = min(cy, cx)
    radial_profile = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    for ri in range(max_r):
        mask = (r == ri)
        radial_profile[ri] = power[mask].sum()
        counts[ri] = mask.sum()
    
    radial_profile = radial_profile / np.maximum(counts, 1)
    wavenumbers = np.arange(max_r)
    
    return wavenumbers, radial_profile


def compute_spectral_metrics_np(
    ens: np.ndarray,
    obs: np.ndarray,
) -> dict:
    """
    Compute spectral analysis metrics (lightweight version).
    
    Compares spectral slopes to assess if ensemble members have
    realistic spatial structure.
    
    Args:
        ens: Ensemble predictions (M, H, W)
        obs: Observations (H, W)
    
    Returns:
        Dictionary with spectral slopes and correlation
    """
    # Observation spectrum
    k_obs, spec_obs = compute_radial_spectrum_np(obs)
    
    # Ensemble mean spectrum
    ens_mean = np.mean(ens, axis=0)
    k_pred, spec_pred = compute_radial_spectrum_np(ens_mean)
    
    # Average member spectrum (to check if members have realistic structure)
    member_specs = []
    for m in range(ens.shape[0]):
        _, spec_m = compute_radial_spectrum_np(ens[m])
        member_specs.append(spec_m)
    spec_members_avg = np.mean(member_specs, axis=0)
    
    # Log spectra for slope fitting (skip k=0)
    log_k = np.log10(k_obs[1:] + 1e-10)
    log_spec_obs = np.log10(spec_obs[1:] + 1e-10)
    log_spec_pred = np.log10(spec_pred[1:] + 1e-10)
    log_spec_members = np.log10(spec_members_avg[1:] + 1e-10)
    
    def fit_slope(log_k, log_spec):
        valid = np.isfinite(log_k) & np.isfinite(log_spec)
        if valid.sum() < 3:
            return np.nan
        coeffs = np.polyfit(log_k[valid], log_spec[valid], 1)
        return coeffs[0]
    
    slope_obs = fit_slope(log_k, log_spec_obs)
    slope_pred = fit_slope(log_k, log_spec_pred)
    slope_members = fit_slope(log_k, log_spec_members)
    
    # Spectral correlation
    valid = np.isfinite(log_spec_obs) & np.isfinite(log_spec_members)
    if valid.sum() > 2:
        corr = np.corrcoef(log_spec_obs[valid], log_spec_members[valid])[0, 1]
    else:
        corr = np.nan
    
    return {
        "spectral_slope_obs": float(slope_obs),
        "spectral_slope_ens_mean": float(slope_pred),
        "spectral_slope_members_avg": float(slope_members),
        "spectral_slope_error": float(slope_members - slope_obs),
        "spectral_correlation": float(corr),
    }


# =============================================================================
# Main Function: Compute All Metrics on Lat/Lon Window
# =============================================================================

def compute_metrics_on_latlon_window_extended(
    pred: xr.DataArray,
    truth: xr.DataArray,
    window_latlon: xr.DataArray,
    *,
    point: str = "mean",
    apply_exact_shape_mask: bool = True,
    # Metric options
    threshold: float = 0.1,
    wet_threshold: float = 0.1,
    heavy_threshold: float = 1.0,
    fss_neighborhoods: list[int] | None = None,
    compute_spectral: bool = True,
    compute_rank_hist: bool = True,
) -> xr.Dataset:
    """
    Compute comprehensive metrics for ensemble super-resolution forecasts.
    
    Extends the basic metrics with:
    - Conditional errors (MSE_wet, MAE_wet, MSE_heavy, MAE_heavy)
    - Event statistics (CSI, POD, FAR, Bias, Brier Score)
    - Member-averaged FSS (avFSS) at multiple scales
    - Ensemble SAL (eS, eA, eL with spreads)
    - Spread-skill relationship
    - Rank histogram reliability
    - Spectral analysis (optional)
    
    Args:
        pred: Predictions with dims ("ensemble", "time", "lead", Y, X)
        truth: Observations with dims ("time", Y, X)
        window_latlon: 2D DataArray with NaNs outside desired window
        point: "mean" or "median" for deterministic point forecast
        apply_exact_shape_mask: Whether to mask inside bbox
        threshold: Threshold for event-based metrics
        wet_threshold: Threshold for "wet" conditional metrics
        heavy_threshold: Threshold for "heavy" conditional metrics
        fss_neighborhoods: Neighborhood sizes for FSS (default: [1, 5, 17])
        compute_spectral: Whether to compute spectral metrics
        compute_rank_hist: Whether to compute rank histogram
    
    Returns:
        xr.Dataset with all metrics per (time, lead)
    """
    if fss_neighborhoods is None:
        fss_neighborhoods = [1, 5, 17]
    
    pred = _ensure_datetime64_time(pred)
    truth = _ensure_datetime64_time(truth)
    
    y_dim, x_dim = _infer_spatial_dims(pred)
    
    # Validate dimensions
    if set(truth.dims) != {"time", y_dim, x_dim}:
        raise ValueError(f"truth dims must be ('time','{y_dim}','{x_dim}'), got {truth.dims}")
    
    lat_dim, lon_dim = window_latlon.dims
    if (window_latlon.sizes[lat_dim] != pred.sizes[y_dim] or 
        window_latlon.sizes[lon_dim] != pred.sizes[x_dim]):
        raise ValueError("window_latlon shape must match pred spatial shape.")
    
    # Get bbox and mask
    y_slice, x_slice, mask_latlon_bool = mask_bbox_from_latlon_window(window_latlon)
    
    # Crop to bbox
    pred_c = pred.isel({y_dim: y_slice, x_dim: x_slice})
    truth_c = truth.isel({y_dim: y_slice, x_dim: x_slice})
    
    # Apply exact mask
    if apply_exact_shape_mask:
        mask_c = mask_latlon_bool.isel({lat_dim: y_slice, lon_dim: x_slice}).data
        mask_c_da = xr.DataArray(mask_c, dims=(y_dim, x_dim))
        pred_c = pred_c.where(mask_c_da)
        truth_c = truth_c.where(mask_c_da)
    
    # Align time
    truth_aligned = truth_c.sel(time=pred_c["time"])
    
    # Get dimensions
    times = pred_c["time"].values
    leads = pred_c["lead"].values
    n_times = len(times)
    n_leads = len(leads)
    
    # Initialize result arrays
    results = {
        # Basic metrics
        "crps": np.full((n_times, n_leads), np.nan),
        "mse": np.full((n_times, n_leads), np.nan),
        "mae": np.full((n_times, n_leads), np.nan),
        "rmse": np.full((n_times, n_leads), np.nan),
        "bias": np.full((n_times, n_leads), np.nan),
        # Conditional errors
        "mse_wet": np.full((n_times, n_leads), np.nan),
        "mae_wet": np.full((n_times, n_leads), np.nan),
        "mse_heavy": np.full((n_times, n_leads), np.nan),
        "mae_heavy": np.full((n_times, n_leads), np.nan),
        "wet_fraction": np.full((n_times, n_leads), np.nan),
        # Event statistics
        "csi": np.full((n_times, n_leads), np.nan),
        "pod": np.full((n_times, n_leads), np.nan),
        "far": np.full((n_times, n_leads), np.nan),
        "freq_bias": np.full((n_times, n_leads), np.nan),
        "brier_score": np.full((n_times, n_leads), np.nan),
        # Spread-skill
        "spread": np.full((n_times, n_leads), np.nan),
        "spread_skill_ratio": np.full((n_times, n_leads), np.nan),
        # eSAL
        "eS": np.full((n_times, n_leads), np.nan),
        "eA": np.full((n_times, n_leads), np.nan),
        "eL": np.full((n_times, n_leads), np.nan),
        "eS_spread": np.full((n_times, n_leads), np.nan),
        "eA_spread": np.full((n_times, n_leads), np.nan),
        "eL_spread": np.full((n_times, n_leads), np.nan),
    }
    
    # avFSS at multiple scales
    for n in fss_neighborhoods:
        results[f"avfss_{n}"] = np.full((n_times, n_leads), np.nan)
    
    # Optional metrics
    if compute_rank_hist:
        results["reliability_index"] = np.full((n_times, n_leads), np.nan)
    
    if compute_spectral:
        results["spectral_slope_error"] = np.full((n_times, n_leads), np.nan)
        results["spectral_correlation"] = np.full((n_times, n_leads), np.nan)
    
    # Loop over time and lead
    for t_idx, t in enumerate(times):
        for l_idx, lead in enumerate(leads):
            # Extract ensemble and observation
            ens = pred_c.sel(time=t, lead=lead).values  # (M, H, W)
            obs = truth_aligned.sel(time=t).values  # (H, W)
            
            # Skip if all NaN
            if np.all(np.isnan(obs)) or np.all(np.isnan(ens)):
                continue
            
            # Replace NaNs for computation
            ens_clean = np.nan_to_num(ens, nan=0.0)
            obs_clean = np.nan_to_num(obs, nan=0.0)
            
            # --- Basic metrics ---
            crps_val = crps_ensemble_np(ens_clean, obs_clean, axis=0)
            results["crps"][t_idx, l_idx] = np.nanmean(crps_val)
            
            if point == "mean":
                point_pred = np.mean(ens_clean, axis=0)
            else:
                point_pred = np.median(ens_clean, axis=0)
            
            err = point_pred - obs_clean
            results["mse"][t_idx, l_idx] = np.nanmean(err ** 2)
            results["mae"][t_idx, l_idx] = np.nanmean(np.abs(err))
            results["rmse"][t_idx, l_idx] = np.sqrt(results["mse"][t_idx, l_idx])
            results["bias"][t_idx, l_idx] = np.nanmean(err)
            
            # --- Conditional errors ---
            cond = compute_conditional_errors(ens_clean, obs_clean, wet_threshold, heavy_threshold)
            results["mse_wet"][t_idx, l_idx] = cond["MSE_wet"]
            results["mae_wet"][t_idx, l_idx] = cond["MAE_wet"]
            results["mse_heavy"][t_idx, l_idx] = cond["MSE_heavy"]
            results["mae_heavy"][t_idx, l_idx] = cond["MAE_heavy"]
            results["wet_fraction"][t_idx, l_idx] = cond["wet_fraction"]
            
            # --- Event statistics ---
            event = compute_ensemble_event_stats(ens_clean, obs_clean, threshold)
            results["csi"][t_idx, l_idx] = event["CSI"]
            results["pod"][t_idx, l_idx] = event["POD"]
            results["far"][t_idx, l_idx] = event["FAR"]
            results["freq_bias"][t_idx, l_idx] = event["Bias"]
            results["brier_score"][t_idx, l_idx] = event["Brier_Score"]
            
            # --- Spread-skill ---
            ss = compute_spread_skill_np(ens_clean, obs_clean)
            results["spread"][t_idx, l_idx] = ss["spread"]
            results["spread_skill_ratio"][t_idx, l_idx] = ss["spread_skill_ratio"]
            
            # --- avFSS ---
            for n in fss_neighborhoods:
                avfss = compute_avfss_np(ens_clean, obs_clean, threshold, n)
                results[f"avfss_{n}"][t_idx, l_idx] = avfss["avFSS"]
            
            # --- eSAL ---
            esal = compute_esal_np(ens_clean, obs_clean, threshold)
            results["eS"][t_idx, l_idx] = esal["eS"]
            results["eA"][t_idx, l_idx] = esal["eA"]
            results["eL"][t_idx, l_idx] = esal["eL"]
            results["eS_spread"][t_idx, l_idx] = esal["eS_spread"]
            results["eA_spread"][t_idx, l_idx] = esal["eA_spread"]
            results["eL_spread"][t_idx, l_idx] = esal["eL_spread"]
            
            # --- Rank histogram ---
            if compute_rank_hist:
                rh = compute_rank_histogram_np(ens_clean, obs_clean)
                results["reliability_index"][t_idx, l_idx] = rh["reliability_index"]
            
            # --- Spectral ---
            if compute_spectral:
                spec = compute_spectral_metrics_np(ens_clean, obs_clean)
                results["spectral_slope_error"][t_idx, l_idx] = spec["spectral_slope_error"]
                results["spectral_correlation"][t_idx, l_idx] = spec["spectral_correlation"]
    
    # Build xr.Dataset
    coords = {"time": times, "lead": leads}
    
    data_vars = {
        name: (["time", "lead"], arr) for name, arr in results.items()
    }
    
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add attributes
    ds.attrs.update({
        "point_forecast": point,
        "threshold": threshold,
        "wet_threshold": wet_threshold,
        "heavy_threshold": heavy_threshold,
        "fss_neighborhoods": str(fss_neighborhoods),
        "window_defined_in": f"{window_latlon.dims}",
        "applied_by": "index (lat->y, lon->x)",
    })
    
    return ds


def compute_metrics_per_lead(
    ds: xr.Dataset,
    metrics: list[str] | None = None,
) -> dict:
    """
    Aggregate metrics per lead time (averaging over time).
    
    Args:
        ds: Dataset from compute_metrics_on_latlon_window_extended
        metrics: List of metrics to include (default: all)
    
    Returns:
        Dictionary with {lead: {metric: value}}
    """
    if metrics is None:
        metrics = list(ds.data_vars.keys())
    
    per_lead = {}
    
    for lead in ds["lead"].values:
        lead_ds = ds.sel(lead=lead)
        per_lead[int(lead)] = {
            m: float(lead_ds[m].mean(dim="time", skipna=True).values)
            for m in metrics if m in ds.data_vars
        }
    
    return per_lead


def compute_global_metrics(
    ds: xr.Dataset,
    metrics: list[str] | None = None,
) -> dict:
    """
    Aggregate metrics globally (averaging over time and lead).
    
    Args:
        ds: Dataset from compute_metrics_on_latlon_window_extended
        metrics: List of metrics to include (default: all)
    
    Returns:
        Dictionary with {metric: value}
    """
    if metrics is None:
        metrics = list(ds.data_vars.keys())
    
    return {
        m: float(ds[m].mean(skipna=True).values)
        for m in metrics if m in ds.data_vars
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_rank_histogram(
    pred: xr.DataArray,
    truth: xr.DataArray,
    window_latlon: xr.DataArray,
    lead: int = 0,
    figsize: tuple = (8, 5),
    title: str | None = None,
):
    """
    Plot rank histogram for a specific lead time.
    """
    import matplotlib.pyplot as plt
    
    pred = _ensure_datetime64_time(pred)
    truth = _ensure_datetime64_time(truth)
    
    y_dim, x_dim = _infer_spatial_dims(pred)
    lat_dim, lon_dim = window_latlon.dims
    
    y_slice, x_slice, mask_latlon_bool = mask_bbox_from_latlon_window(window_latlon)
    
    pred_c = pred.isel({y_dim: y_slice, x_dim: x_slice}).sel(lead=lead)
    truth_c = truth.isel({y_dim: y_slice, x_dim: x_slice})
    truth_aligned = truth_c.sel(time=pred_c["time"])
    
    # Aggregate all time steps
    all_ens = []
    all_obs = []
    
    for t in pred_c["time"].values:
        ens = pred_c.sel(time=t).values
        obs = truth_aligned.sel(time=t).values
        all_ens.append(ens)
        all_obs.append(obs)
    
    all_ens = np.concatenate([e.reshape(e.shape[0], -1) for e in all_ens], axis=1)
    all_obs = np.concatenate([o.flatten() for o in all_obs])
    
    # Compute histogram
    M = all_ens.shape[0]
    n_bins = M + 1
    
    valid = ~np.isnan(all_obs)
    ens_valid = all_ens[:, valid].T
    obs_valid = all_obs[valid]
    
    ranks = np.sum(ens_valid < obs_valid[:, None], axis=1)
    hist, _ = np.histogram(ranks, bins=np.arange(n_bins + 1) - 0.5)
    hist = hist / hist.sum()
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#F5F5F5')
    
    ax.bar(range(n_bins), hist, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axhline(y=1/n_bins, color='red', linestyle='--', linewidth=2, label='Uniform')
    
    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_xticks(range(n_bins))
    ax.legend()
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Rank Histogram (Lead {lead})', fontsize=14, fontweight='bold')
    
    ax.grid(True, color='white', linewidth=1)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_metrics_by_lead(
    ds: xr.Dataset,
    metrics: list[str] | None = None,
    figsize: tuple = (14, 10),
    title: str | None = None,
):
    """
    Plot selected metrics across lead times.
    """
    import matplotlib.pyplot as plt
    
    if metrics is None:
        metrics = ["crps", "mse", "csi", "avfss_5", "eL", "spread_skill_ratio"]
    
    # Filter to available metrics
    metrics = [m for m in metrics if m in ds.data_vars]
    n_metrics = len(metrics)
    
    if n_metrics == 0:
        raise ValueError("No valid metrics to plot")
    
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    axes = np.atleast_2d(axes).flatten()
    
    leads = ds["lead"].values
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor('#F5F5F5')
        
        # Mean and std over time
        mean_vals = ds[metric].mean(dim="time", skipna=True).values
        std_vals = ds[metric].std(dim="time", skipna=True).values
        
        ax.plot(leads, mean_vals, marker='o', linewidth=2, color='steelblue')
        ax.fill_between(leads, mean_vals - std_vals, mean_vals + std_vals, 
                       alpha=0.3, color='steelblue')
        
        ax.set_xlabel('Lead', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric.upper().replace('_', ' '), fontsize=12, fontweight='bold')
        ax.grid(True, color='white', linewidth=1)
    
    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes



# =====================================================================
# =====================================================================
#
#  PIT wrapper on lat/lon window, separated from the other metrics
#
# =====================================================================
# =====================================================================

def compute_pit_on_latlon_window(
    pred: xr.DataArray,
    truth: xr.DataArray,
    window_latlon: xr.DataArray,
    *,
    apply_exact_shape_mask: bool = True,
    n_bins: int = 20,
    randomize: bool = True,
    seed: int = 42,
    intensity_thresholds: list[float] | None = None,
) -> dict:
    if intensity_thresholds is None:
        intensity_thresholds = [0.0, 0.1, 1.0, np.inf]

    rng = np.random.default_rng(seed)

    pred = _ensure_datetime64_time(pred)
    truth = _ensure_datetime64_time(truth)

    y_dim, x_dim = _infer_spatial_dims(pred)
    lat_dim, lon_dim = window_latlon.dims

    y_slice, x_slice, mask_latlon_bool = mask_bbox_from_latlon_window(window_latlon)

    pred_c = pred.isel({y_dim: y_slice, x_dim: x_slice})
    truth_c = truth.isel({y_dim: y_slice, x_dim: x_slice})

    if apply_exact_shape_mask:
        mask_c = mask_latlon_bool.isel({lat_dim: y_slice, lon_dim: x_slice}).data
        mask_c_da = xr.DataArray(mask_c, dims=(y_dim, x_dim))
        pred_c = pred_c.where(mask_c_da)
        truth_c = truth_c.where(mask_c_da)

    truth_aligned = truth_c.sel(time=pred_c["time"])

    times = pred_c["time"].values
    leads = pred_c["lead"].values
    n_times = len(times)
    n_leads = len(leads)

    pit_reliability = np.full((n_times, n_leads), np.nan)
    pit_kl = np.full((n_times, n_leads), np.nan)
    pit_ks = np.full((n_times, n_leads), np.nan)

    all_pit_values = []
    per_lead_pit = {int(l): [] for l in leads}
    all_ens_agg = []
    all_obs_agg = []

    for t_idx, t in enumerate(times):
        for l_idx, lead in enumerate(leads):
            ens = pred_c.sel(time=t, lead=lead).values
            obs = truth_aligned.sel(time=t).values

            if np.all(np.isnan(obs)) or np.all(np.isnan(ens)):
                continue

            ens_clean = np.nan_to_num(ens, nan=0.0)
            obs_clean = np.nan_to_num(obs, nan=0.0)

            pit_result = compute_pit_histogram_np(
                ens_clean, obs_clean, n_bins=n_bins, randomize=randomize, rng=rng
            )
            pit_reliability[t_idx, l_idx] = pit_result["reliability_index"]
            pit_kl[t_idx, l_idx] = pit_result["kl_divergence"]
            pit_ks[t_idx, l_idx] = pit_result["ks_statistic"]

            pit_vals = pit_result["pit_values"]
            all_pit_values.append(pit_vals)
            per_lead_pit[int(lead)].append(pit_vals)

            all_ens_agg.append(ens_clean)
            all_obs_agg.append(obs_clean)

    # ---- xr.Dataset ----
    ds = xr.Dataset(
        {
            "pit_reliability": (["time", "lead"], pit_reliability),
            "pit_kl": (["time", "lead"], pit_kl),
            "pit_ks": (["time", "lead"], pit_ks),
        },
        coords={"time": times, "lead": leads},
    )
    ds.attrs["n_bins"] = n_bins
    ds.attrs["randomize"] = str(randomize)
    ds.attrs["seed"] = seed

    # ---- Aggregated global histogram ----
    if all_pit_values:
        all_pit_flat = np.concatenate(all_pit_values)
    else:
        all_pit_flat = np.array([])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    uniform = 1.0 / n_bins
    if len(all_pit_flat) > 0:
        hist, _ = np.histogram(all_pit_flat, bins=bin_edges)
        hist_norm = hist / hist.sum()
        global_histogram = {
            "histogram": hist_norm,
            "bin_edges": bin_edges,
            "reliability_index": float(np.mean(np.abs(hist_norm - uniform))),
            "kl_divergence": _kl_from_uniform(hist_norm),
            "n_values": len(all_pit_flat),
        }
    else:
        global_histogram = {
            "histogram": np.full(n_bins, np.nan),
            "bin_edges": bin_edges,
            "reliability_index": np.nan,
            "kl_divergence": np.nan,
            "n_values": 0,
        }

    # ---- Per-lead histograms ----
    per_lead_histograms = {}
    for lead_val, pit_list in per_lead_pit.items():
        if pit_list:
            pit_cat = np.concatenate(pit_list)
            hist, _ = np.histogram(pit_cat, bins=bin_edges)
            hist_norm = hist / hist.sum()
            per_lead_histograms[lead_val] = {
                "histogram": hist_norm,
                "bin_edges": bin_edges,
                "reliability_index": float(np.mean(np.abs(hist_norm - uniform))),
                "kl_divergence": _kl_from_uniform(hist_norm),
                "n_values": len(pit_cat),
            }
        else:
            per_lead_histograms[lead_val] = {
                "histogram": np.full(n_bins, np.nan),
                "bin_edges": bin_edges,
                "reliability_index": np.nan,
                "kl_divergence": np.nan,
                "n_values": 0,
            }

    # ---- Intensity-conditioned ----
    if all_ens_agg:
        all_ens_stack = np.concatenate(
            [e.reshape(e.shape[0], -1) for e in all_ens_agg], axis=1
        )
        all_obs_stack = np.concatenate([o.ravel() for o in all_obs_agg])
        by_intensity = compute_pit_by_intensity_np(
            all_ens_stack, all_obs_stack,
            thresholds=intensity_thresholds,
            n_bins=n_bins, randomize=randomize, rng=rng,
        )
    else:
        by_intensity = {}

    return {
        "ds": ds,
        "pit_all": all_pit_flat,
        "histogram": global_histogram,
        "by_intensity": by_intensity,
        "per_lead_histograms": per_lead_histograms,
    }



def plot_pit_summary(
    pit_result: dict,
    model_name: str = "",
    figsize: tuple = (18, 5),
) -> tuple:
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")
    prefix = f"{model_name} — " if model_name else ""

    # ---- Panel 1: Global histogram ----
    ax = axes[0]
    ax.set_facecolor("#F5F5F5")
    hist_data = pit_result["histogram"]
    edges = hist_data["bin_edges"]
    n_bins = len(hist_data["histogram"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax.bar(centers, hist_data["histogram"], width=width * 0.9,
           color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axhline(y=1.0 / n_bins, color="red", linestyle="--", linewidth=2,
               label="Uniform (ideal)")
    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Relative frequency", fontsize=11)
    ax.set_title(f"{prefix}Global PIT histogram", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ri = hist_data["reliability_index"]
    kl = hist_data["kl_divergence"]
    n_val = hist_data["n_values"]
    ax.text(0.98, 0.95, f"RI = {ri:.4f}  |  KL = {kl:.4f}\nn = {n_val:,}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(fontsize=9)
    ax.grid(True, color="white", linewidth=1)

    # ---- Panel 2: By intensity ----
    ax = axes[1]
    ax.set_facecolor("#F5F5F5")
    by_int = pit_result["by_intensity"]
    colors_int = ["#90CAF9", "#42A5F5", "#1565C0", "#0D47A1"]
    n_regimes = len(by_int)

    if n_regimes > 0:
        bar_width = width * 0.85 / n_regimes
        for idx, (regime_label, regime_data) in enumerate(by_int.items()):
            hist = regime_data["histogram"]
            if np.all(np.isnan(hist)):
                continue
            c = colors_int[idx % len(colors_int)]
            offset = (idx - n_regimes / 2 + 0.5) * bar_width
            kl_r = regime_data["kl_divergence"]
            ax.bar(centers + offset, hist, width=bar_width, color=c,
                   edgecolor="white", linewidth=0.3, alpha=0.8,
                   label=f"{regime_label} (KL={kl_r:.4f})")
        ax.axhline(y=1.0 / n_bins, color="red", linestyle="--", linewidth=2)
        ax.legend(fontsize=8, loc="upper left")
    else:
        ax.text(0.5, 0.5, "No intensity data", transform=ax.transAxes,
                ha="center", va="center")

    ax.set_xlabel("PIT value", fontsize=11)
    ax.set_ylabel("Relative frequency", fontsize=11)
    ax.set_title(f"{prefix}PIT by intensity", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(True, color="white", linewidth=1)

    # ---- Panel 3: KL divergence by lead ----
    ax = axes[2]
    ax.set_facecolor("#F5F5F5")
    ds = pit_result["ds"]
    leads = ds["lead"].values
    kl_per_lead = ds["pit_kl"].mean(dim="time", skipna=True).values
    kl_std = ds["pit_kl"].std(dim="time", skipna=True).values

    ax.plot(leads, kl_per_lead, "o-", color="steelblue", linewidth=2, markersize=5)
    ax.fill_between(leads, kl_per_lead - kl_std, kl_per_lead + kl_std,
                    alpha=0.25, color="steelblue")
    ax.set_xlabel("Lead", fontsize=11)
    ax.set_ylabel("KL divergence from uniform", fontsize=11)
    ax.set_title(f"{prefix}Calibration by lead time", fontsize=12, fontweight="bold")
    ax.grid(True, color="white", linewidth=1)

    plt.tight_layout()
    return fig, axes


def plot_pit_comparison_multi(
    pit_results: dict[str, dict],
    n_bins: int = 20,
    figsize: tuple = (9, 5),
    title: str = "PIT Histogram Comparison",
) -> tuple:
    colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2", "#F9A825", "#00838F"]
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("#F5F5F5")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]
    n_models = len(pit_results)
    bar_width = width * 0.85 / n_models

    for idx, (name, pit_res) in enumerate(pit_results.items()):
        hist = pit_res["histogram"]["histogram"]
        kl = pit_res["histogram"]["kl_divergence"]
        c = colors[idx % len(colors)]
        offset = (idx - n_models / 2 + 0.5) * bar_width
        ax.bar(centers + offset, hist, width=bar_width, color=c,
               edgecolor="white", linewidth=0.3, alpha=0.8,
               label=f"{name} (KL={kl:.4f})")

    ax.axhline(y=1.0 / n_bins, color="red", linestyle="--", linewidth=2,
               label="Uniform (ideal)")
    ax.set_xlabel("PIT value", fontsize=12)
    ax.set_ylabel("Relative frequency", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, color="white", linewidth=1)

    plt.tight_layout()
    return fig, ax
 

def plot_pit_leads_panel(
    pit_result: dict,
    leads_to_show: list[int] | None = None,
    n_bins: int = 20,
    figsize: tuple | None = None,
    model_name: str = "",
) -> tuple:
    per_lead = pit_result["per_lead_histograms"]
    if leads_to_show is None:
        leads_to_show = sorted(per_lead.keys())

    n = len(leads_to_show)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="white")
    axes_flat = np.atleast_1d(axes).flatten()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1] - bin_edges[0]

    for idx, lead in enumerate(leads_to_show):
        ax = axes_flat[idx]
        ax.set_facecolor("#F5F5F5")

        data = per_lead.get(lead, None)
        if data is None or np.all(np.isnan(data["histogram"])):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            ax.bar(centers, data["histogram"], width=width * 0.9,
                   color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.axhline(y=1.0 / n_bins, color="red", linestyle="--", linewidth=1.5)
            kl = data["kl_divergence"]
            ax.text(0.97, 0.95, f"KL={kl:.4f}", transform=ax.transAxes,
                    fontsize=8, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        ax.set_title(f"Lead {lead}", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(True, color="white", linewidth=1)
        if idx % ncols == 0:
            ax.set_ylabel("Freq.", fontsize=9)
        if idx >= n - ncols:
            ax.set_xlabel("PIT", fontsize=9)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    prefix = f"{model_name} — " if model_name else ""
    fig.suptitle(f"{prefix}PIT by lead time", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig, axes


# =====================================================================
#  Example usage
# =====================================================================
#
#  # --- Compute for one model ---
#  pit_result = compute_pit_on_latlon_window(
#      pred=my_pred_da,       # (ensemble, time, lead, y, x)
#      truth=my_truth_da,     # (time, y, x)
#      window_latlon=my_window,
#      n_bins=20,
#  )
#
#  # 3-panel summary
#  fig, axes = plot_pit_summary(pit_result, model_name="CorrDiff UNet_λ2")
#
#  # Per-lead panel
#  fig, axes = plot_pit_leads_panel(pit_result, model_name="CorrDiff UNet_λ2")
#
#  # --- Compare multiple models ---
#  pit_model_A = compute_pit_on_latlon_window(pred_A, truth, window)
#  pit_model_B = compute_pit_on_latlon_window(pred_B, truth, window)
#
#  fig, ax = plot_pit_comparison_multi(
#      {"Model A": pit_model_A, "Model B": pit_model_B},
#      title="PIT Comparison",
#  )
#
#  # --- Access the xr.Dataset for further analysis ---
#  ds = pit_result["ds"]
#  print(ds["pit_reliability"].mean(dim="time"))  # RI per lead


"""
Multi-model PIT comparison plots.

Two functions:
  1. plot_kl_by_lead()   — KL divergence vs lead time, one line per model
  2. plot_pit_by_intensity_multi() — grouped bars: KL per intensity class per model

Both can be combined in a single 1×2 figure via plot_pit_comparison_combined().

Expected input: dict of PIT results per model, where each entry has:
    pit_result["per_lead_histograms"]  — dict {lead_idx: {"histogram": ..., "kl_divergence": float}}
    pit_result["by_intensity"]         — dict {label: {"histogram": ..., "kl_divergence": float}}
    pit_result["histogram"]            — global {"histogram": ..., "kl_divergence": float, "reliability_index": float}

Alternatively, you can pass raw arrays directly (see function signatures).
"""


def _format_lead(lead_idx):
    h = (int(lead_idx) + 1) * 6
    if h >= 48 and h % 24 == 0:
        return f"{h // 24}d"
    return f"{h}h"


# Colorblind-friendly palette (Wong 2011)
_CB_PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]


def plot_kl_by_lead(
    model_kl,
    leads=None,
    colors=None,
    figsize=(7, 4),
    title="KL divergence by lead time",
    ax=None,
):
    """
    Line plot: KL divergence vs lead time for multiple models.

    Parameters
    ----------
    model_kl : dict[str, dict or array-like]
        {model_name: data} where data is one of:
          - dict mapping lead_idx (int) → KL value (float)
          - pit_result dict with pit_result["per_lead_histograms"][lead]["kl_divergence"]
          - 1D array of KL values (then `leads` must be provided)
    leads : list[int] or None
        Lead indices. Auto-detected from first model's keys if None.
    colors : list[str] or None
    figsize : tuple
    title : str
    ax : matplotlib.Axes or None

    Returns
    -------
    fig (or None if ax was provided), ax
    """
    model_names = list(model_kl.keys())
    n_models = len(model_names)
    if colors is None:
        colors = [_CB_PALETTE[i % len(_CB_PALETTE)] for i in range(n_models)]

    # Parse data
    parsed = {}
    for name, data in model_kl.items():
        if isinstance(data, dict) and "per_lead_histograms" in data:
            # Full pit_result
            plh = data["per_lead_histograms"]
            parsed[name] = {int(k): v["kl_divergence"] for k, v in plh.items()}
        elif isinstance(data, dict):
            # Already lead_idx → KL
            parsed[name] = {int(k): float(v) for k, v in data.items()}
        else:
            # Array
            assert leads is not None, "Provide `leads` when passing arrays"
            parsed[name] = {int(l): float(v) for l, v in zip(leads, data)}

    # Determine lead indices
    if leads is None:
        leads = sorted(parsed[model_names[0]].keys())

    hours = [(l + 1) * 6 for l in leads]

    # Plot
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    for mi, name in enumerate(model_names):
        kl_vals = [parsed[name].get(l, np.nan) for l in leads]
        ax.plot(hours, kl_vals, "-o", color=colors[mi], lw=1.8,
                markersize=4, label=name, alpha=0.85)

    ax.set_xlabel("Lead time (hours)", fontsize=13, color="black")
    ax.set_ylabel("KL divergence from uniform", fontsize=13, color="black")
    ax.set_xticks(hours)
    ax.set_xticklabels([_format_lead(l) for l in leads], fontsize=8)
    ax.legend(fontsize=10, frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, color="black", lw=0.3)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', labelsize=13, rotation=70)
    ax.tick_params(axis='y', labelsize=13)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    if own_fig:
        fig.tight_layout()
    return fig, ax


def plot_pit_by_intensity_multi(
    model_intensity,
    colors=None,
    figsize=(7, 4),
    title="KL divergence by precipitation intensity",
    ax=None,
):
    """
    Grouped bar chart: KL per intensity class, one group per model.

    Parameters
    ----------
    model_intensity : dict[str, dict]
        {model_name: data} where data is one of:
          - dict mapping class_label → KL value
            e.g. {"<0.1mm": 0.08, "0.1-1.0mm": 1.19, ">=1.0mm": 1.86}
          - pit_result dict with pit_result["by_intensity"][label]["kl_divergence"]
    colors : list[str] or None
    figsize : tuple
    title : str
    ax : matplotlib.Axes or None

    Returns
    -------
    fig (or None), ax
    """
    model_names = list(model_intensity.keys())
    n_models = len(model_names)
    if colors is None:
        colors = [_CB_PALETTE[i % len(_CB_PALETTE)] for i in range(n_models)]

    # Parse
    parsed = {}
    for name, data in model_intensity.items():
        if isinstance(data, dict) and "by_intensity" in data:
            bi = data["by_intensity"]
            parsed[name] = {k: v["kl_divergence"] for k, v in bi.items()}
        else:
            parsed[name] = {k: float(v) for k, v in data.items()}

    # Get intensity classes from first model
    classes = list(parsed[model_names[0]].keys())
    n_classes = len(classes)

    # Bar positions
    x = np.arange(n_classes)
    bar_width = 0.8 / n_models

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    for mi, name in enumerate(model_names):
        offsets = x - 0.4 + bar_width * (mi + 0.5)
        vals = [parsed[name].get(c, 0) for c in classes]
        bars = ax.bar(offsets, vals, bar_width * 0.9, color=colors[mi],
                      alpha=0.8, label=name, edgecolor="white", lw=0.5)

        # Value labels on top
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                    color=colors[mi], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylabel("KL divergence", fontsize=13, color="black")
    ax.set_xlabel("Precipitation intensity", fontsize=13, color="black")
    ax.legend(fontsize=10, frameon=True, fancybox=True)
    ax.grid(True, axis="y", alpha=0.3, color="black", lw=0.3)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', labelsize=13)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    if own_fig:
        fig.tight_layout()
    return fig, ax


def plot_pit_histogram_multi(
    model_histograms,
    n_bins=20,
    colors=None,
    figsize=(6, 4),
    title="Global PIT histogram",
    ax=None,
):
    """
    Overlaid PIT histograms for multiple models (grouped bars).

    Parameters
    ----------
    model_histograms : dict[str, dict or array-like]
        {model_name: data} where data is one of:
          - pit_result dict with pit_result["histogram"]["histogram"] (array)
            and pit_result["histogram"]["kl_divergence"] (float)
          - 1D array of histogram bin heights (length n_bins)
    n_bins : int
    colors, figsize, title, ax : as usual

    Returns
    -------
    fig (or None), ax
    """
    model_names = list(model_histograms.keys())
    n_models = len(model_names)
    if colors is None:
        colors = [_CB_PALETTE[i % len(_CB_PALETTE)] for i in range(n_models)]

    # Parse
    hists = {}
    kl_vals = {}
    for name, data in model_histograms.items():
        if isinstance(data, dict) and "histogram" in data:
            inner = data["histogram"]
            if isinstance(inner, dict):
                hists[name] = np.array(inner["histogram"])
                kl_vals[name] = inner.get("kl_divergence", None)
            else:
                hists[name] = np.array(inner)
                kl_vals[name] = None
        else:
            hists[name] = np.array(data)
            kl_vals[name] = None

    n_bins = len(hists[model_names[0]])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = (1.0 / n_bins) * 0.85 / n_models

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    for mi, name in enumerate(model_names):
        offsets = bin_centers - (0.85 / n_bins) / 2 + bar_width * (mi + 0.5)
        kl_str = f" (KL={kl_vals[name]:.3f})" if kl_vals[name] is not None else ""
        ax.bar(offsets, hists[name], bar_width * 0.9, color=colors[mi],
               alpha=0.75, label=f"{name}{kl_str}", edgecolor="white", lw=0.3)

    # Uniform reference
    ax.axhline(1.0 / n_bins, color="red", ls="--", lw=1.5, label="Uniform (ideal)", zorder=5)

    ax.set_xlabel("PIT value", fontsize=13, color="black")
    ax.set_ylabel("Relative frequency", fontsize=13, color="black")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10, frameon=True, fancybox=True, loc="upper center")
    ax.grid(True, axis="y", alpha=0.3, color="black", lw=0.3)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', labelsize=13)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    if own_fig:
        fig.tight_layout()
    return fig, ax


def plot_pit_comparison_combined(
    pit_results,
    leads=None,
    colors=None,
    figsize=(18, 4.5),
    suptitle=None,
):
    """
    Combined 1×3 figure:
      (a) Global PIT histogram
      (b) KL divergence by lead time
      (c) KL divergence by intensity

    Parameters
    ----------
    pit_results : dict[str, dict]
        {model_name: pit_result} where pit_result contains:
          - "histogram"            → {"histogram": array, "kl_divergence": float, ...}
          - "per_lead_histograms"  → {lead: {"kl_divergence": float, ...}}
          - "by_intensity"         → {class: {"kl_divergence": float, ...}}
    leads : list[int] or None
    colors : list[str] or None
    figsize : tuple
    suptitle : str

    Returns
    -------
    fig
    """
    model_names = list(pit_results.keys())
    n_models = len(model_names)
    if colors is None:
        colors = [_CB_PALETTE[i % len(_CB_PALETTE)] for i in range(n_models)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    plot_pit_histogram_multi(pit_results, colors=colors,
                             title="(a) Global PIT histogram", ax=ax1)

    plot_kl_by_lead(pit_results, leads=leads, colors=colors,
                    title="(b) KL divergence by lead time", ax=ax2)

    plot_pit_by_intensity_multi(pit_results, colors=colors,
                                title="(c) KL divergence by intensity", ax=ax3)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig


