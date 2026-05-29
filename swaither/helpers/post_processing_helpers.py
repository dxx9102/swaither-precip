"""
Post-processing helpers (after the super resolution step).

Contains helpers that are used within the Step2 evaluation notebook, to postprocess after high resolution samples are imported.
"""

import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union

def postprocess_precipitation_data(
    pred: xr.DataArray,
    obs: xr.DataArray,
    lr_bias_corrected: Optional[xr.Dataset] = None,
    lr_var_name: str = "precip6h_cumulative_capped",
    threshold: float = 0.1,
    use_lr_mask: bool = True,
    lr_intensity_percentile: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Unified post-processing for precipitation predictions and observations.
    
    Applies consistent processing to ensure fair evaluation:
    1. (Optional) LR mask on predictions: zero out HR pixels where LR has no precipitation
    2. (Optional) LR intensity mask: additionally zero out pixels where LR intensity is below percentile
    3. Threshold on predictions: set values < threshold to 0
    4. Same threshold on observations: ensures fair comparison
    
    Args:
        pred: HR predictions (ensemble, time, lead, y, x)
        obs: HR observations (time, y, x) 
        lr_bias_corrected: LR bias-corrected Dataset (time, prediction_delta, lat, lon)
                          Required if use_lr_mask=True
        lr_var_name: Variable name in LR Dataset
        threshold: Precipitation threshold in mm (applied to both pred and obs)
        use_lr_mask: Whether to apply LR spatial mask to predictions
        lr_intensity_percentile: If set, additionally mask out LR pixels below this
                                 intensity percentile (e.g., 20 = bottom 20%)
        verbose: Print detailed statistics
    
    Returns:
        Tuple of (pred_clean, obs_clean)
    
    Example:
        pred_clean, obs_clean = postprocess_precipitation_data(
            pred=pred1,
            obs=truth_hr,
            lr_bias_corrected=bias_corrected_1,
            threshold=0.1,
            use_lr_mask=True,
            lr_intensity_percentile=20,  # Optional: mask low-intensity LR edges
            verbose=True,
        )
    """
    if verbose:
        print("=" * 60)
        print("PRECIPITATION POST-PROCESSING")
        print("=" * 60)
        print(f"\nSettings:")
        print(f"  Threshold: {threshold} mm")
        print(f"  Use LR mask: {use_lr_mask}")
        if lr_intensity_percentile is not None:
            print(f"  LR intensity percentile filter: {lr_intensity_percentile}%")
        print()
    
    # -------------------------------------------------------------------------
    # Step 1: Process predictions
    # -------------------------------------------------------------------------
    pred_clean = pred.copy(deep=True)
    pred_vals = pred_clean.values.copy()
    
    if verbose:
        n_pred_positive_initial = np.sum(pred_vals > 0)
        sum_pred_initial = np.sum(pred_vals)
        print("PREDICTIONS:")
        print(f"  Initial: {n_pred_positive_initial:,} positive pixels, sum={sum_pred_initial:.1f}")
    
    # Step 1a: Apply LR mask (if requested)
    if use_lr_mask:
        if lr_bias_corrected is None:
            raise ValueError("lr_bias_corrected is required when use_lr_mask=True")
        
        pred_vals = _apply_lr_mask(
            pred_vals=pred_vals,
            pred_da=pred,
            lr_bias_corrected=lr_bias_corrected,
            lr_var_name=lr_var_name,
            lr_intensity_percentile=lr_intensity_percentile,
            verbose=verbose,
        )
        
        if verbose:
            n_after_lr_mask = np.sum(pred_vals > 0)
            print(f"  After LR mask: {n_after_lr_mask:,} positive pixels "
                  f"({n_pred_positive_initial - n_after_lr_mask:,} removed)")
    
    # Step 1b: Apply threshold to predictions
    n_below_threshold = np.sum((pred_vals > 0) & (pred_vals < threshold))
    pred_vals[pred_vals < threshold] = 0.0
    
    if verbose:
        n_after_threshold = np.sum(pred_vals > 0)
        sum_pred_final = np.sum(pred_vals)
        print(f"  After threshold ({threshold} mm): {n_after_threshold:,} positive pixels "
              f"({n_below_threshold:,} removed)")
        print(f"  Final sum: {sum_pred_final:.1f} "
              f"({100*(sum_pred_initial - sum_pred_final)/sum_pred_initial:.1f}% total reduction)")
    
    pred_clean.values = pred_vals
    
    # -------------------------------------------------------------------------
    # Step 2: Process observations (threshold only - for fair comparison)
    # -------------------------------------------------------------------------
    obs_clean = obs.copy(deep=True)
    obs_vals = obs_clean.values.copy()
    
    if verbose:
        n_obs_positive_initial = np.sum(obs_vals > 0)
        sum_obs_initial = np.sum(obs_vals)
        print(f"\nOBSERVATIONS:")
        print(f"  Initial: {n_obs_positive_initial:,} positive pixels, sum={sum_obs_initial:.1f}")
    
    n_obs_below_threshold = np.sum((obs_vals > 0) & (obs_vals < threshold))
    obs_vals[obs_vals < threshold] = 0.0
    
    if verbose:
        n_obs_after_threshold = np.sum(obs_vals > 0)
        sum_obs_final = np.sum(obs_vals)
        print(f"  After threshold ({threshold} mm): {n_obs_after_threshold:,} positive pixels "
              f"({n_obs_below_threshold:,} removed)")
        print(f"  Final sum: {sum_obs_final:.1f} "
              f"({100*(sum_obs_initial - sum_obs_final)/sum_obs_initial:.1f}% total reduction)")
    
    obs_clean.values = obs_vals
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    if verbose:
        print(f"\n" + "=" * 60)
        print("POST-PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Predictions: {n_pred_positive_initial:,} → {np.sum(pred_clean.values > 0):,} positive pixels")
        print(f"Observations: {n_obs_positive_initial:,} → {np.sum(obs_clean.values > 0):,} positive pixels")
        print(f"\nBoth datasets are now consistently processed and ready for")
        print(f"fair metric computation and spectral analysis.")
        print("=" * 60)
    
    return pred_clean, obs_clean


def _apply_lr_mask(
    pred_vals: np.ndarray,
    pred_da: xr.DataArray,
    lr_bias_corrected: xr.Dataset,
    lr_var_name: str,
    lr_intensity_percentile: Optional[float] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Apply LR mask to HR predictions.
    
    Internal function called by postprocess_precipitation_data.
    """
    lr_precip = lr_bias_corrected[lr_var_name]
    
    n_ens = pred_da.sizes['ensemble']
    hr_times = pred_da['time'].values
    hr_leads = pred_da['lead'].values
    hr_spatial_shape = (pred_da.sizes['y'], pred_da.sizes['x'])
    
    # Build lead mapping: HR lead index -> LR prediction_delta
    lead_mapping = {}
    for hr_pos, hr_lead_val in enumerate(hr_leads):
        hours = (int(hr_lead_val) + 1) * 6
        lr_delta = np.timedelta64(int(hours * 3600 * 1e9), 'ns')
        lead_mapping[hr_pos] = lr_delta
    
    n_masked = 0
    n_processed = 0
    
    for t_idx, t in enumerate(hr_times):
        for l_idx in range(len(hr_leads)):
            lr_delta = lead_mapping[l_idx]
            
            try:
                lr_field = lr_precip.sel(time=t, prediction_delta=lr_delta).values
                lr_field = np.asarray(lr_field).squeeze()
            except KeyError:
                continue
            
            if lr_field.ndim != 2:
                continue
            
            # Create LR mask
            lr_mask = lr_field > 0
            
            # Optional: additionally filter by intensity percentile
            if lr_intensity_percentile is not None and np.any(lr_mask):
                lr_positive = lr_field[lr_mask]
                intensity_threshold = np.percentile(lr_positive, lr_intensity_percentile)
                lr_mask = lr_field >= intensity_threshold
            
            # Upscale mask to HR
            lr_mask_float = lr_mask.astype(np.float32)
            lr_tensor = torch.from_numpy(lr_mask_float).unsqueeze(0).unsqueeze(0)
            hr_mask = F.interpolate(lr_tensor, size=hr_spatial_shape, mode='nearest')
            hr_mask = hr_mask.squeeze().numpy().astype(bool)
            
            # Count pixels being masked
            for e in range(n_ens):
                n_masked += np.sum((pred_vals[e, t_idx, l_idx] > 0) & ~hr_mask)
                pred_vals[e, t_idx, l_idx][~hr_mask] = 0.0
            
            n_processed += 1
    
    if verbose:
        print(f"  LR mask applied to {n_processed} (time, lead) pairs")
        if lr_intensity_percentile is not None:
            print(f"  (with {lr_intensity_percentile}th percentile intensity filter)")
    
    return pred_vals

# =============================================================================
# Convenience functions for common use cases
# =============================================================================

def postprocess_threshold_only(
    pred: xr.DataArray,
    obs: xr.DataArray,
    threshold: float = 0.1,
    verbose: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Simple threshold-only post-processing (literature standard).
    
    Use this when you don't have LR bias-corrected data or want
    the simplest approach.
    """
    return postprocess_precipitation_data(
        pred=pred,
        obs=obs,
        lr_bias_corrected=None,
        threshold=threshold,
        use_lr_mask=False,
        verbose=verbose,
    )


def postprocess_with_lr_mask(
    pred: xr.DataArray,
    obs: xr.DataArray,
    lr_bias_corrected: xr.Dataset,
    lr_var_name: str = "precip6h_cumulative_capped",
    threshold: float = 0.1,
    verbose: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Post-processing with LR mask (recommended for two-step methods).
    
    Applies LR spatial mask + threshold to predictions,
    and same threshold to observations.
    """
    return postprocess_precipitation_data(
        pred=pred,
        obs=obs,
        lr_bias_corrected=lr_bias_corrected,
        lr_var_name=lr_var_name,
        threshold=threshold,
        use_lr_mask=True,
        lr_intensity_percentile=None,
        verbose=verbose,
    )


def postprocess_with_lr_mask_and_intensity(
    pred: xr.DataArray,
    obs: xr.DataArray,
    lr_bias_corrected: xr.Dataset,
    lr_var_name: str = "precip6h_cumulative_capped",
    threshold: float = 0.1,
    lr_intensity_percentile: float = 20,
    verbose: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Post-processing with LR mask + intensity filter (most aggressive).
    
    Additionally filters out low-intensity LR edges to reduce
    the noise ring artifact.
    """
    return postprocess_precipitation_data(
        pred=pred,
        obs=obs,
        lr_bias_corrected=lr_bias_corrected,
        lr_var_name=lr_var_name,
        threshold=threshold,
        use_lr_mask=True,
        lr_intensity_percentile=lr_intensity_percentile,
        verbose=verbose,
    )


# =============================================================================
# Usage Examples
# =============================================================================

'''
# Option 1: Simple threshold only (literature standard)
pred_clean, obs_clean = postprocess_threshold_only(
    pred=pred1,
    obs=truth_hr,
    threshold=0.1,
)

# Option 2: LR mask + threshold (recommended)
pred_clean, obs_clean = postprocess_with_lr_mask(
    pred=pred1,
    obs=truth_hr,
    lr_bias_corrected=bias_corrected_1,
    threshold=0.1,
)

# Option 3: LR mask + intensity filter + threshold (most aggressive)
pred_clean, obs_clean = postprocess_with_lr_mask_and_intensity(
    pred=pred1,
    obs=truth_hr,
    lr_bias_corrected=bias_corrected_1,
    threshold=0.1,
    lr_intensity_percentile=20,  # Mask bottom 20% intensity in LR
)

# Option 4: Full control
pred_clean, obs_clean = postprocess_precipitation_data(
    pred=pred1,
    obs=truth_hr,
    lr_bias_corrected=bias_corrected_1,
    lr_var_name="precip6h_cumulative_capped",
    threshold=0.1,
    use_lr_mask=True,
    lr_intensity_percentile=20,
    verbose=True,
)

# Now use for evaluation
metrics = compute_all_metrics(pred_clean, obs_clean, ...)
spectral = compute_spectral_analysis(pred_clean, obs_clean, ...)
'''