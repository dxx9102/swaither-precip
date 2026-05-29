"""
Bias-correction evaluation metrics.

Contains helpers that are used in the Step1 (Bias correction) evaluation notebook.
Contains:
  - baseline_per_lead_metrics   (for numpy-based baseline evaluation)
  - EvaluateModel               (aggregate metrics over all leads)
  - EvaluateModelPerLead         (per-lead-time metrics for DL models)
  - evaluate / evaluate_by_lead_time  (lower-level evaluation loops)
  - helpers: _safe_mean, _event_stats, _masked_errors,
             _fss_fast, _sal_fast, _spatial_metrics_fast,
             _upsample_input, _coarsen_to_target
"""

import torch
import torch.nn.functional as F
import numpy as np
import math

# =============================================================================
# Helper functions 
# =============================================================================
def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(float("nan"), device=x.device)
    return x.mean()


def _upsample_input(
    x: torch.Tensor, target_shape: tuple, mode: str = "bilinear"
) -> torch.Tensor:
    squeeze_channel = False
    if x.dim() == 3:
        x = x.unsqueeze(1)
        squeeze_channel = True

    _, _, H_in, W_in = x.shape
    H_target, W_target = target_shape

    if H_in == H_target and W_in == W_target:
        if squeeze_channel:
            x = x.squeeze(1)
        return x

    align_corners = True if mode in ["bilinear", "bicubic"] else None
    x_up = F.interpolate(
        x, size=(H_target, W_target), mode=mode, align_corners=align_corners
    )

    if squeeze_channel:
        x_up = x_up.squeeze(1)

    return x_up


def _coarsen_to_target(
    y_pred: torch.Tensor, target_shape: tuple, mode: str = "avg"
) -> torch.Tensor:
    squeeze_channel = False
    if y_pred.dim() == 3:
        y_pred = y_pred.unsqueeze(1)
        squeeze_channel = True

    _, _, H_pred, W_pred = y_pred.shape
    H_target, W_target = target_shape

    if H_pred == H_target and W_pred == W_target:
        if squeeze_channel:
            y_pred = y_pred.squeeze(1)
        return y_pred

    if H_pred < H_target or W_pred < W_target:
        raise ValueError(f"Target shape {target_shape} is larger than prediction shape")

    if H_pred % H_target != 0 or W_pred % W_target != 0:
        raise ValueError(f"Prediction shape must be integer multiple of target shape")

    scale_h = H_pred // H_target
    scale_w = W_pred // W_target

    if mode == "avg":
        y_coarse = F.avg_pool2d(
            y_pred, kernel_size=(scale_h, scale_w), stride=(scale_h, scale_w)
        )
    elif mode == "sum":
        y_coarse = F.avg_pool2d(
            y_pred, kernel_size=(scale_h, scale_w), stride=(scale_h, scale_w)
        ) * (scale_h * scale_w)
    else:
        raise ValueError(f"Unknown coarsening mode: {mode}")

    if squeeze_channel:
        y_coarse = y_coarse.squeeze(1)

    return y_coarse


# =============================================================================
# Point-wise metrics
# =============================================================================
def _event_stats(y_true: torch.Tensor, y_pred: torch.Tensor, thr: float = 0.1):
    if y_true.dim() == 4 and y_true.size(1) == 1:
        y_true = y_true[:, 0]
    if y_pred.dim() == 4 and y_pred.size(1) == 1:
        y_pred = y_pred[:, 0]

    obs = y_true >= thr
    prd = y_pred >= thr

    TP = (prd & obs).sum().float()
    FP = (prd & ~obs).sum().float()
    FN = (~prd & obs).sum().float()
    TN = (~prd & ~obs).sum().float()

    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    csi = TP / (TP + FP + FN + eps)
    far = FP / (TP + FP + eps)
    bias = (TP + FP) / (TP + FN + eps)

    return {
        "precision": precision,
        "recall_POD": recall,
        "F1": f1,
        "CSI": csi,
        "FAR": far,
        "Bias": bias,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
    }


def _masked_errors(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    wet_thr: float = 0.1,
    heavy_thr: float = 1.0,
):
    if y_true.dim() == 4 and y_true.size(1) == 1:
        y_true = y_true[:, 0]
    if y_pred.dim() == 4 and y_pred.size(1) == 1:
        y_pred = y_pred[:, 0]

    diff = y_pred - y_true
    se = diff**2
    ae = diff.abs()

    mse_all = _safe_mean(se)
    mae_all = _safe_mean(ae)

    wet_mask = y_true >= wet_thr
    heavy_mask = y_true >= heavy_thr

    mse_wet = _safe_mean(se[wet_mask])
    mae_wet = _safe_mean(ae[wet_mask])
    mse_heavy = _safe_mean(se[heavy_mask])
    mae_heavy = _safe_mean(ae[heavy_mask])

    w = torch.ones_like(y_true)
    w[wet_mask] = 5.0
    weighted_mse = (w * se).sum() / (w.sum() + 1e-8)

    log_err_sq = (
        torch.log1p(y_pred.clamp(min=0.0)) - torch.log1p(y_true.clamp(min=0.0))
    ) ** 2
    log1p_rmse = torch.sqrt(_safe_mean(log_err_sq))

    return {
        "MSE_all": mse_all,
        "MAE_all": mae_all,
        "MSE_wet": mse_wet,
        "MAE_wet": mae_wet,
        "MSE_heavy": mse_heavy,
        "MAE_heavy": mae_heavy,
        "WeightedMSE": weighted_mse,
        "Log1p_RMSE": log1p_rmse,
        "wet_count": wet_mask.sum().float(),
        "heavy_count": heavy_mask.sum().float(),
        "total_count": torch.tensor(y_true.numel(), device=y_true.device).float(),
    }


# =============================================================================
# FAST FSS - using avg_pool2d instead of conv2d
# =============================================================================
def _fss_fast(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float,
    neighborhood: int,
) -> float:
    """
    Fast FSS computation using avg_pool2d.
    Much faster than conv2d approach!
    """
    if y_true.dim() == 4 and y_true.size(1) == 1:
        y_true = y_true[:, 0]
    if y_pred.dim() == 4 and y_pred.size(1) == 1:
        y_pred = y_pred[:, 0]

    # Binary fields
    obs_binary = (y_true >= threshold).float().unsqueeze(1)  # (N, 1, H, W)
    pred_binary = (y_pred >= threshold).float().unsqueeze(1)

    # Use avg_pool2d with padding for fractions (faster than conv2d!)
    pad = neighborhood // 2

    # Pad manually to handle edges
    obs_padded = F.pad(obs_binary, (pad, pad, pad, pad), mode="reflect")
    pred_padded = F.pad(pred_binary, (pad, pad, pad, pad), mode="reflect")

    frac_obs = F.avg_pool2d(obs_padded, kernel_size=neighborhood, stride=1)
    frac_pred = F.avg_pool2d(pred_padded, kernel_size=neighborhood, stride=1)

    # MSE of fractions
    mse_frac = ((frac_pred - frac_obs) ** 2).mean()

    # Reference MSE
    mse_ref = (frac_obs**2).mean() + (frac_pred**2).mean()

    if mse_ref < 1e-10:
        return np.nan

    fss = 1.0 - (mse_frac / mse_ref)
    return fss.item()


# =============================================================================
# FAST SAL - simplified and vectorized
# =============================================================================
def _sal_fast(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.1,
) -> dict[str, float]:
    """
    Fast SAL computation - simplified version.
    """
    if y_true.dim() == 4 and y_true.size(1) == 1:
        y_true = y_true[:, 0]
    if y_pred.dim() == 4 and y_pred.size(1) == 1:
        y_pred = y_pred[:, 0]

    # Flatten to (N*H*W) for simplicity, or work on mean fields
    # For speed, compute SAL on the MEAN field across batch
    obs = y_true.mean(dim=0)  # (H, W)
    pred = y_pred.mean(dim=0)  # (H, W)

    H, W = obs.shape
    device = obs.device

    # Mask below threshold
    obs_masked = torch.where(obs >= threshold, obs, torch.zeros_like(obs))
    pred_masked = torch.where(pred >= threshold, pred, torch.zeros_like(pred))

    R_obs = obs_masked.sum()
    R_pred = pred_masked.sum()

    if R_obs + R_pred < 1e-10:
        return {"SAL_S": np.nan, "SAL_A": np.nan, "SAL_L": np.nan}

    # A (Amplitude)
    A = (R_pred - R_obs) / (0.5 * (R_pred + R_obs))

    # L (Location) - simplified
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

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

    max_distance = np.sqrt(H**2 + W**2)
    cm_distance = torch.sqrt((y_cm_obs - y_cm_pred) ** 2 + (x_cm_obs - x_cm_pred) ** 2)
    L = (cm_distance / max_distance).item()

    # S (Structure) - simplified
    obs_max = obs_masked.max()
    pred_max = pred_masked.max()

    V_obs = R_obs / obs_max.clamp(min=1e-10) if obs_max > 0 else torch.tensor(0.0)
    V_pred = R_pred / pred_max.clamp(min=1e-10) if pred_max > 0 else torch.tensor(0.0)

    if V_obs + V_pred < 1e-10:
        S = 0.0
    else:
        S = ((V_pred - V_obs) / (0.5 * (V_pred + V_obs))).item()

    return {
        "SAL_S": S,
        "SAL_A": A.item() if isinstance(A, torch.Tensor) else A,
        "SAL_L": L,
    }


# =============================================================================
# Combined spatial metrics - FAST version with fewer neighborhoods
# =============================================================================
def _spatial_metrics_fast(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.1,
    neighborhoods: list[int] | None = None,
) -> dict:
    """
    Fast spatial metrics: FSS at 2-3 scales + SAL.
    """
    if neighborhoods is None:
        # Default: just 3 scales (1=pixel, 5=small, 17=medium)
        neighborhoods = [1, 5, 17]

    results = {}

    # FSS at selected scales only
    for n in neighborhoods:
        results[f"FSS_{n}"] = _fss_fast(y_true, y_pred, threshold, n)

    # SAL (fast version on mean field)
    sal = _sal_fast(y_true, y_pred, threshold)
    results.update(sal)

    return results


# =============================================================================
# Main evaluation function - FAST
# =============================================================================


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    loss_fn=None,
    wet_thr: float = 0.1,
    event_thr: float = 0.1,
    heavy_thr: float = 1.0,
    lt_aware: bool = False,
    max_batches: int | None = None,
    coarsen: bool = False,
    coarsen_mode: str = "avg",
    upsample_input: bool = False,
    upsample_shape: tuple | None = None,
    upsample_mode: str = "bilinear",
    # Spatial metrics
    compute_spatial: bool = True,
    fss_neighborhoods: list[int] | None = None,  # Default: [1, 5, 17]
):
    """
    Evaluate model with point-wise and spatial metrics (fast version).
    """
    if upsample_input and upsample_shape is None:
        raise ValueError("upsample_shape must be provided when upsample_input=True")

    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []

    if max_batches is None:
        bs = getattr(dataloader, "batch_size", None)
        ds_len = len(dataloader.dataset)
        if bs is not None:
            max_batches = math.ceil(ds_len / bs)
        else:
            max_batches = 100

    for ib, batch in enumerate(dataloader):
        if ib >= max_batches:
            break

        if lt_aware:
            img_clean_val, img_lr_val, lead, time_id = batch
            img_clean_val = img_clean_val.to(device)
            img_lr_val = img_lr_val.to(device)
            lead = lead.to(device)

            if upsample_input:
                img_lr_val = _upsample_input(
                    img_lr_val, upsample_shape, mode=upsample_mode
                )

            B, _, H, W = img_lr_val.shape
            c_hr = model.img_out_channels
            x0 = torch.zeros((B, c_hr, H, W), device=device)
            y_pred = model(x0, img_lr_val, lead_time_label=lead)
        else:
            img_clean_val, img_lr_val, *_ = batch if len(batch) == 2 else batch[:2]
            img_clean_val = img_clean_val.to(device)
            img_lr_val = img_lr_val.to(device)

            if upsample_input:
                img_lr_val = _upsample_input(
                    img_lr_val, upsample_shape, mode=upsample_mode
                )

            B, _, H, W = img_lr_val.shape
            c_hr = model.img_out_channels
            x0 = torch.zeros((B, c_hr, H, W), device=device)
            y_pred = model(x0, img_lr_val)

        if coarsen:
            if img_clean_val.dim() == 4:
                target_shape = (img_clean_val.size(2), img_clean_val.size(3))
            else:
                target_shape = (img_clean_val.size(1), img_clean_val.size(2))
            y_pred = _coarsen_to_target(y_pred, target_shape, mode=coarsen_mode)

        if loss_fn is not None:
            loss = loss_fn(y_pred, img_clean_val)
            total_loss += loss.item() * B
            total_samples += B

        all_preds.append(y_pred.detach().cpu())
        all_targets.append(img_clean_val.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    avg_loss = None
    if loss_fn is not None and total_samples > 0:
        avg_loss = total_loss / total_samples

    # Point-wise metrics
    me = _masked_errors(all_targets, all_preds, wet_thr=wet_thr, heavy_thr=heavy_thr)
    ev = _event_stats(all_targets, all_preds, thr=event_thr)

    metrics = {
        k: (v.item() if isinstance(v, torch.Tensor) else v)
        for k, v in {**me, **ev}.items()
    }

    # Spatial metrics (fast)
    if compute_spatial:
        spatial = _spatial_metrics_fast(
            all_targets,
            all_preds,
            threshold=event_thr,
            neighborhoods=fss_neighborhoods,
        )
        metrics.update(spatial)

    return avg_loss, metrics, all_preds, all_targets


# =============================================================================
# Evaluation by lead time - FAST
# =============================================================================
@torch.no_grad()
def evaluate_by_lead_time(
    model,
    dataloader,
    device,
    loss_fn=None,
    wet_thr: float = 0.1,
    event_thr: float = 0.1,
    heavy_thr: float = 1.0,
    lt_aware: bool = True,
    max_batches: int | None = None,
    coarsen: bool = False,
    coarsen_mode: str = "avg",
    upsample_input: bool = False,
    upsample_shape: tuple | None = None,
    upsample_mode: str = "bilinear",
    # Spatial metrics
    compute_spatial: bool = True,
    fss_neighborhoods: list[int] | None = None,  # Default: [1, 5, 17]
):
    """
    Evaluate model by lead time with fast spatial metrics.
    """
    if upsample_input and upsample_shape is None:
        raise ValueError("upsample_shape must be provided when upsample_input=True")

    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_leads = []

    if max_batches is None:
        bs = getattr(dataloader, "batch_size", None)
        ds_len = len(dataloader.dataset)
        if bs is not None:
            max_batches = math.ceil(ds_len / bs)
        else:
            max_batches = 100

    for ib, batch in enumerate(dataloader):
        if ib >= max_batches:
            break

        if lt_aware:
            img_clean_val, img_lr_val, lead, time_id = batch
            img_clean_val = img_clean_val.to(device)
            img_lr_val = img_lr_val.to(device)
            lead = lead.to(device)

            if lead.dim() == 2 and lead.size(1) == 1:
                lead = lead[:, 0]

            if upsample_input:
                img_lr_val = _upsample_input(
                    img_lr_val, upsample_shape, mode=upsample_mode
                )

            B, _, H, W = img_lr_val.shape
            c_hr = model.img_out_channels
            x0 = torch.zeros((B, c_hr, H, W), device=device)
            y_pred = model(x0, img_lr_val, lead_time_label=lead)

            all_leads.append(lead.detach().cpu())
        else:
            img_clean_val, img_lr_val, *_ = batch if len(batch) == 2 else batch[:2]
            img_clean_val = img_clean_val.to(device)
            img_lr_val = img_lr_val.to(device)

            if upsample_input:
                img_lr_val = _upsample_input(
                    img_lr_val, upsample_shape, mode=upsample_mode
                )

            B, _, H, W = img_lr_val.shape
            c_hr = model.img_out_channels
            x0 = torch.zeros((B, c_hr, H, W), device=device)
            y_pred = model(x0, img_lr_val)

        if coarsen:
            if img_clean_val.dim() == 4:
                target_shape = (img_clean_val.size(2), img_clean_val.size(3))
            else:
                target_shape = (img_clean_val.size(1), img_clean_val.size(2))
            y_pred = _coarsen_to_target(y_pred, target_shape, mode=coarsen_mode)

        if loss_fn is not None:
            loss = loss_fn(y_pred, img_clean_val)
            total_loss += loss.item() * B
            total_samples += B

        all_preds.append(y_pred.detach().cpu())
        all_targets.append(img_clean_val.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if lt_aware and len(all_leads) > 0:
        all_leads = torch.cat(all_leads, dim=0)
    else:
        all_leads = None

    avg_loss = None
    if loss_fn is not None and total_samples > 0:
        avg_loss = total_loss / total_samples

    # Global metrics
    me_global = _masked_errors(
        all_targets, all_preds, wet_thr=wet_thr, heavy_thr=heavy_thr
    )
    ev_global = _event_stats(all_targets, all_preds, thr=event_thr)

    global_metrics = {
        k: (v.item() if isinstance(v, torch.Tensor) else v)
        for k, v in {**me_global, **ev_global}.items()
    }

    if compute_spatial:
        spatial_global = _spatial_metrics_fast(
            all_targets,
            all_preds,
            threshold=event_thr,
            neighborhoods=fss_neighborhoods,
        )
        global_metrics.update(spatial_global)

    # Per-lead metrics
    per_lead_metrics = {}

    if lt_aware and all_leads is not None:
        unique_leads = torch.unique(all_leads)

        for lead_val in unique_leads:
            mask = all_leads == lead_val
            n = int(mask.sum().item())
            if n == 0:
                continue

            preds_lt = all_preds[mask]
            targets_lt = all_targets[mask]

            me_lt = _masked_errors(
                targets_lt, preds_lt, wet_thr=wet_thr, heavy_thr=heavy_thr
            )
            ev_lt = _event_stats(targets_lt, preds_lt, thr=event_thr)

            metrics_lt = {
                k: (v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in {**me_lt, **ev_lt}.items()
            }

            if compute_spatial:
                spatial_lt = _spatial_metrics_fast(
                    targets_lt,
                    preds_lt,
                    threshold=event_thr,
                    neighborhoods=fss_neighborhoods,
                )
                metrics_lt.update(spatial_lt)

            metrics_lt["N_samples"] = n
            per_lead_metrics[int(lead_val.item())] = metrics_lt

    return avg_loss, global_metrics, per_lead_metrics, all_preds, all_targets, all_leads


# =============================================================================
# Wrapper functions
# =============================================================================
def EvaluateModel(
    model,
    train_loader,
    val_loader,
    lt_aware=True,
    coarsen=False,
    coarsen_mode="avg",
    upsample_input=False,
    upsample_shape=None,
    upsample_mode="bilinear",
    compute_spatial=True,
    fss_neighborhoods=None,
):
    """Evaluate model on train and validation sets (fast version)."""
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_fn = torch.nn.MSELoss()

    train_loss, train_metrics, train_preds, train_targets = evaluate(
        model,
        train_loader,
        device,
        loss_fn=loss_fn,
        wet_thr=0.1,
        event_thr=0.1,
        heavy_thr=1.0,
        lt_aware=lt_aware,
        coarsen=coarsen,
        coarsen_mode=coarsen_mode,
        upsample_input=upsample_input,
        upsample_shape=upsample_shape,
        upsample_mode=upsample_mode,
        compute_spatial=compute_spatial,
        fss_neighborhoods=fss_neighborhoods,
    )
    val_loss, val_metrics, val_preds, val_targets = evaluate(
        model,
        val_loader,
        device,
        loss_fn=loss_fn,
        wet_thr=0.1,
        event_thr=0.1,
        heavy_thr=1.0,
        lt_aware=lt_aware,
        coarsen=coarsen,
        coarsen_mode=coarsen_mode,
        upsample_input=upsample_input,
        upsample_shape=upsample_shape,
        upsample_mode=upsample_mode,
        compute_spatial=compute_spatial,
        fss_neighborhoods=fss_neighborhoods,
    )

    train_metrics["Loss"] = train_loss
    val_metrics["Loss"] = val_loss

    df_metrics = pd.DataFrame([train_metrics, val_metrics], index=["Train", "Val"]).T
    print(df_metrics.round(3))

    return (
        df_metrics,
        train_loss,
        train_metrics,
        train_preds,
        train_targets,
        val_loss,
        val_metrics,
        val_preds,
        val_targets,
    )


def EvaluateModelPerLead(
    model,
    train_loader,
    val_loader,
    lt_aware=True,
    coarsen=False,
    coarsen_mode="avg",
    upsample_input=False,
    upsample_shape=None,
    upsample_mode="bilinear",
    compute_spatial=True,
    fss_neighborhoods=None,
):
    """Evaluate model with per-lead-time breakdown (fast version)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_fn = torch.nn.MSELoss()
    results_model = {}

    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        print(f"\n{'=' * 50}")
        print(f"Evaluating {split_name.upper()}")
        print(f"{'=' * 50}")

        avg_loss, global_metrics, per_lead_metrics, preds, targets, leads = (
            evaluate_by_lead_time(
                model,
                loader,
                device,
                loss_fn=loss_fn,
                wet_thr=0.1,
                event_thr=0.1,
                heavy_thr=1.0,
                lt_aware=lt_aware,
                coarsen=coarsen,
                coarsen_mode=coarsen_mode,
                upsample_input=upsample_input,
                upsample_shape=upsample_shape,
                upsample_mode=upsample_mode,
                compute_spatial=compute_spatial,
                fss_neighborhoods=fss_neighborhoods,
            )
        )

        results_model[split_name] = {
            "avg_loss": avg_loss,
            "global": global_metrics,
            "per_lead": per_lead_metrics,
            "preds": preds,
            "targets": targets,
            "leads": leads,
        }

        # Compact summary
        if avg_loss is not None:
            print(f"Loss: {avg_loss:.4f}")

        print(
            f"\nGlobal: MSE={global_metrics['MSE_all']:.4f}, "
            f"CSI={global_metrics['CSI']:.4f}, "
            f"FSS_1={global_metrics.get('FSS_1', np.nan):.4f}, "
            f"SAL_L={global_metrics.get('SAL_L', np.nan):.4f}"
        )

        if per_lead_metrics:
            print(f"\nPer-lead (showing MSE, CSI):")
            for lt in sorted(per_lead_metrics.keys()):
                m = per_lead_metrics[lt]
                hours = (lt + 1) * 6
                print(
                    f"  Lead {lt:2d} ({hours:3d}h): MSE={m['MSE_all']:.4f}, CSI={m['CSI']:.4f}"
                )

    return results_model



def baseline_per_lead_metrics(
    target_TLHW: np.ndarray,
    pred_TLHW: np.ndarray,
    lead_levels: np.ndarray | None = None,
    wet_thr: float = 0.1,
    event_thr: float = 0.1,
    heavy_thr: float = 1.0,
    compute_spatial: bool = True,
    fss_neighborhoods: list[int] | None = None,
):
    """
    Compute per-lead metrics for baselines, including spatial metrics.

    Inputs:
      target_TLHW: (T, L, H, W)
      pred_TLHW:   (T, L, H, W)
      lead_levels: array of lead values of length L (e.g. [0, 1, ..., 23]).
                   If None, will use 0..L-1.
      wet_thr: threshold for wet pixels
      event_thr: threshold for event detection (also used for FSS/SAL)
      heavy_thr: threshold for heavy precipitation
      compute_spatial: if True, compute FSS and SAL metrics
      fss_neighborhoods: list of neighborhood sizes for FSS (default: [1, 5, 17])

    Returns:
      per_lead: dict[lead_value] -> metrics dict
    """
    assert target_TLHW.shape == pred_TLHW.shape
    T, L, H, W = target_TLHW.shape

    if lead_levels is None:
        lead_levels = np.arange(L)
    assert len(lead_levels) == L

    if fss_neighborhoods is None:
        fss_neighborhoods = [1, 5, 17]

    per_lead = {}

    for li in range(L):
        lead_val = int(lead_levels[li])

        # slice that lead: (T, H, W) → (T, 1, H, W)
        y_true_li = torch.from_numpy(target_TLHW[:, li]).float().unsqueeze(1)
        y_pred_li = torch.from_numpy(pred_TLHW[:, li]).float().unsqueeze(1)

        # Point-wise metrics
        me_li = _masked_errors(
            y_true_li, y_pred_li, wet_thr=wet_thr, heavy_thr=heavy_thr
        )
        ev_li = _event_stats(y_true_li, y_pred_li, thr=event_thr)

        metrics_li = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in {**me_li, **ev_li}.items()
        }

        # Spatial metrics
        if compute_spatial:
            spatial_li = _spatial_metrics_fast(
                y_true_li,
                y_pred_li,
                threshold=event_thr,
                neighborhoods=fss_neighborhoods,
            )
            metrics_li.update(spatial_li)

        metrics_li["N_samples"] = y_true_li.shape[0]  # T

        per_lead[lead_val] = metrics_li

    return per_lead