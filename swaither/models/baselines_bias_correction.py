# baselines_precip.py

import numpy as np
import torch
from typing import Dict, Any, Sequence, Optional, Tuple

try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    Ridge = None
    RandomForestRegressor = None


# ----------------------------------------------------------------------
# Utils: flatten / unflatten, and array making function from the dataset
# ----------------------------------------------------------------------

def flatten_spatiotemporal(
    feats: np.ndarray,
    target: np.ndarray,
    mask_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten (T, L, H, W, C) features and (T, L, H, W) target into (N, C), (N,)
    Optionally apply a boolean mask_valid of shape (T, L, H, W).
    """
    assert feats.ndim == 5  # (T, L, H, W, C)
    assert target.ndim == 4  # (T, L, H, W)
    T, L, H, W, C = feats.shape

    X = feats.reshape(T * L * H * W, C)
    y = target.reshape(T * L * H * W)

    if mask_valid is not None:
        mask_flat = mask_valid.reshape(T * L * H * W)
        X = X[mask_flat]
        y = y[mask_flat]

    # drop NaNs
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[good], y[good]


def unflatten_to_grid(
    y_pred_flat: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """
    Reshape 1D y_pred_flat back to shape of template (T, L, H, W).
    Assumes you predicted on *all* pixels in rasterized order.
    """
    return y_pred_flat.reshape(template.shape)



def build_baseline_spatiotemporal_from_dataset(
    base_dataset,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    precip_out_channel: int = 0,
    dtype: np.dtype = np.float32,
) -> Dict[str, Any]:
    """
    Build *spatio-temporal* arrays for baselines directly from your base_dataset.

    Assumes base_dataset[i] returns:
        (img_clean, img_lr, lead, time)

      where:
        img_clean: torch.Tensor (C_out, H, W)
        img_lr   : torch.Tensor (C_in,  H, W)
        lead     : scalar tensor (e.g. (1,) or (1,1))
        time     : scalar tensor (e.g. (1,) or (1,1))

    Returns
    -------
    dict with:
      - feats_train:  (T_train, L, H, W, C_in)
      - target_train:(T_train, L, H, W)
      - feats_val:    (T_val,   L, H, W, C_in)
      - target_val:   (T_val,   L, H, W)
      - lead_levels:  (L,) sorted unique lead labels (shared train/val)
      - time_levels_train: (T_train,)
      - time_levels_val:   (T_val,)
      - meta: {
            "H", "W", "C_in", "C_out",
            "precip_out_channel",
            "train_idx", "val_idx",
        }
    """

    # --- Infer shapes & flags from one sample -------------------------------
    H, W = base_dataset.image_shape()
    C_in = len(base_dataset.input_channels())
    C_out = len(base_dataset.output_channels())

    sample0 = base_dataset[train_idx[0]]
    n_items = len(sample0)
    has_lead = n_items >= 3
    has_time = n_items >= 4

    if not (has_lead and has_time):
        raise ValueError(
            "build_baseline_spatiotemporal_from_dataset expects "
            "base_dataset[i] to return (img_clean, img_lr, lead, time)."
        )

    img_clean0, img_lr0 = sample0[0], sample0[1]
    assert img_clean0.shape[1:] == (H, W), "img_clean shape != image_shape()"
    assert img_lr0.shape[1:] == (H, W), "img_lr shape != image_shape()"
    assert img_lr0.shape[0] == C_in, "img_lr channels != input_channels() length"
    assert img_clean0.shape[0] == C_out, "img_clean channels != output_channels() length"

    # --- Collect flat (N, C, H, W) + lead/time first -----------------------

    def collect_flat(idx_list: Sequence[int]):
        N = len(idx_list)
        X_flat = np.empty((N, C_in, H, W), dtype=dtype)
        y_flat = np.empty((N, H, W),       dtype=dtype)
        lead_arr = np.empty(N, dtype=np.int64)
        time_arr = np.empty(N, dtype=np.int64)

        for k, idx in enumerate(idx_list):
            sample = base_dataset[idx]
            img_clean, img_lr = sample[0], sample[1]
            lead_tensor = sample[2]
            time_tensor = sample[3]

            X_flat[k] = img_lr.detach().cpu().numpy()
            y_flat[k] = img_clean[precip_out_channel].detach().cpu().numpy()
            lead_arr[k] = int(lead_tensor.item())
            time_arr[k] = int(time_tensor.item())

        return X_flat, y_flat, lead_arr, time_arr

    X_train_flat, y_train_flat, lead_train, time_train = collect_flat(train_idx)
    X_val_flat,   y_val_flat,   lead_val,   time_val   = collect_flat(val_idx)

    # --- Build (T, L, H, W, C) from flat arrays ----------------------------

    def to_TLHW(
        X_flat: np.ndarray,   # (N, C_in, H, W)
        y_flat: np.ndarray,   # (N, H, W)
        lead: np.ndarray,     # (N,)
        time: np.ndarray,     # (N,)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N, C, H_, W_ = X_flat.shape
        assert H_ == H and W_ == W

        time_levels = np.unique(time)
        lead_levels = np.unique(lead)
        T = len(time_levels)
        L = len(lead_levels)

        time_to_idx = {t: i for i, t in enumerate(time_levels)}
        lead_to_idx = {l: j for j, l in enumerate(lead_levels)}

        # NOTE: feats is (T, L, H, W, C)  -> channels-last
        feats = np.empty((T, L, H, W, C), dtype=dtype)
        target = np.empty((T, L, H, W),   dtype=dtype)
        seen = np.zeros((T, L), dtype=bool)

        for n in range(N):
            ti = time_to_idx[time[n]]
            li = lead_to_idx[lead[n]]
            if seen[ti, li]:
                raise ValueError(
                    f"Duplicate sample for time={time[n]}, lead={lead[n]} "
                    f"(T_idx={ti}, L_idx={li}) in TLHW building."
                )
            seen[ti, li] = True

            # X_flat[n] is (C, H, W) → move channel to last: (H, W, C)
            feats[ti, li] = np.moveaxis(X_flat[n], 0, -1)
            target[ti, li] = y_flat[n]   # already (H, W)

        if not np.all(seen):
            missing = np.argwhere(~seen)
            print("[WARN] missing (time, lead) combinations in TLHW:", missing[:10])

        return feats, target, time_levels, lead_levels

    feats_train, target_train, time_levels_train, lead_levels_train = to_TLHW(
        X_train_flat, y_train_flat, lead_train, time_train
    )
    feats_val, target_val, time_levels_val, lead_levels_val = to_TLHW(
        X_val_flat, y_val_flat, lead_val, time_val
    )

    # check that the set of leads is the same in train & val
    if not np.array_equal(lead_levels_train, lead_levels_val):
        raise ValueError(
            f"Train/val lead sets differ:\n"
            f"train={lead_levels_train}\nval={lead_levels_val}"
        )
    lead_levels = lead_levels_train

    meta: Dict[str, Any] = {
        "H": H,
        "W": W,
        "C_in": C_in,
        "C_out": C_out,
        "precip_out_channel": precip_out_channel,
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
    }

    return {
        "feats_train": feats_train,         # (T_train, L, H, W, C_in)
        "target_train": target_train,       # (T_train, L, H, W)
        "feats_val": feats_val,             # (T_val,   L, H, W, C_in)
        "target_val": target_val,           # (T_val,   L, H, W)
        "lead_levels": lead_levels,         # (L,)
        "time_levels_train": time_levels_train,
        "time_levels_val": time_levels_val,
        "meta": meta,
    }

# ------------------------------------------------------------------
# B0 – Raw AIFS precip
# ------------------------------------------------------------------

class BaselineB0_Raw:
    """
    B0: Just returns raw model precipitation.
    No training needed.
    """

    def fit(self, aifs_precip_train: np.ndarray, obs_precip_train: np.ndarray):
        # nothing to learn
        return self

    def predict(self, aifs_precip: np.ndarray) -> np.ndarray:
        # identity
        return np.array(aifs_precip, copy=True)


# ------------------------------------------------------------------
# B1 – Affine global
# ------------------------------------------------------------------

class BaselineB1_AffineGlobal:
    """
    B1: Single global affine correction:
         y_t,l,h,w ≈ a * x_t,l,h,w + b

    Same slope & intercept for all leads and all grid cells.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.slope: float | None = None
        self.intercept: float | None = None

    def fit(
        self,
        aifs_precip_train: np.ndarray,  # (T, L, H, W)
        obs_precip_train: np.ndarray,   # (T, L, H, W)
    ):
        x = np.asarray(aifs_precip_train, dtype=np.float64)
        y = np.asarray(obs_precip_train,  dtype=np.float64)

        # flatten everything
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        mask = np.isfinite(x_flat) & np.isfinite(y_flat)
        x_use = x_flat[mask]
        y_use = y_flat[mask]

        if x_use.size < 2:
            # degenerate fallback: identity
            self.slope = 1.0
            self.intercept = 0.0
            return self

        mean_x = x_use.mean()
        mean_y = y_use.mean()

        x_c = x_use - mean_x
        y_c = y_use - mean_y

        var_x  = (x_c * x_c).mean()
        cov_xy = (x_c * y_c).mean()

        if var_x < self.eps:
            self.slope = 1.0
            self.intercept = 0.0
        else:
            self.slope = cov_xy / var_x
            self.intercept = mean_y - self.slope * mean_x

        return self

    def predict(self, aifs_precip: np.ndarray) -> np.ndarray:
        assert self.slope is not None, "Call fit() first."
        x = np.asarray(aifs_precip, dtype=np.float64)
        y_hat = self.slope * x + self.intercept
        return y_hat.astype(np.float32)


# ------------------------------------------------------------------
# B2 – Affine per leag, global
# ------------------------------------------------------------------

class BaselineB2_AffinePerLeadGlobal:
    """
    B2: Per-lead global affine correction:

      For each lead ℓ:
        y_t,ℓ,h,w ≈ a[ℓ] * x_t,ℓ,h,w + b[ℓ]

    a[ℓ], b[ℓ] are shared across all spatial grid cells for that lead.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.slope: np.ndarray | None = None   # shape (L,)
        self.intercept: np.ndarray | None = None  # shape (L,)

    def fit(
        self,
        aifs_precip_train: np.ndarray,  # (T, L, H, W)
        obs_precip_train: np.ndarray,   # (T, L, H, W)
    ):
        x = np.asarray(aifs_precip_train, dtype=np.float64)
        y = np.asarray(obs_precip_train,  dtype=np.float64)

        T, L, H, W = x.shape

        slope     = np.zeros(L, dtype=np.float64)
        intercept = np.zeros(L, dtype=np.float64)

        for ell in range(L):
            x_ell = x[:, ell, :, :].reshape(-1)
            y_ell = y[:, ell, :, :].reshape(-1)

            mask  = np.isfinite(x_ell) & np.isfinite(y_ell)
            x_use = x_ell[mask]
            y_use = y_ell[mask]

            if x_use.size < 2:
                # fallback: identity for this lead
                slope[ell] = 1.0
                intercept[ell] = 0.0
                continue

            mean_x = x_use.mean()
            mean_y = y_use.mean()

            x_c = x_use - mean_x
            y_c = y_use - mean_y

            var_x  = (x_c * x_c).mean()
            cov_xy = (x_c * y_c).mean()

            if var_x < self.eps:
                slope[ell] = 1.0
                intercept[ell] = 0.0
            else:
                slope[ell] = cov_xy / var_x
                intercept[ell] = mean_y - slope[ell] * mean_x

        self.slope     = slope
        self.intercept = intercept
        return self

    def predict(self, aifs_precip: np.ndarray) -> np.ndarray:
        assert self.slope is not None, "Call fit() first."
        x = np.asarray(aifs_precip, dtype=np.float64)   # (T, L, H, W)
        # broadcast slope/intercept: (1,L,1,1)
        y_hat = self.slope[None, :, None, None] * x + self.intercept[None, :, None, None]
        return y_hat.astype(np.float32)

# ------------------------------------------------------------------
# B3 – Per-grid-cell affine correction: y ≈ a*x + b
# ------------------------------------------------------------------

class BaselineB3_AffinePerGrid:
    """
    B3: Per-grid-cell univariate affine correction on AIFS precip.

    For each (lead, y, x) we fit:
        y_t ≈ a(l,y,x) * x_t + b(l,y,x)

    using closed-form linear regression over time.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.slope = None
        self.intercept = None

    def fit(
        self,
        aifs_precip_train: np.ndarray,  # (T, L, H, W)
        obs_precip_train: np.ndarray,   # (T, L, H, W)
    ):
        x = np.array(aifs_precip_train, dtype=np.float64)
        y = np.array(obs_precip_train, dtype=np.float64)

        # Time mean
        mean_x = np.nanmean(x, axis=0)  # (L, H, W)
        mean_y = np.nanmean(y, axis=0)

        # Centered
        x_c = x - mean_x  # broadcasts on axis 0
        y_c = y - mean_y

        # Var and cov along time
        var_x = np.nanmean(x_c * x_c, axis=0)
        cov_xy = np.nanmean(x_c * y_c, axis=0)

        slope = np.zeros_like(var_x)
        mask = var_x > self.eps
        slope[mask] = cov_xy[mask] / var_x[mask]

        # intercept
        intercept = mean_y - slope * mean_x

        self.slope = slope
        self.intercept = intercept
        return self

    def predict(self, aifs_precip: np.ndarray) -> np.ndarray:
        assert self.slope is not None, "Call fit() first."
        x = np.array(aifs_precip, dtype=np.float64)
        # broadcast slope/intercept across time
        y_hat = self.slope[None, ...] * x + self.intercept[None, ...]
        return y_hat.astype(np.float32)


# ------------------------------------------------------------------
# B4 – Per-grid-cell multivariate linear regression (all channels)
# ------------------------------------------------------------------

class BaselineB4_MultiLinearPerGrid:
    """
    B4: At each (lead, y, x) we fit multivariate linear regression:

        y_t ≈ w0 + w1*x1_t + ... + wC*xC_t

    where x_t is the vector of features at that grid cell.
    """

    def __init__(self, l2_reg: float = 1e-4):
        self.l2_reg = l2_reg
        self.weights = None  # shape (L, H, W, C+1)

    def fit(
        self,
        feats_train: np.ndarray,       # (T, L, H, W, C)
        obs_precip_train: np.ndarray,  # (T, L, H, W)
    ):
        T, L, H, W, C = feats_train.shape
        self.weights = np.zeros((L, H, W, C + 1), dtype=np.float64)

        X = feats_train.astype(np.float64)
        y = obs_precip_train.astype(np.float64)

        I = np.eye(C + 1, dtype=np.float64)
        I[0, 0] = 0.0  # don't penalize bias

        for l in range(L):
            for i in range(H):
                for j in range(W):
                    Xi = X[:, l, i, j, :]   # (T, C)
                    yi = y[:, l, i, j]      # (T,)

                    # mask out NaNs
                    mask = np.isfinite(yi) & np.all(np.isfinite(Xi), axis=1)
                    if mask.sum() < C + 1:
                        # not enough points, keep zeros
                        continue

                    X_design = np.concatenate(
                        [np.ones((mask.sum(), 1)), Xi[mask]], axis=1
                    )  # (n, C+1)
                    y_vec = yi[mask]  # (n,)

                    # Ridge-regularized normal equations
                    XtX = X_design.T @ X_design + self.l2_reg * I
                    Xty = X_design.T @ y_vec

                    try:
                        w = np.linalg.solve(XtX, Xty)  # (C+1,)
                    except np.linalg.LinAlgError:
                        w, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)

                    self.weights[l, i, j, :] = w

        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """
        Predict on feats: (T, L, H, W, C) using per-grid weights.
        Returns: (T, L, H, W)
        """
        assert self.weights is not None, "Call fit() first."
        X = feats.astype(np.float64)  # (T, L, H, W, C)
        T, L, H, W, C = X.shape
        w = self.weights              # (L, H, W, C+1)

        # bias(l,h,w) and coef(l,h,w,c)
        bias = w[..., 0]        # (L, H, W)
        coef = w[..., 1:]       # (L, H, W, C)

        # Broadcast coef over time: (1, L, H, W, C) * (T, L, H, W, C)
        # Sum over channels → (T, L, H, W)
        y_hat = bias[None, ...] + np.sum(coef[None, ...] * X, axis=-1)

        return y_hat.astype(np.float32)

# ------------------------------------------------------------------
# B5 – Per-grid-cell quantile mapping (AIFS precip only)
# ------------------------------------------------------------------

class BaselineB5_QuantileMapping:
    """
    B5: Per-grid-cell quantile mapping.

    For each (lead, y, x) we store:
      - sorted model precip (x_sorted)
      - sorted obs precip (y_sorted)

    Prediction:
      - find quantile of new x in x_sorted
      - return interpolated y at same quantile.
    """

    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self.x_sorted = None  # (L, H, W, n) as object array
        self.y_sorted = None

    def fit(
        self,
        aifs_precip_train: np.ndarray,  # (T, L, H, W)
        obs_precip_train: np.ndarray,   # (T, L, H, W)
    ):
        T, L, H, W = aifs_precip_train.shape
        self.x_sorted = np.empty((L, H, W), dtype=object)
        self.y_sorted = np.empty((L, H, W), dtype=object)

        x = aifs_precip_train.astype(np.float64)
        y = obs_precip_train.astype(np.float64)

        for l in range(L):
            for i in range(H):
                for j in range(W):
                    xi = x[:, l, i, j]
                    yi = y[:, l, i, j]

                    mask = np.isfinite(xi) & np.isfinite(yi)
                    xi = xi[mask]
                    yi = yi[mask]

                    if xi.size < self.min_samples:
                        # fallback: identity mapping
                        self.x_sorted[l, i, j] = np.array([0.0, 1.0])
                        self.y_sorted[l, i, j] = np.array([0.0, 1.0])
                        continue

                    order = np.argsort(xi)
                    xs = xi[order]
                    ys = yi[order]

                    self.x_sorted[l, i, j] = xs
                    self.y_sorted[l, i, j] = ys

        return self

    def _map_cell(self, x_new, xs, ys):
        """
        Map scalar or array x_new through the empirical quantile mapping defined
        by xs (model) and ys (obs).
        """
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        if xs.size <= 1:
            # degenerate, return identity
            return x_new

        # Compute quantile of x_new in xs
        # rank / (N - 1) in [0,1]
        ranks = np.searchsorted(xs, x_new, side="right")
        p = ranks / max(xs.size - 1, 1)

        # Now interpolate ys at quantile p
        # Build a fake "probability axis" for ys
        q = np.linspace(0.0, 1.0, num=ys.size)
        return np.interp(p, q, ys)

    def predict(self, aifs_precip: np.ndarray) -> np.ndarray:
        assert self.x_sorted is not None, "Call fit() first."
        x = aifs_precip.astype(np.float64)
        T, L, H, W = x.shape
        y_hat = np.zeros_like(x)

        for l in range(L):
            for i in range(H):
                for j in range(W):
                    xs = self.x_sorted[l, i, j]
                    ys = self.y_sorted[l, i, j]
                    y_hat[:, l, i, j] = self._map_cell(x[:, l, i, j], xs, ys)

        return y_hat.astype(np.float32)


# ------------------------------------------------------------------
# B7 – Global ridge regression on all pixels
# ------------------------------------------------------------------

class BaselineB7_RidgeGlobal:
    """
    B7: One global Ridge regression on all pixels.

    Features: all channels per pixel (feats: T×L×H×W×C).
    Target: obs_precip: T×L×H×W.
    """

    def __init__(self, alpha: float = 1.0):
        if Ridge is None:
            raise ImportError("scikit-learn is required for BaselineB7_RidgeGlobal.")
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(
        self,
        feats_train: np.ndarray,       # (T, L, H, W, C)
        obs_precip_train: np.ndarray,  # (T, L, H, W)
        mask_valid: Optional[np.ndarray] = None,
    ):
        X, y = flatten_spatiotemporal(feats_train, obs_precip_train, mask_valid)
        self.model.fit(X, y)
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        T, L, H, W, C = feats.shape
        X = feats.reshape(T * L * H * W, C)
        y_pred_flat = self.model.predict(X)
        return y_pred_flat.reshape(T, L, H, W).astype(np.float32)


# ------------------------------------------------------------------
# B8 – Global random forest on all pixels
# ------------------------------------------------------------------

class BaselineB8_RFGlobal:
    """
    B8: One global Random Forest regression on all pixels.

    Features: all channels per pixel (feats: T×L×H×W×C).
    Target: obs_precip: T×L×H×W.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
    ):
        if RandomForestRegressor is None:
            raise ImportError("scikit-learn is required for BaselineB8_RFGlobal.")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            oob_score=False,
            random_state=0,
        )

    def fit(
        self,
        feats_train: np.ndarray,       # (T, L, H, W, C)
        obs_precip_train: np.ndarray,  # (T, L, H, W)
        mask_valid: Optional[np.ndarray] = None,
    ):
        X, y = flatten_spatiotemporal(feats_train, obs_precip_train, mask_valid)
        self.model.fit(X, y)
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        T, L, H, W, C = feats.shape
        X = feats.reshape(T * L * H * W, C)
        y_pred_flat = self.model.predict(X)
        return y_pred_flat.reshape(T, L, H, W).astype(np.float32)
