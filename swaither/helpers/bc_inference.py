"""
Inference pipelines for bias-correction models (STEP 1).

Contains:
  - denorm_and_delog
  - RunInferenceWeekly          (multi-lead models)
  - RunInferenceWeeklySingleLead (single-lead models)
"""
import torch

def denorm_and_delog(dataset, y_norm: torch.Tensor) -> torch.Tensor:
    """
    Undo min–max normalization (in log space) and log1p.
    y_norm: (B, C_out, H, W)
    Returns: y_phys in mm, same shape.
    """
    out_min = torch.as_tensor(
        dataset.output_min, device=y_norm.device, dtype=y_norm.dtype
    )[:, None, None]
    out_max = torch.as_tensor(
        dataset.output_max, device=y_norm.device, dtype=y_norm.dtype
    )[:, None, None]
    out_range = torch.clamp(out_max - out_min, min=1e-6)

    y_log = y_norm * out_range + out_min
    y_lin = torch.expm1(y_log)
    y_lin = torch.clamp(y_lin, min=0.0)
    return y_lin


# ================================================================
# Main inference pipeline for STEP 1 (bias correction)
# ================================================================
def RunInferenceWeekly(
    cfg_path: str,
    checkpoint_dir: str,
    mdl_path: str,
    out_path: str,
    *,
    batch_size: int = 256,
    data_loader_kwargs: dict | None = None,
    precip_out_channel: int = 0,
):
    """
    Runs inference on the weekly-split VAL set and saves bias-corrected LR precip as:
      (time, prediction_delta, lat, lon)

    Assumes dataset __getitem__ returns (img_clean, img_lr, lead, time_id).
    """
    if data_loader_kwargs is None:
        data_loader_kwargs = {}

    USE_X0 = True  # your current model signature needs x0

    # ------------------------------------------------------------
    # Init distributed manager
    # ------------------------------------------------------------
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # ------------------------------------------------------------
    # Load Hydra config (same as training STEP1)
    # ------------------------------------------------------------
    cfg = OmegaConf.load(cfg_path)
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    # Force crop to match diffusion dataset (your requirement)
    cfg.dataset.crop_mult = 16
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    # ------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------
    register_dataset(cfg.dataset.type)
    full_dataset, _ = init_dataset_from_config(
        dataset_cfg,
        {},  # dataloader cfg unused here
        None,  # batch_size=None (dataset-level)
        cfg.training.hp.seed,
    )

    print("[INFO] Full dataset length (samples):", len(full_dataset))
    print("[INFO] Image shape:", full_dataset.image_shape())
    print("[INFO] Num leads:", len(full_dataset.lead_values))

    # ------------------------------------------------------------
    # Weekly split (same logic as training)
    # ------------------------------------------------------------
    N = len(full_dataset)
    times_list = full_dataset.time()  # length N, one timestamp per sample (time,lead)
    if len(times_list) != N:
        print(
            f"[WARN] full_dataset.time() length {len(times_list)} != len(full_dataset) {N}"
        )

    first_time = times_list[0]
    L_infer = 1
    for i in range(1, N):
        if times_list[i] == first_time:
            L_infer += 1
        else:
            break

    if N % L_infer != 0:
        print(f"[WARN] N={N} not multiple of L={L_infer}; groups may be irregular")

    T_unique = N // L_infer
    print(f"[INFO] Inferred num leads L       = {L_infer}")
    print(f"[INFO] Inferred num unique times  = {T_unique}")

    train_idx, val_idx = [], []
    for g in range(T_unique):
        start = g * L_infer
        end = (g + 1) * L_infer
        if (g % 7) < 6:
            train_idx.extend(range(start, end))
        else:
            val_idx.extend(range(start, end))

    print(f"[INFO] Train samples (6-on): {len(train_idx)}")
    print(f"[INFO] Val samples   (1-off): {len(val_idx)}")
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("6-on/1-off split produced empty train or val set.")

    val_ds = Subset(full_dataset, val_idx)

    # ------------------------------------------------------------
    # Finite, deterministic VAL loader (NO InfiniteSampler)
    # ------------------------------------------------------------
    if dist.world_size > 1:
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=dist.world_size,
            rank=dist.rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        **data_loader_kwargs,
    )

    # ------------------------------------------------------------
    # Load trained bias-correction model (.mdlus)
    # ------------------------------------------------------------
    mdl_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".mdlus")]
    print("[INFO] Available model files:", mdl_files)
    print("[INFO] Loading model:", mdl_path)

    model = Module.from_checkpoint(mdl_path, {})
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    if hasattr(model, "amp_mode"):
        model.amp_mode = False

    # ------------------------------------------------------------
    # Prepare output TLHW for VAL ONLY (NaN init)
    # ------------------------------------------------------------
    # Determine which low-time indices exist in the VAL subset
    subset_indices = val_ds.indices  # indices into full_dataset
    sample_triplets = [full_dataset.samples_idx[i] for i in subset_indices]
    ti_lows = [t for (t, _, _) in sample_triplets]
    unique_t_indices = sorted(set(ti_lows))

    T_val = len(unique_t_indices)
    lead_vals = list(full_dataset.lead_values)
    L = len(lead_vals)
    H, W = full_dataset.image_shape()

    print(f"[INFO] Val unique times: {T_val} | leads: {L} | HxW: {H}x{W}")

    tindex_map = {ti: k for k, ti in enumerate(unique_t_indices)}
    pred_bc_val = np.full((T_val, L, H, W), np.nan, dtype=np.float32)

    # Robust lead mapping (only used if lead in batch is not an integer index)
    lead_to_li = {lv: i for i, lv in enumerate(lead_vals)}

    # ------------------------------------------------------------
    # Inference loop (indexes via time_id + lead from batch)
    # ------------------------------------------------------------
    with torch.no_grad():
        if val_sampler is not None:
            val_sampler.set_epoch(0)

        for batch_idx, batch in enumerate(val_loader):
            if len(batch) != 4:
                raise RuntimeError(
                    f"Expected batch=(img_clean, img_lr, lead, time_id) but got len={len(batch)}"
                )

            img_clean_val, img_lr_val, lead_t, time_id = batch
            img_clean_val = img_clean_val.to(device)
            img_lr_val = img_lr_val.to(device, dtype=torch.float32).to(
                memory_format=torch.channels_last
            )
            lead_t = lead_t.to(device)
            time_id = time_id.to(device)

            # squeeze (B,1) -> (B,)
            if lead_t.dim() == 2 and lead_t.size(1) == 1:
                lead_flat = lead_t[:, 0]
            else:
                lead_flat = lead_t
            if time_id.dim() == 2 and time_id.size(1) == 1:
                time_flat = time_id[:, 0]
            else:
                time_flat = time_id

            B, _, H0, W0 = img_lr_val.shape
            c_hr = (
                model.img_out_channels
                if hasattr(model, "img_out_channels")
                else img_clean_val.size(1)
            )

            if USE_X0:
                x0 = torch.zeros((B, c_hr, H0, W0), device=device)
                y_pred_norm = model(x0, img_lr_val, lead_time_label=lead_flat)
            else:
                y_pred_norm = model(img_lr_val, lead_time_label=lead_flat)

            if isinstance(y_pred_norm, dict):
                for key in ["output", "y", "pred"]:
                    if key in y_pred_norm:
                        y_pred_norm = y_pred_norm[key]
                        break

            if y_pred_norm.dim() == 3:
                y_pred_norm = y_pred_norm[:, None, :, :]

            # If multi-channel output, keep precip channel only (usually channel 0)
            if y_pred_norm.size(1) > 1:
                y_pred_norm = y_pred_norm[
                    :, precip_out_channel : precip_out_channel + 1
                ]

            # denorm + delog → mm
            y_pred_lin = denorm_and_delog(full_dataset, y_pred_norm)  # (B,1,H,W)
            y_pred_np = y_pred_lin.squeeze(1).detach().cpu().numpy()  # (B,H,W)

            # Fill TLHW
            for b in range(B):
                ti_low = int(time_flat[b].item())
                if ti_low not in tindex_map:
                    continue
                t_compact = tindex_map[ti_low]

                lv = lead_flat[b].item()

                # If lv is an integer index already:
                if float(lv).is_integer() and 0 <= int(lv) < L:
                    li = int(lv)
                else:
                    # Otherwise try mapping by actual lead value
                    li = lead_to_li.get(lv, None)
                    if li is None:
                        continue

                pred_bc_val[t_compact, li, :, :] = y_pred_np[b]

    filled = np.isfinite(pred_bc_val).mean()
    print(f"[INFO] pred_bc_val filled fraction: {filled * 100:.2f}%")
    print("[INFO] Filled pred_bc_val shape:", pred_bc_val.shape)

    # ------------------------------------------------------------
    # Wrap in xarray.Dataset and save
    # ------------------------------------------------------------
    times_val = [full_dataset.low_times[ti] for ti in unique_t_indices]
    lat_vals = full_dataset.latitude()
    lon_vals = full_dataset.longitude()

    da_bc = xr.DataArray(
        pred_bc_val,
        dims=("time", "prediction_delta", "lat", "lon"),
        coords={
            "time": times_val,
            "prediction_delta": lead_vals,
            "lat": lat_vals,
            "lon": lon_vals,
        },
        name="precip6h_cumulative_capped",
    )
    ds_bc = xr.Dataset({"precip6h_cumulative_capped": da_bc})

    ds_bc.to_netcdf(out_path)
    print("[INFO] Saved bias-corrected LR val precip to:", out_path)


def RunInferenceWeeklySingleLead(
    cfg_path: str,
    checkpoint_dir: str,
    mdl_path: str,
    out_path: str,
    *,
    batch_size: int = 256,
    data_loader_kwargs: dict | None = None,
    precip_out_channel: int = 0,
    add_prediction_delta_dim: bool = False,
    prediction_delta_value=None,  # e.g. np.timedelta64(6, "h") or 0
):
    """
    Single-lead STEP1 bias-correction inference on weekly-split VAL.

    Assumes dataset __getitem__ returns either:
      - (img_clean, img_lr)   [most common for single-lead]
      - (img_clean, img_lr, time_id) or other extra fields (we'll ignore extras)

    IMPORTANT:
      - We DO NOT rely on time_id being returned by the dataset.
      - We recover ti_low from full_dataset.samples_idx[dataset_index].
    """

    if data_loader_kwargs is None:
        data_loader_kwargs = {}

    USE_X0 = True  # keep same convention as your lead-aware run (many PhysicsNeMo models expect x0)

    # ------------------------------------------------------------
    # Init distributed
    # ------------------------------------------------------------
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # ------------------------------------------------------------
    # Load config and dataset
    # ------------------------------------------------------------
    cfg = OmegaConf.load(cfg_path)
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    # Keep your crop constraint
    cfg.dataset.crop_mult = 16
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    register_dataset(cfg.dataset.type)
    full_dataset, _ = init_dataset_from_config(
        dataset_cfg,
        {},  # dataloader cfg unused here
        None,  # batch_size=None
        getattr(cfg.training.hp, "seed", 0) if "training" in cfg else 0,
    )

    N = len(full_dataset)
    times_list = full_dataset.time()
    if len(times_list) != N:
        print(
            f"[WARN] full_dataset.time() length {len(times_list)} != len(full_dataset) {N}"
        )

    print("[INFO] Full dataset length:", N)
    print("[INFO] Image shape:", full_dataset.image_shape())

    # ------------------------------------------------------------
    # Infer L (should be 1 for single-lead, but keep generic)
    # ------------------------------------------------------------
    first_time = times_list[0]
    L = 1
    for i in range(1, N):
        if times_list[i] == first_time:
            L += 1
        else:
            break

    if N % L != 0:
        print(f"[WARN] N={N} not multiple of inferred L={L}; groups may be irregular")

    T_unique = N // L
    print(f"[INFO] Inferred L={L} | unique times={T_unique}")

    # ------------------------------------------------------------
    # Weekly split: 6-on train, 1-off val (by time blocks)
    # ------------------------------------------------------------
    train_idx, val_idx = [], []
    for g in range(T_unique):
        start = g * L
        end = (g + 1) * L
        if (g % 7) < 6:
            train_idx.extend(range(start, end))
        else:
            val_idx.extend(range(start, end))

    print(f"[INFO] Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    if len(val_idx) == 0:
        raise RuntimeError("Weekly split produced empty VAL set.")

    val_ds = Subset(full_dataset, val_idx)

    # ------------------------------------------------------------
    # FINITE + deterministic VAL loader (IMPORTANT)
    # ------------------------------------------------------------
    if dist.world_size > 1:
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=dist.world_size,
            rank=dist.rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        **data_loader_kwargs,
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    mdl_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".mdlus")]
    print("[INFO] Available model files:", mdl_files)
    print("[INFO] Loading model:", mdl_path)

    model = Module.from_checkpoint(mdl_path, {})
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    if hasattr(model, "amp_mode"):
        model.amp_mode = False

    # ------------------------------------------------------------
    # Prepare TLHW output on VAL unique times (NaN init)
    # We recover ti_low from full_dataset.samples_idx[dataset_index]
    # ------------------------------------------------------------
    subset_indices = val_ds.indices  # these are indices into full_dataset

    # samples_idx for single-lead often stores (ti_low, ti_high)
    # (or possibly (ti_low, ti_high, li) if something changed) => handle both
    ti_lows = []
    for idx in subset_indices:
        tup = full_dataset.samples_idx[idx]
        ti_low = int(tup[0])
        ti_lows.append(ti_low)

    unique_t_indices = sorted(set(ti_lows))
    T_val = len(unique_t_indices)

    H, W = full_dataset.image_shape()
    pred_bc_val = np.full((T_val, H, W), np.nan, dtype=np.float32)

    tindex_map = {ti: k for k, ti in enumerate(unique_t_indices)}

    print(f"[INFO] VAL unique times: {T_val} | HxW: {H}x{W}")

    # ------------------------------------------------------------
    # Inference loop
    # We must know which dataset indices are in THIS batch to recover ti_low.
    # Easiest: track a running pointer over subset_indices in the same order
    # DataLoader yields samples in Subset order when shuffle=False and sampler fixed.
    # DistributedSampler partitions deterministically; we still need the per-rank order.
    # We'll reconstruct the exact per-rank sequence of Subset positions.
    # ------------------------------------------------------------
    if val_sampler is not None:
        val_sampler.set_epoch(0)
        # DistributedSampler iterates over indices [0..len(val_ds)-1] in deterministic rank order.
        # We can reproduce it by iterating over the sampler itself.
        subset_positions_this_rank = list(iter(val_sampler))  # positions inside val_ds
    else:
        subset_positions_this_rank = list(range(len(val_ds)))

    # Map subset position -> full_dataset index
    full_indices_this_rank = [subset_indices[pos] for pos in subset_positions_this_rank]

    ptr = 0

    with torch.no_grad():
        for batch in val_loader:
            # batch can be (y,x) or (y,x,...) -- we only need first two
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise RuntimeError(
                    f"Unexpected batch structure: {type(batch)} len={len(batch) if isinstance(batch, (list, tuple)) else 'NA'}"
                )

            img_clean_val = batch[0].to(device)
            img_lr_val = (
                batch[1]
                .to(device, dtype=torch.float32)
                .to(memory_format=torch.channels_last)
            )

            B, _, H0, W0 = img_lr_val.shape
            c_hr = (
                model.img_out_channels
                if hasattr(model, "img_out_channels")
                else img_clean_val.size(1)
            )

            if USE_X0:
                x0 = torch.zeros((B, c_hr, H0, W0), device=device)
                # single-lead: no lead_time_label
                y_pred_norm = model(x0, img_lr_val)
            else:
                y_pred_norm = model(img_lr_val)

            # unwrap dict output if needed
            if isinstance(y_pred_norm, dict):
                for key in ["output", "y", "pred"]:
                    if key in y_pred_norm:
                        y_pred_norm = y_pred_norm[key]
                        break

            if y_pred_norm.dim() == 3:
                y_pred_norm = y_pred_norm[:, None, :, :]

            if y_pred_norm.size(1) > 1:
                y_pred_norm = y_pred_norm[
                    :, precip_out_channel : precip_out_channel + 1
                ]

            y_pred_lin = denorm_and_delog(full_dataset, y_pred_norm)  # (B,1,H,W)
            y_pred_np = y_pred_lin.squeeze(1).detach().cpu().numpy()  # (B,H,W)

            # write into compact time index using ti_low from full_dataset.samples_idx
            for b in range(B):
                full_i = full_indices_this_rank[ptr]
                ptr += 1

                ti_low = int(full_dataset.samples_idx[full_i][0])
                t_compact = tindex_map.get(ti_low, None)
                if t_compact is None:
                    continue
                pred_bc_val[t_compact, :, :] = y_pred_np[b]

    filled = np.isfinite(pred_bc_val).mean()
    print(f"[INFO] pred_bc_val filled fraction: {filled * 100:.2f}%")
    print("[INFO] pred_bc_val shape:", pred_bc_val.shape)

    # ------------------------------------------------------------
    # Wrap & save
    # ------------------------------------------------------------
    times_val = [full_dataset.low_times[ti] for ti in unique_t_indices]
    lat_vals = full_dataset.latitude()
    lon_vals = full_dataset.longitude()

    if add_prediction_delta_dim:
        if prediction_delta_value is None:
            prediction_delta_value = 0
        da = xr.DataArray(
            pred_bc_val[:, None, :, :],  # (time, prediction_delta=1, lat, lon)
            dims=("time", "prediction_delta", "lat", "lon"),
            coords={
                "time": times_val,
                "prediction_delta": [prediction_delta_value],
                "lat": lat_vals,
                "lon": lon_vals,
            },
            name="precip6h_cumulative_capped",
        )
    else:
        da = xr.DataArray(
            pred_bc_val,
            dims=("time", "lat", "lon"),
            coords={"time": times_val, "lat": lat_vals, "lon": lon_vals},
            name="precip6h_cumulative_capped",
        )

    ds_bc = xr.Dataset({"precip6h_cumulative_capped": da})
    ds_bc.to_netcdf(out_path)
    print("[INFO] Saved single-lead bias-corrected LR VAL precip to:", out_path)
