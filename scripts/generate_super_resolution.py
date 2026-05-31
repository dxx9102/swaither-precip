# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Dan Assouline and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Originally based on physicsnemo/examples/generative/corrdiff/generate.py.

import os
import contextlib
from functools import partial

import hydra
from omegaconf import OmegaConf, DictConfig

import numpy as np
import netCDF4 as nc
from tqdm.auto import tqdm

import torch
import torch._dynamo
import nvtx
from numpy.random import default_rng

from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SequentialSampler

from hydra.utils import to_absolute_path

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.patching import GridPatching2D
from physicsnemo import Module

from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler
from physicsnemo.utils.corrdiff import (
    get_time_from_range,
    regression_step,
    diffusion_step,
)

from helpers.generate_helpers import get_dataset_and_sampler
from helpers.train_helpers import set_patch_shape
from datasets.dataset import register_dataset

import xarray as xr

# -----------------------------
# denorm + delog 
# -----------------------------
def denorm_and_delog(dataset, y_norm: torch.Tensor) -> torch.Tensor:
    out_min = torch.as_tensor(dataset.output_min, device=y_norm.device, dtype=y_norm.dtype)[:, None, None]
    out_max = torch.as_tensor(dataset.output_max, device=y_norm.device, dtype=y_norm.dtype)[:, None, None]
    out_range = torch.clamp(out_max - out_min, min=1e-6)
    y_log = y_norm * out_range + out_min
    y_lin = torch.expm1(y_log)
    y_lin = torch.clamp(y_lin, min=0.0)
    return y_lin


def _cftime_to_iso(t) -> str:
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}T{t.hour:02d}:{t.minute:02d}:{t.second:02d}"


def extract_all_times_from_low_res_file(low_res_path: str) -> list[str]:
    """
    Read only the 'time' coordinate from the low-res bias-corrected file and return ISO strings.
    Works whether time is datetime64, cftime, or already strings.
    """
    ds = xr.open_dataset(low_res_path, decode_times=True)

    if "time" not in ds.coords:
        raise KeyError(f"No 'time' coordinate found in low_res_file: {low_res_path}")

    t = ds["time"].values  # 1D array typically
    ds.close()

    times_iso: list[str] = []
    for ti in t:
        # already a string (numpy <U.. or object)
        if isinstance(ti, (str, np.str_)):
            times_iso.append(str(ti))
            continue

        # numpy datetime64
        if isinstance(ti, np.datetime64):
            # cast to seconds precision ISO like YYYY-MM-DDTHH:MM:SS
            s = np.datetime_as_string(ti, unit="s")
            # ensure 'T' separator
            if "T" not in s and " " in s:
                s = s.replace(" ", "T")
            times_iso.append(s)
            continue

        # cftime-like object
        if hasattr(ti, "year") and hasattr(ti, "month") and hasattr(ti, "day"):
            times_iso.append(_cftime_to_iso(ti))
            continue

        # fallback
        times_iso.append(str(ti))

    return times_iso

# to apply filters to the automatic times detection (stride: consider every n time, and max number of times)
def apply_time_valves(times: list[str], time_stride: int = 1, max_times: int | None = None) -> list[str]:
    if time_stride is None:
        time_stride = 1
    time_stride = int(time_stride)
    if time_stride < 1:
        raise ValueError(f"time_stride must be >= 1, got {time_stride}")

    out = times[::time_stride]

    if max_times is not None:
        max_times = int(max_times)
        if max_times < 1:
            raise ValueError(f"max_times must be >= 1, got {max_times}")
        out = out[:max_times]

    return out


rng = default_rng()

@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    # -----------------------------
    # Distributed
    # -----------------------------
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    logger = PythonLogger("generate")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    if dist.world_size > 1:
        torch.distributed.barrier()

    # -----------------------------
    # Parse times (requested by user) OR auto from low-res file
    # + valves: time_stride, max_times
    # -----------------------------
    user_provided_times = bool(cfg.generation.times) or bool(cfg.generation.times_range)
    force_all_times = bool(cfg.generation.get("use_all_times_from_low_res", False))

    time_stride = int(cfg.generation.get("time_stride", 1))
    max_times_cfg = cfg.generation.get("max_times", None)
    max_times = None if max_times_cfg in (None, "none", "None") else int(max_times_cfg)

    if force_all_times or not user_provided_times:
        low_res_path = cfg.dataset.low_res_file
        if low_res_path in (None, "???"):
            raise ValueError("To auto-extract times, cfg.dataset.low_res_file must be set (not '???').")

        times_req = extract_all_times_from_low_res_file(low_res_path)

        # apply valves
        times_req = apply_time_valves(times_req, time_stride=time_stride, max_times=max_times)

        if dist.rank == 0:
            logger0.info(
                f"[ALL] Auto-selected {len(times_req)} times from low_res_file "
                f"(time_stride={time_stride}, max_times={max_times})"
            )
    else:
        if cfg.generation.times_range and cfg.generation.times:
            raise ValueError("Either times_range or times must be provided, but not both")
        if cfg.generation.times_range:
            times_req = list(get_time_from_range(cfg.generation.times_range))
        else:
            times_req = list(cfg.generation.times)

        # apply valves even for explicit time lists (useful)
        times_req = apply_time_valves(times_req, time_stride=time_stride, max_times=max_times)

        if dist.rank == 0:
            logger0.info(
                f"[ALL] Using {len(times_req)} user-specified times "
                f"(time_stride={time_stride}, max_times={max_times})"
            )


    # -----------------------------
    # Dataset base config
    # -----------------------------
    dataset_cfg_base = OmegaConf.to_container(cfg.dataset, resolve=True)
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    num_horizons_total = int(cfg.generation.get("num_horizons", 24))
    num_ensembles = int(cfg.generation.num_ensembles)
    seed_batch_size = int(cfg.generation.seed_batch_size)

    # -----------------------------
    # Choose horizons
    # -----------------------------
    horizons_cfg = cfg.generation.get("horizons", None)
    if horizons_cfg is None:
        horizons = list(range(num_horizons_total))
    else:
        horizons = [int(h) for h in horizons_cfg]

    # validate
    bad = [h for h in horizons if h < 0 or h >= num_horizons_total]
    if bad:
        raise ValueError(
            f"Some requested horizons are out of bounds for num_horizons={num_horizons_total}: {bad}"
        )

    # (optional) deduplicate but keep order
    seen = set()
    horizons = [h for h in horizons if not (h in seen or seen.add(h))]

    if dist.rank == 0:
        logger0.info(f"[ALL] Generating horizons: {horizons}")

    if seed_batch_size != num_ensembles and dist.rank == 0:
        logger0.info(
            f"[WARN] seed_batch_size={seed_batch_size} != num_ensembles={num_ensembles}. "
            f"This script will only write the first chunk unless you implement chunk accumulation."
        )

    # One progress bar for horizons only (rank 0)
    horizon_iter = horizons
    if dist.rank == 0:
        horizon_iter = tqdm(horizon_iter, desc="Horizons", dynamic_ncols=True)

    # -----------------------------
    # Prepare ONE output file (rank 0)
    # -----------------------------
    f = None
    pred_var = None
    truth_var = None
    input_var = None
    time_var = None
    lead_var = None
    written_truth = None
    written_input = None

    # lead coord stores the *selected* horizon indices
    lead_vals = np.array(horizons, dtype=np.int32)

    base_out = getattr(cfg.generation.io, "output_filename", "ALL_LEADS_samples.nc")
    output_path_abs = os.path.abspath(base_out)
    os.makedirs(os.path.dirname(output_path_abs), exist_ok=True)

    # global mapping from requested ISO->index in time dimension
    time_to_global = {t: i for i, t in enumerate(times_req)}

    # map horizon index -> lead position in output file
    horizon_to_leadpos = {h: i for i, h in enumerate(horizons)}

    # -----------------------------
    # Loop selected horizons
    # -----------------------------
    for hori_idx in horizon_iter:
        lead_pos = horizon_to_leadpos[hori_idx]  # 0..len(horizons)-1

        # Build per-horizon dataset cfg
        dataset_cfg = dict(dataset_cfg_base)
        dataset_cfg["prediction_delta_index"] = hori_idx
        dataset_cfg["horizon_index"] = hori_idx
        dataset_cfg["lead_index"] = hori_idx

        # Create dataset + sampler for THIS horizon
        dataset, sampler = get_dataset_and_sampler(
            dataset_cfg=dataset_cfg,
            times=times_req,
            has_lead_time=True,
            hori_idx=hori_idx,
        )

        if len(sampler) == 0:
            if dist.rank == 0:
                logger0.info(f"[H{hori_idx:02d}] No samples for this horizon. Skipping.")
            continue

        # Debug
        if dist.rank == 0:
            all_times = dataset.time()
            logger0.info(f"[H{hori_idx:02d}] len(dataset)={len(dataset)} sampler.len={len(sampler)}")
            logger0.info(f"[H{hori_idx:02d}] sampler[:10]={sampler[:10]}")
            logger0.info(f"[H{hori_idx:02d}] times at sampler[:5]={ [all_times[i] for i in sampler[:5]] }")

        # ----------------------------------
        # Shapes + patching
        # ----------------------------------
        img_shape = dataset.image_shape()
        img_out_channels = len(dataset.output_channels())

        if cfg.generation.patching:
            patch_shape_x = cfg.generation.patch_shape_x
            patch_shape_y = cfg.generation.patch_shape_y
        else:
            patch_shape_x, patch_shape_y = None, None

        patch_shape = (patch_shape_y, patch_shape_x)
        use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)

        if use_patching:
            patching = GridPatching2D(
                img_shape=img_shape,
                patch_shape=patch_shape,
                boundary_pix=cfg.generation.boundary_pix,
                overlap_pix=cfg.generation.overlap_pix,
            )
        else:
            patching = None

        # ----------------------------------
        # Inference mode
        # ----------------------------------
        if cfg.generation.inference_mode == "regression":
            load_net_reg, load_net_res = True, False
        elif cfg.generation.inference_mode == "diffusion":
            load_net_reg, load_net_res = False, True
        elif cfg.generation.inference_mode == "all":
            load_net_reg, load_net_res = True, True
        else:
            raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

        # ----------------------------------
        # Load nets
        # ----------------------------------
        net_res, net_reg = None, None

        if load_net_res:
            res_ckpt_filename = cfg.generation.io.res_ckpt_filename
            if dist.rank == 0:
                logger0.info(f"[H{hori_idx:02d}] Loading residual net: {res_ckpt_filename}")
            net_res = Module.from_checkpoint(to_absolute_path(res_ckpt_filename))
            net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
            if cfg.generation.perf.force_fp16:
                net_res.use_fp16 = True
            if hasattr(net_res, "amp_mode"):
                net_res.amp_mode = False

        if load_net_reg:
            reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
            if dist.rank == 0:
                logger0.info(f"[H{hori_idx:02d}] Loading reg net: {reg_ckpt_filename}")
            net_reg = Module.from_checkpoint(to_absolute_path(reg_ckpt_filename))
            net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
            if cfg.generation.perf.force_fp16:
                net_reg.use_fp16 = True
            if hasattr(net_reg, "amp_mode"):
                net_reg.amp_mode = False

        if cfg.generation.perf.use_torch_compile:
            torch._dynamo.reset()
            if net_res:
                net_res = torch.compile(net_res, mode="reduce-overhead")

        # ----------------------------------
        # Sampler function
        # ----------------------------------
        if cfg.sampler.type == "deterministic":
            if cfg.generation.hr_mean_conditioning:
                raise NotImplementedError("hr_mean_conditioning not implemented for deterministic sampler")
            sampler_fn = partial(
                deterministic_sampler,
                num_steps=cfg.sampler.num_steps,
                solver=cfg.sampler.solver,
            )
        elif cfg.sampler.type == "stochastic":
            sampler_fn = partial(stochastic_sampler, patching=patching)
        else:
            raise ValueError(f"Unknown sampling method {cfg.sampler.type}")

        # ----------------------------------
        # Build subset loader (deterministic ordering)
        # ----------------------------------
        subset = Subset(dataset, sampler)
        seq_sampler = SequentialSampler(subset)

        data_loader = DataLoader(
            subset,
            sampler=seq_sampler,
            batch_size=1,
            pin_memory=True,
        )

        # Times for THIS horizon subset
        dataset_times = dataset.time()
        times_subset_cftime = [dataset_times[i] for i in sampler]
        times_subset_iso = [_cftime_to_iso(t) for t in times_subset_cftime]

        # ----------------------------------
        # Create ONE netCDF file lazily on first written horizon (rank 0)
        # ----------------------------------
        if dist.rank == 0 and f is None:
            logger0.info(f"[ALL] Writing single file to: {output_path_abs}")
            f = nc.Dataset(output_path_abs, "w", format="NETCDF4")
            f.cfg = str(cfg)

            lat_1d = dataset.latitude()
            lon_1d = dataset.longitude()
            H = len(lat_1d)
            W = len(lon_1d)

            # dims
            f.createDimension("ensemble", num_ensembles)
            f.createDimension("time", len(times_req))
            f.createDimension("lead", len(horizons))  # <-- NEW
            f.createDimension("y", H)
            f.createDimension("x", W)

            # coords
            time_var = f.createVariable("time", str, ("time",))
            time_var[:] = np.array(times_req, dtype=object)

            lead_var = f.createVariable("lead", "i4", ("lead",))
            lead_var[:] = lead_vals  # stores actual horizon indices

            lat_var = f.createVariable("lat", "f4", ("y",))
            lon_var = f.createVariable("lon", "f4", ("x",))
            lat_var[:] = lat_1d.astype(np.float32)
            lon_var[:] = lon_1d.astype(np.float32)

            truth_var = f.createVariable("truth", "f4", ("time", "y", "x"), zlib=True, complevel=2)
            input_var = f.createVariable("input", "f4", ("time", "y", "x"), zlib=True, complevel=2)
            pred_var = f.createVariable("pred", "f4", ("ensemble", "time", "lead", "y", "x"), zlib=True, complevel=2)

            written_truth = np.zeros((len(times_req),), dtype=bool)
            written_input = np.zeros((len(times_req),), dtype=bool)

        # ----------------------------------
        # Per-horizon batches bar (rank 0)
        # ----------------------------------
        pbar_batches = None
        if dist.rank == 0:
            pbar_batches = tqdm(total=len(subset), desc=f"H{hori_idx:02d} samples", dynamic_ncols=True)

        # ----------------------------------
        # Generation function
        # ----------------------------------
        rank_batches = None

        def generate_fn(image_lr: torch.Tensor, lead_time_label: torch.Tensor | None) -> torch.Tensor:
            nonlocal rank_batches
            with nvtx.annotate("generate_fn", color="green"):
                img_lr = image_lr.to(memory_format=torch.channels_last)

                image_reg = None
                if net_reg:
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=img_lr,
                        latents_shape=(seed_batch_size, img_out_channels, img_shape[0], img_shape[1]),
                        lead_time_label=lead_time_label,
                    )

                image_res = None
                if net_res:
                    mean_hr = image_reg[0:1] if (image_reg is not None and cfg.generation.hr_mean_conditioning) else None
                    image_res = diffusion_step(
                        net=net_res,
                        sampler_fn=sampler_fn,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=img_lr.expand(seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last),
                        rank=dist.rank,
                        device=device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                    )

                if cfg.generation.inference_mode == "regression":
                    return image_reg
                if cfg.generation.inference_mode == "diffusion":
                    return image_res

                # "all"
                B_reg = image_reg.shape[0]
                B_res = image_res.shape[0]
                if B_res % B_reg != 0:
                    raise RuntimeError(f"Cannot tile regression ({B_reg}) into residual ({B_res})")
                repeats = B_res // B_reg
                return image_reg.repeat(repeats, 1, 1, 1) + image_res

        # ----------------------------------
        # Inference loop
        # ----------------------------------
        torch_cuda_profiler = (
            torch.cuda.profiler.profile() if torch.cuda.is_available() else contextlib.nullcontext()
        )
        torch_nvtx_profiler = (
            torch.autograd.profiler.emit_nvtx() if torch.cuda.is_available() else contextlib.nullcontext()
        )

        with torch_cuda_profiler:
            with torch_nvtx_profiler:
                for local_idx, batch in enumerate(data_loader):
                    if len(batch) == 2:
                        image_tar, image_lr = batch
                        lead_time_label = None
                    elif len(batch) == 3:
                        image_tar, image_lr, lead_time_label = batch
                        lead_time_label = lead_time_label.to(dist.device).contiguous()
                    else:
                        image_tar, image_lr, lead_time_label, _time_id = batch
                        lead_time_label = lead_time_label.to(dist.device).contiguous()

                    # seeds per sample
                    seeds = rng.integers(low=0, high=2**31 - 1, size=num_ensembles, dtype=np.int64).tolist()
                    num_batches = ((len(seeds) - 1) // (seed_batch_size * dist.world_size) + 1) * dist.world_size
                    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
                    rank_batches = all_batches[dist.rank :: dist.world_size]

                    image_lr = image_lr.to(device=device).to(torch.float32).to(memory_format=torch.channels_last)
                    image_tar = image_tar.to(device=device).to(torch.float32)

                    if torch.isnan(image_lr).all():
                        if dist.rank == 0 and pbar_batches is not None:
                            pbar_batches.update(1)
                        continue

                    image_out = generate_fn(image_lr=image_lr, lead_time_label=lead_time_label)

                    if dist.rank == 0:
                        t_iso = times_subset_iso[local_idx]
                        if t_iso not in time_to_global:
                            raise KeyError(f"Time {t_iso} not in requested times: {times_req}")
                        t_global = time_to_global[t_iso]

                        out_w = denorm_and_delog(dataset, image_out).cpu().numpy()  # (B, C, H, W)
                        tar_w = denorm_and_delog(dataset, image_tar).cpu().numpy()  # (1, C, H, W)
                        lr_w = denorm_and_delog(dataset, image_lr).cpu().numpy()    # (1, Cin, H, W)

                        # write truth/input once per time
                        if not written_truth[t_global]:
                            truth_var[t_global, :, :] = tar_w[0, 0, :, :]
                            written_truth[t_global] = True

                        if not written_input[t_global]:
                            input_var[t_global, :, :] = lr_w[0, 0, :, :]
                            written_input[t_global] = True

                        # write pred for this selected horizon at lead_pos
                        if seed_batch_size == num_ensembles:
                            pred_var[:, t_global, lead_pos, :, :] = out_w[:, 0, :, :]
                        else:
                            pred_var[0:seed_batch_size, t_global, lead_pos, :, :] = out_w[:, 0, :, :]

                        if pbar_batches is not None:
                            pbar_batches.update(1)

        if dist.rank == 0 and pbar_batches is not None:
            pbar_batches.close()

        del net_reg, net_res
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dist.rank == 0 and f is not None:
        f.close()

    logger0.info("Generation Completed.")


if __name__ == "__main__":
    main()
