#!/usr/bin/env python3
"""Train residual diffusion on top of a frozen wind regression U-Net."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from physicsnemo import Module
from physicsnemo.metrics.diffusion import ResidualLoss
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.wind_helpers import (
    EpochRandomSubsetSampler,
    configure_storage,
    resume_epoch_and_step,
    uniform_random_sample_indices,
)

configure_storage()

from swaither.datasets.wind import WindDiffusionDataset, WindForecastDataset
from swaither.models import (
    WindEDMPrecondSuperResolution,
    WindRandomPatching2D,
    WindRegressionLeadAdapter,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset(cfg: DictConfig) -> WindForecastDataset:
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs.pop("type", None)
    kwargs.pop("leads", None)
    return WindForecastDataset(**kwargs)


def _loader(
    dataset,
    cfg: DictConfig,
    device: torch.device,
    shuffle: bool,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    workers = int(cfg.num_workers)
    kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    if sampler is None:
        kwargs["shuffle"] = shuffle
    else:
        kwargs["sampler"] = sampler
    return DataLoader(
        dataset, batch_size=int(cfg.batch_size), drop_last=False, **kwargs
    )


def _atomic_torch_save(payload: Dict, path: Path) -> None:
    """Write a checkpoint without leaving a partial ``last.pt`` on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _last_checkpoint_payload(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    next_batch_step: int,
    epoch_complete: bool,
    epoch_loss_sum: float,
    epoch_sample_count: int,
) -> Dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val": float(best_val),
        "next_batch_step": int(next_batch_step),
        "epoch_complete": bool(epoch_complete),
        "epoch_loss_sum": float(epoch_loss_sum),
        "epoch_sample_count": int(epoch_sample_count),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def _patches(
    cfg: DictConfig, dataset: WindForecastDataset
) -> WindRandomPatching2D | None:
    shape_y = int(cfg.get("patch_shape_y", 0))
    shape_x = int(cfg.get("patch_shape_x", 0))
    if shape_y <= 0 or shape_x <= 0:
        return None
    patch_shape = (
        min(shape_y, dataset.image_shape()[0]),
        min(shape_x, dataset.image_shape()[1]),
    )
    if patch_shape == dataset.image_shape():
        return None
    return WindRandomPatching2D(
        img_shape=dataset.image_shape(),
        patch_shape=patch_shape,
        patch_num=int(cfg.get("patch_num", 1)),
    )


def _validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: ResidualLoss,
    amp: bool,
) -> Tuple[float, float]:
    model.eval()
    loss_total = 0.0
    residual_total = 0.0
    count = 0
    with torch.no_grad():
        for y, x, lead in loader:
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            lead = lead.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                loss = loss_fn(
                    net=model,
                    img_clean=y,
                    img_lr=x,
                    lead_time_label=lead,
                )
            loss_total += float(loss.sum().cpu())
            mean = loss_fn.y_mean
            residual_total += float(torch.sum((y.float() - mean.float()) ** 2).cpu())
            count += y.shape[0]
    return loss_total / max(count, 1), residual_total / max(count, 1)


@hydra.main(
    version_base="1.2",
    config_path="../conf",
    config_name="config_training_wind_diffusion",
)
def main(cfg: DictConfig) -> None:
    configure_storage()
    _seed_everything(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA is unavailable; wind diffusion will run on CPU")

    train_leads = list(cfg.dataset.leads) if cfg.dataset.get("leads") else None
    validation_leads = (
        list(cfg.validation.leads) if cfg.validation.get("leads") else None
    )
    train_base = _dataset(cfg.dataset)
    validation_base = _dataset(cfg.validation.dataset)
    train_dataset = WindDiffusionDataset(train_base, leads=train_leads)
    validation_dataset = WindDiffusionDataset(validation_base, leads=validation_leads)
    epoch_samples = int(cfg.training.get("epoch_samples", 0))
    batch_size = int(cfg.training.batch_size)
    accumulation = max(int(cfg.training.get("gradient_accumulation_steps", 1)), 1)
    train_sampler = EpochRandomSubsetSampler(
        len(train_dataset), epoch_samples, int(cfg.training.seed)
    )
    epoch_items = len(train_sampler)
    epoch_steps = (epoch_items + batch_size - 1) // batch_size
    optimizer_steps = (epoch_steps + accumulation - 1) // accumulation
    max_val = int(cfg.validation.get("max_samples", 0))
    validation_seed = int(cfg.validation.get("sample_seed", cfg.training.seed))
    validation_indices = uniform_random_sample_indices(
        len(validation_dataset), max_val, validation_seed
    )
    val_data = Subset(validation_dataset, validation_indices.tolist())
    validation_times = validation_dataset.time()
    print(
        f"[validation] uniformly sampled {len(validation_indices)}/"
        f"{len(validation_dataset)} (initial_time, lead) items with "
        f"seed={validation_seed}; range={validation_times[int(validation_indices[0])]}.."
        f"{validation_times[int(validation_indices[-1])]}",
        flush=True,
    )
    sampled_fraction = 100.0 * epoch_items / len(train_dataset)
    print(
        f"[train] sampling {epoch_items}/{len(train_dataset)} "
        f"(initial_time, lead) items ({sampled_fraction:.4f}%) per epoch; "
        f"batch_size={batch_size} steps={epoch_steps} "
        f"optimizer_steps={optimizer_steps} accumulation={accumulation}",
        flush=True,
    )
    train_loader = _loader(
        train_dataset,
        cfg.training,
        device,
        shuffle=False,
        sampler=train_sampler,
    )
    val_loader = _loader(val_data, cfg.validation, device, shuffle=False)

    regression_path = Path(str(cfg.training.regression_checkpoint))
    if not regression_path.exists():
        raise FileNotFoundError(f"Regression checkpoint not found: {regression_path}")
    regression_net = (
        Module.from_checkpoint(str(regression_path), {}).eval().requires_grad_(False)
    )
    regression_net = regression_net.to(device)
    adapter = WindRegressionLeadAdapter(
        regression_net,
        forecast_hours=train_base.forecast_horizon_hours,
        wind_components=len(train_base.output_variables),
        conditioning_input_channels=len(train_base.physical_input_names)
        + train_base.grid_channels,
        lead_encoding_channels=2,
    ).to(device)

    model_args = OmegaConf.to_container(cfg.model.model_args, resolve=True)
    model_args.update(
        {
            "img_resolution": list(train_base.image_shape()),
            "use_fp16": False,
        }
    )
    conditioning_channels = (
        len(train_base.physical_input_names) + train_base.grid_channels + 2
    )
    model = WindEDMPrecondSuperResolution(
        img_in_channels=conditioning_channels + len(train_base.output_variables),
        img_out_channels=len(train_base.output_variables),
        **model_args,
    ).to(device)
    amp = bool(cfg.training.amp) and device.type == "cuda"
    if hasattr(model, "amp_mode"):
        model.amp_mode = amp
    loss_fn = ResidualLoss(
        regression_net=adapter,
        P_mean=float(cfg.model.P_mean),
        P_std=float(cfg.model.P_std),
        sigma_data=float(cfg.model.sigma_data),
        hr_mean_conditioning=bool(cfg.model.hr_mean_conditioning),
    )
    patching = _patches(cfg.training, train_base)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.training.lr),
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    if bool(cfg.training.get("memory_probe", True)) and device.type == "cuda":
        y, x, lead = next(iter(train_loader))
        y = y.to(device)
        x = x.to(device)
        lead = lead.to(device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss = loss_fn(
                net=model,
                img_clean=y,
                img_lr=x,
                lead_time_label=lead,
                patching=patching,
            )
            loss = loss.sum() / y.shape[0]
        loss.backward()
        torch.cuda.synchronize(device)
        print(
            "[MEMORY] Joint diffusion probe: "
            f"allocated={torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB "
            f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB "
            f"peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
            flush=True,
        )
        optimizer.zero_grad(set_to_none=True)

    run_dir = Path(cfg.training.run_dir)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    writer = SummaryWriter(log_dir=str(Path(cfg.training.tensorboard_dir)))
    start_epoch = 0
    resume_batch_step = 0
    best_val = float("inf")
    epochs_without_improvement = 0
    resumed_epoch_loss_sum = 0.0
    resumed_epoch_sample_count = 0
    resume = Path(str(cfg.training.resume)) if cfg.training.get("resume") else None
    if resume is not None and resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, resume_batch_step = resume_epoch_and_step(state)
        best_val = float(state.get("best_val", best_val))
        resumed_epoch_loss_sum = float(state.get("epoch_loss_sum", 0.0))
        resumed_epoch_sample_count = int(state.get("epoch_sample_count", 0))
        print(
            f"Resumed from {resume} at epoch={start_epoch} "
            f"next_batch_step={resume_batch_step}"
        )
    metrics_path = run_dir / "validation_metrics.json"
    log_every_steps = max(int(cfg.training.get("log_every_steps", 10)), 1)
    checkpoint_every_steps = max(
        int(cfg.training.get("checkpoint_every_steps", 500)), 0
    )
    last_checkpoint_path = checkpoint_dir / "last.pt"

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        model.train()
        adapter.eval()
        train_sampler.set_epoch(epoch)
        epoch_resume_step = resume_batch_step if epoch == start_epoch else 0
        train_sampler.set_start_index(epoch_resume_step * batch_size)
        started = time.perf_counter()
        total_loss_tensor = torch.tensor(
            resumed_epoch_loss_sum if epoch == start_epoch else 0.0,
            dtype=torch.float32,
            device=device,
        )
        window_loss_tensor = torch.zeros((), dtype=torch.float32, device=device)
        sample_count = resumed_epoch_sample_count if epoch == start_epoch else 0
        optimizer.zero_grad(set_to_none=True)
        iterator = iter(train_loader)
        step = epoch_resume_step
        interval_start = time.perf_counter()
        interval_data_seconds = 0.0
        interval_steps = 0
        interval_items = 0
        steps_since_checkpoint = 0
        last_grad_norm: Optional[float] = None
        print(
            f"[epoch] epoch={epoch + 1}/{cfg.training.epochs} "
            f"items={epoch_items} steps={epoch_steps} "
            f"resume_step={epoch_resume_step}",
            flush=True,
        )
        while True:
            data_started = time.perf_counter()
            try:
                y, x, lead = next(iterator)
            except StopIteration:
                break
            interval_data_seconds += time.perf_counter() - data_started
            step += 1
            interval_steps += 1
            steps_since_checkpoint += 1
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            lead = lead.to(device, non_blocking=True)
            if patching is not None:
                patching.reset_patch_indices()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                loss = loss_fn(
                    net=model,
                    img_clean=y,
                    img_lr=x,
                    lead_time_label=lead,
                    patching=patching,
                )
                reduced = loss.sum() / y.shape[0]
            (reduced / accumulation).backward()
            optimizer_step = step % accumulation == 0 or step == epoch_steps
            if optimizer_step:
                if cfg.training.get("grad_clip") is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.training.grad_clip)
                    )
                    last_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss_tensor.add_(reduced.detach().float())
            window_loss_tensor.add_(reduced.detach().float())
            interval_items += y.shape[0]
            sample_count += y.shape[0]

            should_log = step % log_every_steps == 0 or step == epoch_steps
            if should_log:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                now = time.perf_counter()
                duration = max(now - interval_start, 1e-9)
                average_loss = float(window_loss_tensor.item()) / max(interval_items, 1)
                items_per_second = interval_items / duration
                step_ms = 1000.0 * duration / max(interval_steps, 1)
                data_ms = 1000.0 * interval_data_seconds / max(interval_steps, 1)
                compute_ms = max(step_ms - data_ms, 0.0)
                remaining_steps = max(epoch_steps - step, 0)
                eta_seconds = remaining_steps * duration / max(interval_steps, 1)
                global_step = epoch * epoch_steps + step
                grad_norm_text = (
                    "n/a" if last_grad_norm is None else f"{last_grad_norm:.4f}"
                )
                if device.type == "cuda":
                    gpu_text = (
                        f"alloc={torch.cuda.memory_allocated(device) / 1024**3:.2f}GiB "
                        f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.2f}GiB"
                    )
                else:
                    gpu_text = "gpu=n/a"
                print(
                    f"[progress] epoch={epoch + 1}/{cfg.training.epochs} "
                    f"step={step}/{epoch_steps} "
                    f"({100.0 * step / max(epoch_steps, 1):.2f}%) "
                    f"loss={average_loss:.6f} items/s={items_per_second:.3f} "
                    f"data_ms={data_ms:.1f} compute_ms={compute_ms:.1f} "
                    f"grad_norm={grad_norm_text} "
                    f"elapsed={now - started:.1f}s eta={eta_seconds:.1f}s {gpu_text}",
                    flush=True,
                )
                writer.add_scalar("train/step_loss", average_loss, global_step)
                writer.add_scalar(
                    "train/items_per_second", items_per_second, global_step
                )
                writer.add_scalar("train/data_time_ms", data_ms, global_step)
                writer.add_scalar("train/compute_time_ms", compute_ms, global_step)
                writer.add_scalar(
                    "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar(
                    "progress/percent", 100.0 * step / max(epoch_steps, 1), global_step
                )
                if last_grad_norm is not None:
                    writer.add_scalar("train/grad_norm", last_grad_norm, global_step)
                if device.type == "cuda":
                    writer.add_scalar(
                        "train/gpu_allocated_gib",
                        torch.cuda.memory_allocated(device) / 1024**3,
                        global_step,
                    )
                    writer.add_scalar(
                        "train/gpu_reserved_gib",
                        torch.cuda.memory_reserved(device) / 1024**3,
                        global_step,
                    )
                window_loss_tensor.zero_()
                interval_start = time.perf_counter()
                interval_data_seconds = 0.0
                interval_steps = 0
                interval_items = 0

            checkpoint_due = (
                checkpoint_every_steps > 0
                and step < epoch_steps
                and steps_since_checkpoint >= checkpoint_every_steps
                and optimizer_step
            )
            if checkpoint_due:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_loss_sum = float(total_loss_tensor.item())
                _atomic_torch_save(
                    _last_checkpoint_payload(
                        cfg,
                        model,
                        optimizer,
                        epoch,
                        best_val,
                        step,
                        False,
                        epoch_loss_sum,
                        sample_count,
                    ),
                    last_checkpoint_path,
                )
                writer.flush()
                print(
                    f"[checkpoint] saved {last_checkpoint_path} "
                    f"epoch={epoch + 1} step={step}/{epoch_steps}",
                    flush=True,
                )
                steps_since_checkpoint = 0
                if should_log:
                    interval_start = time.perf_counter()

        train_elapsed = time.perf_counter() - started
        train_loss = float(total_loss_tensor.item()) / max(sample_count, 1)
        validation_started = time.perf_counter()
        val_loss, val_residual_mse = _validation(
            model, val_loader, device, loss_fn, amp
        )
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)
        writer.add_scalar("validation/residual_mse", val_residual_mse, epoch)
        print(
            f"epoch={epoch + 1}/{cfg.training.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} residual_mse={val_residual_mse:.6f} "
            f"train_items={sample_count} "
            f"train_items_per_s={sample_count / max(train_elapsed, 1e-9):.3f} "
            f"train_elapsed={train_elapsed:.1f}s "
            f"validation_elapsed={time.perf_counter() - validation_started:.1f}s"
        )
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            model.save(str(checkpoint_dir / "WindEDMPrecondSuperResolution.best.mdlus"))
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "residual_mse": val_residual_mse,
                    },
                    handle,
                    indent=2,
                )
        else:
            epochs_without_improvement += 1
        _atomic_torch_save(
            _last_checkpoint_payload(
                cfg,
                model,
                optimizer,
                epoch,
                best_val,
                0,
                True,
                float(total_loss_tensor.item()),
                sample_count,
            ),
            last_checkpoint_path,
        )
        if epochs_without_improvement >= int(cfg.training.early_stopping_patience):
            print(
                f"Early stopping after {epochs_without_improvement} non-improving epochs"
            )
            break

    writer.close()
    print(
        f"Best validation diffusion loss={best_val:.6f}; checkpoints are in {checkpoint_dir}"
    )


if __name__ == "__main__":
    main()
