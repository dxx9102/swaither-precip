#!/usr/bin/env python3
"""Train residual diffusion on top of a frozen wind regression U-Net."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from physicsnemo import Module
from physicsnemo.metrics.diffusion import ResidualLoss
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.wind_helpers import configure_storage

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


def _loader(dataset, cfg: DictConfig, device: torch.device, shuffle: bool) -> DataLoader:
    workers = int(cfg.num_workers)
    kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        drop_last=False,
        **kwargs,
    )


def _patches(cfg: DictConfig, dataset: WindForecastDataset) -> WindRandomPatching2D | None:
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


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_training_wind_diffusion")
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
    validation_dataset = WindDiffusionDataset(
        validation_base, leads=validation_leads
    )
    max_train = int(cfg.training.get("max_samples", 0))
    max_val = int(cfg.validation.get("max_samples", 0))
    train_data = (
        train_dataset
        if max_train <= 0 or max_train >= len(train_dataset)
        else Subset(train_dataset, range(max_train))
    )
    val_data = (
        validation_dataset
        if max_val <= 0 or max_val >= len(validation_dataset)
        else Subset(validation_dataset, range(max_val))
    )
    train_loader = _loader(train_data, cfg.training, device, shuffle=True)
    val_loader = _loader(val_data, cfg.validation, device, shuffle=False)

    regression_path = Path(str(cfg.training.regression_checkpoint))
    if not regression_path.exists():
        raise FileNotFoundError(f"Regression checkpoint not found: {regression_path}")
    regression_net = Module.from_checkpoint(str(regression_path), {}).eval().requires_grad_(False)
    regression_net = regression_net.to(device)
    adapter = WindRegressionLeadAdapter(
        regression_net,
        forecast_hours=train_base.forecast_horizon_hours,
        wind_components=len(train_base.output_variables),
        conditioning_input_channels=len(train_base.physical_input_names) + train_base.grid_channels,
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
    best_val = float("inf")
    epochs_without_improvement = 0
    resume = Path(str(cfg.training.resume)) if cfg.training.get("resume") else None
    if resume is not None and resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best_val", best_val))
        print(f"Resumed from {resume} at epoch {start_epoch}")
    accumulation = max(int(cfg.training.get("gradient_accumulation_steps", 1)), 1)
    metrics_path = run_dir / "validation_metrics.json"

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        model.train()
        adapter.eval()
        started = time.time()
        running_loss = 0.0
        sample_count = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (y, x, lead) in enumerate(train_loader, start=1):
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
            if step % accumulation == 0 or step == len(train_loader):
                if cfg.training.get("grad_clip") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.training.grad_clip)
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(reduced.detach().cpu()) * y.shape[0]
            sample_count += y.shape[0]

        train_loss = running_loss / max(sample_count, 1)
        val_loss, val_residual_mse = _validation(
            model, val_loader, device, loss_fn, amp
        )
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)
        writer.add_scalar("validation/residual_mse", val_residual_mse, epoch)
        print(
            f"epoch={epoch + 1}/{cfg.training.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} residual_mse={val_residual_mse:.6f} "
            f"elapsed={time.time() - started:.1f}s"
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
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            checkpoint_dir / "last.pt",
        )
        if epochs_without_improvement >= int(cfg.training.early_stopping_patience):
            print(f"Early stopping after {epochs_without_improvement} non-improving epochs")
            break

    writer.close()
    print(f"Best validation diffusion loss={best_val:.6f}; checkpoints are in {checkpoint_dir}")


if __name__ == "__main__":
    main()
