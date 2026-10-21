#!/usr/bin/env python3
"""Train the deterministic 6-hour-history, one-hour-ahead wind U-Net."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.wind_helpers import configure_storage

# Must precede physicsnemo imports because LOCAL_CACHE is captured at import.
configure_storage()

from swaither.datasets.wind import WindForecastDataset
from swaither.models import WindForecastUNet
from swaither.wind_helpers import wind_metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset(cfg: DictConfig) -> WindForecastDataset:
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs.pop("type", None)
    return WindForecastDataset(**kwargs)


def _model(cfg: DictConfig, dataset: WindForecastDataset, use_fp16: bool) -> WindForecastUNet:
    unet_kwargs = OmegaConf.to_container(cfg.unet, resolve=True)
    unet_kwargs.update(
        {
            "img_resolution": list(dataset.image_shape()),
            "use_fp16": use_fp16,
            "model_type": "SongUNet",
        }
    )
    return WindForecastUNet(
        img_in_channels=len(dataset.input_channels()) + dataset.grid_channels,
        img_out_channels=2,
        unet_kwargs=unet_kwargs,
    )


def _validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset: WindForecastDataset,
    amp: bool,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    losses = []
    prediction = []
    truth = []
    with torch.no_grad():
        for y, x in loader:
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(torch.zeros_like(y), x)
                loss = torch.mean((pred.float() - y.float()) ** 2)
            losses.append(float(loss.detach().cpu()))
            prediction.append(pred.float().cpu())
            truth.append(y.float().cpu())
    pred_np = dataset.denormalize_output(torch.cat(prediction).numpy())
    truth_np = dataset.denormalize_output(torch.cat(truth).numpy())
    return float(np.mean(losses)), wind_metrics(
        pred_np[:, 0], pred_np[:, 1], truth_np[:, 0], truth_np[:, 1]
    )


def _probe_cuda_memory(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
) -> None:
    """Run one training-sized batch and report peak GPU memory before training."""
    if device.type != "cuda":
        return
    y, x = next(iter(loader))
    y = y.to(device, non_blocking=True)
    x = x.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        prediction = model(torch.zeros_like(y), x)
        loss = torch.mean((prediction.float() - y.float()) ** 2)
    loss.backward()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(
        f"[MEMORY] Step1 one-batch probe: allocated={allocated:.2f} GiB "
        f"reserved={reserved:.2f} GiB peak={peak:.2f} GiB",
        flush=True,
    )
    optimizer.zero_grad(set_to_none=True)


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_training_wind_step1_unet")
def main(cfg: DictConfig) -> None:
    storage = configure_storage()
    seed = int(cfg.training.seed)
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA is unavailable; Step1 will run on CPU")

    train_dataset = _dataset(cfg.dataset)
    validation_dataset = _dataset(cfg.validation.dataset)
    if set(train_dataset.time()) & set(validation_dataset.time()):
        raise RuntimeError("Step1 train and validation timestamps overlap")

    max_train = int(cfg.training.get("max_samples", 0))
    max_val = int(cfg.validation.get("max_samples", 0))
    train_data = train_dataset if max_train <= 0 or max_train >= len(train_dataset) else Subset(train_dataset, range(max_train))
    val_data = validation_dataset if max_val <= 0 or max_val >= len(validation_dataset) else Subset(validation_dataset, range(max_val))

    workers = int(cfg.training.num_workers)
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
    }
    train_loader = DataLoader(
        train_data,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=int(cfg.validation.batch_size),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    amp = bool(cfg.training.amp) and device.type == "cuda"
    model = _model(cfg.model, train_dataset, use_fp16=False).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.training.lr_factor),
        patience=int(cfg.training.lr_patience),
    )
    if bool(cfg.training.get("memory_probe", True)):
        _probe_cuda_memory(model, train_loader, optimizer, device, amp)

    run_dir = Path(cfg.training.run_dir)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        OmegaConf.save(cfg, handle)
    writer = SummaryWriter(log_dir=str(Path(cfg.training.tensorboard_dir)))

    start_epoch = 0
    best_val = float("inf")
    epochs_without_improvement = 0
    resume = Path(cfg.training.resume) if cfg.training.get("resume") else None
    if resume is not None and resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state.get("best_val", best_val))
        print(f"Resumed from {resume} at epoch {start_epoch}")

    metrics_path = run_dir / "validation_metrics.json"
    for epoch in range(start_epoch, int(cfg.training.epochs)):
        model.train()
        started = time.time()
        running = 0.0
        count = 0
        for y, x in train_loader:
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(torch.zeros_like(y), x)
                loss = torch.mean((pred.float() - y.float()) ** 2)
            loss.backward()
            if cfg.training.get("grad_clip") is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
            optimizer.step()
            running += float(loss.detach().cpu())
            count += 1
        train_loss = running / max(count, 1)
        val_loss, metrics = _validation(model, val_loader, device, validation_dataset, amp)
        scheduler.step(val_loss)
        writer.add_scalar("train/mse", train_loss, epoch)
        writer.add_scalar("validation/mse", val_loss, epoch)
        for name, value in metrics.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        print(
            f"epoch={epoch + 1}/{cfg.training.epochs} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} vector_rmse={metrics['vector_rmse']:.4f} "
            f"elapsed={time.time() - started:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            best_path = checkpoint_dir / "WindForecastUNet.best.mdlus"
            model.save(str(best_path))
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump({"epoch": epoch + 1, "val_mse": val_loss, **metrics}, handle, indent=2)
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
    print(f"Best validation MSE={best_val:.6f}; checkpoints are in {checkpoint_dir}")


if __name__ == "__main__":
    main()
