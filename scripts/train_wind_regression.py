#!/usr/bin/env python3
"""Train the one-shot 72-hour high-resolution wind regression U-Net."""

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
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.wind_helpers import configure_storage

# PhysicsNeMo captures LOCAL_CACHE at import time.
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


def _model(
    cfg: DictConfig,
    dataset: WindForecastDataset,
    use_fp16: bool,
    amp: bool,
) -> WindForecastUNet:
    unet_kwargs = OmegaConf.to_container(cfg.unet, resolve=True)
    unet_kwargs.update(
        {
            "img_resolution": list(dataset.image_shape()),
            "use_fp16": use_fp16,
            "amp_mode": amp,
            "model_type": "SongUNet",
        }
    )
    return WindForecastUNet(
        img_in_channels=len(dataset.input_channels()) + dataset.grid_channels,
        forecast_hours=dataset.forecast_horizon_hours,
        wind_components=len(dataset.output_channels()),
        unet_kwargs=unet_kwargs,
    )


def _validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset: WindForecastDataset,
    amp: bool,
) -> Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]:
    model.eval()
    squared_error = 0.0
    value_count = 0
    metric_sums: Dict[str, Tuple[float, int]] = {}
    per_lead_sums: Dict[str, Dict[str, Tuple[float, int]]] = {
        str(lead): {} for lead in range(1, dataset.forecast_horizon_hours + 1)
    }

    def add_metrics(
        destination: Dict[str, Tuple[float, int]],
        metrics: Dict[str, float],
        weight: int,
    ) -> None:
        for name, value in metrics.items():
            previous_sum, previous_weight = destination.get(name, (0.0, 0))
            destination[name] = (previous_sum + value * weight, previous_weight + weight)

    with torch.no_grad():
        for y, x in loader:
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                prediction = model(torch.zeros_like(y), x)
            squared_error += float(torch.mean((prediction.float() - y.float()) ** 2).cpu()) * y.numel()
            value_count += y.numel()

            pred_fields = dataset.forecast_fields(prediction.float().cpu().numpy())
            true_fields = dataset.forecast_fields(y.float().cpu().numpy())
            pred_physical = dataset.denormalize_output(pred_fields)
            true_physical = dataset.denormalize_output(true_fields)
            overall = wind_metrics(
                pred_physical[:, :, 0],
                pred_physical[:, :, 1],
                true_physical[:, :, 0],
                true_physical[:, :, 1],
            )
            add_metrics(metric_sums, overall, int(pred_physical[:, :, 0].size))
            for lead_index in range(dataset.forecast_horizon_hours):
                lead_metrics = wind_metrics(
                    pred_physical[:, lead_index, 0],
                    pred_physical[:, lead_index, 1],
                    true_physical[:, lead_index, 0],
                    true_physical[:, lead_index, 1],
                )
                lead_metrics["normalized_mse"] = float(
                    np.mean(
                        (
                            pred_fields[:, lead_index]
                            - true_fields[:, lead_index]
                        )
                        ** 2
                    )
                )
                add_metrics(
                    per_lead_sums[str(lead_index + 1)],
                    lead_metrics,
                    int(pred_physical[:, lead_index, 0].size),
                )

    overall_metrics = {
        name: total / max(weight, 1) for name, (total, weight) in metric_sums.items()
    }
    per_lead = {
        lead: {
            name: total / max(weight, 1)
            for name, (total, weight) in lead_metrics.items()
        }
        for lead, lead_metrics in per_lead_sums.items()
    }
    return squared_error / max(value_count, 1), overall_metrics, per_lead


def _probe_cuda_memory(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
) -> None:
    if device.type != "cuda":
        return
    y, x = next(iter(loader))
    y = y.to(device, non_blocking=True)
    x = x.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        prediction = model(torch.zeros_like(y), x)
        loss = torch.mean((prediction.float() - y.float()) ** 2)
    loss.backward()
    torch.cuda.synchronize(device)
    print(
        "[MEMORY] Joint regression probe: "
        f"allocated={torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB "
        f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.2f} GiB "
        f"peak={torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
        flush=True,
    )
    optimizer.zero_grad(set_to_none=True)


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_training_wind_regression")
def main(cfg: DictConfig) -> None:
    configure_storage()
    _seed_everything(int(cfg.training.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA is unavailable; wind regression will run on CPU")

    train_dataset = _dataset(cfg.dataset)
    validation_dataset = _dataset(cfg.validation.dataset)
    if set(train_dataset.time()) & set(validation_dataset.time()):
        raise RuntimeError("Wind regression train and validation timestamps overlap")

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
    workers = int(cfg.training.num_workers)
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 2
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
    model = _model(cfg.model, train_dataset, use_fp16=False, amp=amp).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
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
        started = time.time()
        running_squared_error = 0.0
        trained_values = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (y, x) in enumerate(train_loader, start=1):
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                prediction = model(torch.zeros_like(y), x)
                loss = torch.mean((prediction.float() - y.float()) ** 2)
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                if cfg.training.get("grad_clip") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.training.grad_clip)
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_squared_error += float(loss.detach().cpu()) * y.numel()
            trained_values += y.numel()

        train_loss = running_squared_error / max(trained_values, 1)
        val_loss, metrics, per_lead_metrics = _validation(
            model, val_loader, device, validation_dataset, amp
        )
        scheduler.step(val_loss)
        writer.add_scalar("train/mse", train_loss, epoch)
        writer.add_scalar("validation/mse", val_loss, epoch)
        for name, value in metrics.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        for lead_hour, lead_metrics in per_lead_metrics.items():
            for name, value in lead_metrics.items():
                writer.add_scalar(
                    f"validation/lead_{int(lead_hour):02d}/{name}", value, epoch
                )
        print(
            f"epoch={epoch + 1}/{cfg.training.epochs} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} vector_rmse={metrics['vector_rmse']:.4f} "
            f"elapsed={time.time() - started:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            model.save(str(checkpoint_dir / "WindForecastUNet.best.mdlus"))
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "epoch": epoch + 1,
                        "val_mse": val_loss,
                        **metrics,
                        "per_lead": per_lead_metrics,
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
    print(f"Best validation MSE={best_val:.6f}; checkpoints are in {checkpoint_dir}")


if __name__ == "__main__":
    main()
