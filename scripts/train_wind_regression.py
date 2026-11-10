#!/usr/bin/env python3
"""Train the one-shot 72-hour high-resolution wind regression U-Net."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

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
from swaither.losses.wind_regression import (
    RegressionLossResult,
    WindRegressionLoss,
    build_wind_regression_loss,
)
from swaither.models import WindForecastUNet
from swaither.wind_helpers import uniform_random_sample_indices, wind_metrics


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


def _augment_labels(
    model: torch.nn.Module, labels: torch.Tensor | None
) -> torch.Tensor | None:
    return labels if int(getattr(model, "augment_dim", 0)) > 0 else None


def _loss_function(
    cfg: DictConfig, dataset: WindForecastDataset, device: torch.device
) -> WindRegressionLoss:
    loss_cfg = cfg.training.get("loss")
    if loss_cfg is None:
        loss_cfg = {
            "terms": {
                "normalized_mse": {
                    "type": "normalized_mse",
                    "weight": 1.0,
                }
            }
        }
    else:
        loss_cfg = OmegaConf.to_container(loss_cfg, resolve=True)
    return build_wind_regression_loss(
        loss_cfg,
        output_min=dataset.output_min,
        output_max=dataset.output_max,
        forecast_hours=dataset.forecast_horizon_hours,
        wind_components=len(dataset.output_channels()),
    ).to(device)


def _accumulate_loss_components(
    sums: Dict[str, Dict[str, float]],
    result: RegressionLossResult,
    weight: int,
) -> None:
    for group_name, values in (("raw", result.raw), ("weighted", result.weighted)):
        destination = sums.setdefault(group_name, {})
        for name, value in values.items():
            destination[name] = (
                destination.get(name, 0.0) + float(value.detach().cpu()) * weight
            )


def _mean_loss_components(
    sums: Mapping[str, Mapping[str, float]], value_count: int
) -> Dict[str, Dict[str, float]]:
    divisor = max(int(value_count), 1)
    return {
        group_name: {name: total / divisor for name, total in values.items()}
        for group_name, values in sums.items()
    }


def _format_loss_terms(components: Mapping[str, float]) -> str:
    return ", ".join(f"{name}={value:.6f}" for name, value in components.items())


def _validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset: WindForecastDataset,
    loss_fn: WindRegressionLoss,
    amp: bool,
) -> Tuple[
    float,
    float,
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    model.eval()
    composite_loss = 0.0
    squared_error = 0.0
    value_count = 0
    loss_sums: Dict[str, Dict[str, float]] = {}
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
            destination[name] = (
                previous_sum + value * weight,
                previous_weight + weight,
            )

    with torch.no_grad():
        for y, x, time_labels in loader:
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            time_labels = time_labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                prediction = model(
                    torch.zeros_like(y),
                    x,
                    augment_labels=_augment_labels(model, time_labels),
                )
            loss_result = loss_fn(prediction, y)
            composite_loss += float(loss_result.total.detach().cpu()) * y.numel()
            _accumulate_loss_components(loss_sums, loss_result, y.numel())
            squared_error += (
                float(torch.mean((prediction.float() - y.float()) ** 2).cpu())
                * y.numel()
            )
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
                        (pred_fields[:, lead_index] - true_fields[:, lead_index]) ** 2
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
    return (
        composite_loss / max(value_count, 1),
        squared_error / max(value_count, 1),
        overall_metrics,
        per_lead,
        _mean_loss_components(loss_sums, value_count),
    )


def _probe_cuda_memory(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: WindRegressionLoss,
    amp: bool,
) -> None:
    if device.type != "cuda":
        return
    y, x, time_labels = next(iter(loader))
    y = y.to(device, non_blocking=True)
    x = x.to(device, non_blocking=True)
    time_labels = time_labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        prediction = model(
            torch.zeros_like(y),
            x,
            augment_labels=_augment_labels(model, time_labels),
        )
        loss = loss_fn(prediction, y).total
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


@hydra.main(
    version_base="1.2",
    config_path="../conf",
    config_name="config_training_wind_regression",
)
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
    validation_seed = int(cfg.validation.get("sample_seed", cfg.training.seed))
    validation_indices = uniform_random_sample_indices(
        len(validation_dataset), max_val, validation_seed
    )
    val_data = Subset(validation_dataset, validation_indices.tolist())
    validation_times = validation_dataset.time()
    print(
        f"[validation] uniformly sampled {len(validation_indices)}/"
        f"{len(validation_dataset)} samples with seed={validation_seed}; "
        f"range={validation_times[int(validation_indices[0])]}.."
        f"{validation_times[int(validation_indices[-1])]}",
        flush=True,
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
    loss_fn = _loss_function(cfg, train_dataset, device)
    checkpoint_metric = str(cfg.training.get("checkpoint_metric", "vector_rmse"))
    if checkpoint_metric not in {"vector_rmse", "validation_loss"}:
        raise ValueError(
            "training.checkpoint_metric must be vector_rmse or validation_loss, "
            f"got {checkpoint_metric!r}"
        )
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
        _probe_cuda_memory(model, train_loader, optimizer, device, loss_fn, amp)

    run_dir = Path(cfg.training.run_dir)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    writer = SummaryWriter(log_dir=str(Path(cfg.training.tensorboard_dir)))

    start_epoch = 0
    best_score = float("inf")
    epochs_without_improvement = 0
    resume = Path(str(cfg.training.resume)) if cfg.training.get("resume") else None
    if resume is not None and resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        saved_metric = state.get("checkpoint_metric")
        if saved_metric == checkpoint_metric:
            best_score = float(
                state.get("best_score", state.get("best_val", best_score))
            )
        else:
            print(
                "Resetting best-checkpoint score because the resumed checkpoint "
                f"used {saved_metric!r}, not {checkpoint_metric!r}"
            )
        print(
            f"Resumed from {resume} at epoch {start_epoch}; "
            f"checkpoint_metric={checkpoint_metric}, best_score={best_score:.6f}"
        )

    accumulation = max(int(cfg.training.get("gradient_accumulation_steps", 1)), 1)
    metrics_path = run_dir / "validation_metrics.json"
    for epoch in range(start_epoch, int(cfg.training.epochs)):
        model.train()
        started = time.time()
        running_loss = 0.0
        train_loss_sums: Dict[str, Dict[str, float]] = {}
        trained_values = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (y, x, time_labels) in enumerate(train_loader, start=1):
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            time_labels = time_labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                prediction = model(
                    torch.zeros_like(y),
                    x,
                    augment_labels=_augment_labels(model, time_labels),
                )
                loss_result = loss_fn(prediction, y)
                loss = loss_result.total
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                if cfg.training.get("grad_clip") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.training.grad_clip)
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu()) * y.numel()
            _accumulate_loss_components(train_loss_sums, loss_result, y.numel())
            trained_values += y.numel()

        train_loss = running_loss / max(trained_values, 1)
        train_loss_components = _mean_loss_components(train_loss_sums, trained_values)
        (
            val_loss,
            val_mse,
            metrics,
            per_lead_metrics,
            val_loss_components,
        ) = _validation(model, val_loader, device, validation_dataset, loss_fn, amp)
        checkpoint_score = (
            float(metrics["vector_rmse"])
            if checkpoint_metric == "vector_rmse"
            else float(val_loss)
        )
        if not np.isfinite(checkpoint_score):
            raise RuntimeError(
                f"Non-finite validation checkpoint score: {checkpoint_score}"
            )
        scheduler.step(checkpoint_score)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)
        writer.add_scalar("validation/mse", val_mse, epoch)
        writer.add_scalar(
            f"validation/checkpoint_{checkpoint_metric}", checkpoint_score, epoch
        )
        for group_name, components in train_loss_components.items():
            for name, value in components.items():
                writer.add_scalar(f"train/loss_{group_name}/{name}", value, epoch)
        for group_name, components in val_loss_components.items():
            for name, value in components.items():
                writer.add_scalar(f"validation/loss_{group_name}/{name}", value, epoch)
        for name, value in metrics.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        for lead_hour, lead_metrics in per_lead_metrics.items():
            for name, value in lead_metrics.items():
                writer.add_scalar(
                    f"validation/lead_{int(lead_hour):02d}/{name}", value, epoch
                )
        print(
            f"epoch={epoch + 1}/{cfg.training.epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_mse={val_mse:.6f} "
            f"vector_rmse={metrics['vector_rmse']:.4f} "
            f"checkpoint_{checkpoint_metric}={checkpoint_score:.6f} "
            f"elapsed={time.time() - started:.1f}s"
        )
        print(
            "[loss raw] train "
            f"{{{_format_loss_terms(train_loss_components.get('raw', {}))}}}; "
            "validation "
            f"{{{_format_loss_terms(val_loss_components.get('raw', {}))}}}",
            flush=True,
        )
        print(
            "[loss weighted] train "
            f"{{{_format_loss_terms(train_loss_components.get('weighted', {}))}}}; "
            "validation "
            f"{{{_format_loss_terms(val_loss_components.get('weighted', {}))}}}",
            flush=True,
        )

        if checkpoint_score < best_score:
            best_score = checkpoint_score
            epochs_without_improvement = 0
            model.save(str(checkpoint_dir / "WindForecastUNet.best.mdlus"))
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_mse": val_mse,
                        "checkpoint_metric": checkpoint_metric,
                        "checkpoint_score": checkpoint_score,
                        "loss_terms": val_loss_components,
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
                "best_score": best_score,
                "best_val": best_score,
                "checkpoint_metric": checkpoint_metric,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            checkpoint_dir / "last.pt",
        )
        if epochs_without_improvement >= int(cfg.training.early_stopping_patience):
            print(
                f"Early stopping after {epochs_without_improvement} non-improving epochs"
            )
            break

    writer.close()
    print(
        f"Best validation {checkpoint_metric}={best_score:.6f}; "
        f"checkpoints are in {checkpoint_dir}"
    )


if __name__ == "__main__":
    main()
