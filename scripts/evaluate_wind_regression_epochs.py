#!/usr/bin/env python3
"""Compare wind-regression checkpoints on a uniform 2023 validation sample."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from physicsnemo import Module
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.wind_helpers import (
    save_metrics_json,
    uniform_random_sample_indices,
)
from train_wind_regression import _dataset, _model, _validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name", default="config_training_wind_regression.yaml"
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=_REPO_ROOT
        / "outputs/wind_regression/checkpoints/WindForecastUNet.best.mdlus",
    )
    parser.add_argument(
        "--last-checkpoint",
        type=Path,
        default=_REPO_ROOT / "outputs/wind_regression/checkpoints/last.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT
        / "outputs/wind_regression/uniform_2023_checkpoint_comparison.json",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of uniformly random validation samples; 0 evaluates all",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BF16 autocast on CUDA",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override, for example paths.raw_data_root=/data/uv10",
    )
    return parser


def _compose_config(config_name: str, overrides: list[str]):
    with initialize_config_dir(
        config_dir=str((_REPO_ROOT / "conf").resolve()), version_base="1.2"
    ):
        return compose(config_name=config_name, overrides=overrides)


def _loader(
    dataset, batch_size: int, num_workers: int, device: torch.device
) -> DataLoader:
    workers = max(int(num_workers), 0)
    kwargs: Dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=max(int(batch_size), 1),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )


def _recorded_epoch(
    metrics_path: Path, checkpoint: Optional[Dict[str, Any]]
) -> Optional[int]:
    if checkpoint is not None and "epoch" in checkpoint:
        return int(checkpoint["epoch"]) + 1
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("epoch") is not None:
            return int(payload["epoch"])
    return None


def main() -> None:
    args = _parser().parse_args()
    cfg = _compose_config(args.config_name, args.override)
    for path in (args.best_checkpoint, args.last_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp = bool(args.amp) and device.type == "cuda"

    validation_dataset = _dataset(cfg.validation.dataset)
    indices = uniform_random_sample_indices(
        len(validation_dataset), args.samples, args.seed
    )
    if len(indices) == 0:
        raise RuntimeError("The validation dataset is empty")
    validation = Subset(validation_dataset, indices.tolist())
    validation_loader = _loader(validation, args.batch_size, args.num_workers, device)
    times = validation_dataset.time()
    selected_times = [times[int(index)] for index in indices]

    checkpoint_payload = torch.load(
        args.last_checkpoint, map_location="cpu", weights_only=False
    )
    if "model" not in checkpoint_payload:
        raise KeyError(f"{args.last_checkpoint} does not contain a model state")

    best_model = Module.from_checkpoint(str(args.best_checkpoint), {})
    best_model = best_model.eval().requires_grad_(False).to(device)
    best_mse, best_metrics, best_per_lead = _validation(
        best_model, validation_loader, device, validation_dataset, amp
    )
    del best_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    last_model = _model(cfg.model, validation_dataset, use_fp16=False, amp=amp)
    last_model.load_state_dict(checkpoint_payload["model"])
    last_model = last_model.eval().requires_grad_(False).to(device)
    last_mse, last_metrics, last_per_lead = _validation(
        last_model, validation_loader, device, validation_dataset, amp
    )

    best_metrics_path = args.best_checkpoint.parent.parent / "validation_metrics.json"
    best_epoch = _recorded_epoch(best_metrics_path, None)
    last_epoch = _recorded_epoch(Path("__missing__"), checkpoint_payload)
    comparison_name = "last" if last_mse < best_mse else "best"
    payload = {
        "validation": {
            "years": [int(year) for year in cfg.validation.dataset.years],
            "strategy": "uniform_random_without_replacement",
            "available_samples": len(validation_dataset),
            "sample_count": len(indices),
            "seed": int(args.seed),
            "first_initial_time": selected_times[0].isoformat(),
            "last_initial_time": selected_times[-1].isoformat(),
            "samples_by_month": {
                str(month): count
                for month, count in sorted(
                    Counter(timestamp.month for timestamp in selected_times).items()
                )
            },
        },
        "checkpoints": {
            "best": {
                "path": str(args.best_checkpoint.resolve()),
                "recorded_epoch": best_epoch,
                "val_mse": best_mse,
                "overall": best_metrics,
                "per_lead": best_per_lead,
            },
            "last": {
                "path": str(args.last_checkpoint.resolve()),
                "recorded_epoch": last_epoch,
                "val_mse": last_mse,
                "overall": last_metrics,
                "per_lead": last_per_lead,
            },
        },
        "comparison": {
            "lower_val_mse": comparison_name,
            "last_minus_best_val_mse": last_mse - best_mse,
            "last_minus_best_vector_rmse": (
                last_metrics["vector_rmse"] - best_metrics["vector_rmse"]
            ),
        },
    }
    save_metrics_json(args.output, payload)
    print(
        f"best: epoch={best_epoch} val_mse={best_mse:.6f} "
        f"vector_rmse={best_metrics['vector_rmse']:.6f}"
    )
    print(
        f"last: epoch={last_epoch} val_mse={last_mse:.6f} "
        f"vector_rmse={last_metrics['vector_rmse']:.6f}"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
