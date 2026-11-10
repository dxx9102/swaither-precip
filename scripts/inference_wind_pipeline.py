#!/usr/bin/env python3
"""Run deterministic regression and residual-diffusion wind inference."""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from physicsnemo import Module
from physicsnemo.utils.corrdiff import diffusion_step
from physicsnemo.utils.diffusion import stochastic_sampler

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.wind_helpers import configure_storage

configure_storage()

from swaither.datasets.wind import WindForecastDataset
from swaither.models import (  # noqa: F401 - checkpoint class registry
    WindEDMPrecondSuperResolution,
    WindTimeConditionedModel,
)
from swaither.wind_helpers import (
    WindForecastNetCDFWriter,
    select_initial_indices,
    select_monthly_initial_indices,
)


def _dataset(cfg: DictConfig) -> WindForecastDataset:
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs.pop("type", None)
    return WindForecastDataset(**kwargs)


def _selected_indices(dataset: WindForecastDataset, cfg: DictConfig) -> List[int]:
    selected = select_initial_indices(
        dataset.time(),
        initial_hour=cfg.get("initial_hour"),
        start=cfg.get("initial_time_start"),
        end=cfg.get("initial_time_end"),
        max_initial_times=int(cfg.get("max_initial_times", 0)),
    )
    return selected.tolist()


def _diffusion_indices(dataset: WindForecastDataset, cfg: DictConfig) -> List[int]:
    samples_per_month = int(cfg.get("diffusion_samples_per_month", 0))
    if samples_per_month == 0:
        return _selected_indices(dataset, cfg)
    selected = select_monthly_initial_indices(
        dataset.time(),
        initial_hour=cfg.get("initial_hour"),
        start=cfg.get("initial_time_start"),
        end=cfg.get("initial_time_end"),
        samples_per_month=samples_per_month,
        seed=int(cfg.get("diffusion_sample_seed", 42)),
    )
    return selected.tolist()


def _regression_batch(
    model: torch.nn.Module,
    dataset: WindForecastDataset,
    indices: List[int],
    device: torch.device,
) -> torch.Tensor:
    conditions = np.stack([dataset.condition(index) for index in indices], axis=0)
    x = torch.from_numpy(conditions).to(device=device, dtype=torch.float32)
    latent_channels = dataset.forecast_horizon_hours * len(dataset.output_variables)
    zeros = torch.zeros(
        (len(indices), latent_channels, *dataset.image_shape()),
        device=device,
        dtype=torch.float32,
    )
    augment_labels = None
    if int(getattr(model, "augment_dim", 0)) > 0:
        augment_labels = torch.from_numpy(
            np.stack(
                [dataset.initial_time_features(index) for index in indices], axis=0
            )
        ).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        return model(zeros, x, augment_labels=augment_labels).float()


def _lead_encoding(dataset: WindForecastDataset, lead: int) -> np.ndarray:
    phase = 2.0 * np.pi * float(lead - 1) / float(dataset.forecast_horizon_hours)
    return np.asarray(
        [
            np.full(dataset.image_shape(), np.sin(phase), np.float32),
            np.full(dataset.image_shape(), np.cos(phase), np.float32),
        ]
    )


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m"


def _fill_regression_outputs(
    regression_file: Path,
    dataset: WindForecastDataset,
    model: torch.nn.Module,
    indices: List[int],
    device: torch.device,
    batch_size: int,
) -> None:
    horizons = list(range(1, dataset.forecast_horizon_hours + 1))
    initial_times = [dataset.time()[index] for index in indices]
    chunksizes = (1, 1, *dataset.image_shape())
    with WindForecastNetCDFWriter(
        regression_file,
        initial_times,
        horizons,
        dataset.latitude(),
        dataset.longitude(),
        source="wind-regression",
        chunksizes=chunksizes,
    ) as writer:
        started_at = time.perf_counter()
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_started = time.perf_counter()
            flat = _regression_batch(model, dataset, indices[start:end], device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_elapsed = time.perf_counter() - batch_started

            postprocess_started = time.perf_counter()
            fields = dataset.forecast_fields(flat.cpu().numpy())
            physical = dataset.denormalize_output(fields)
            postprocess_elapsed = time.perf_counter() - postprocess_started

            write_started = time.perf_counter()
            for local_index in range(end - start):
                writer.write_initial(
                    start + local_index,
                    physical[local_index, :, 0],
                    physical[local_index, :, 1],
                )
            write_elapsed = time.perf_counter() - write_started
            elapsed = time.perf_counter() - started_at
            rate = end / max(elapsed, 1e-9)
            remaining = (len(indices) - end) / rate if rate > 0.0 else 0.0
            print(
                f"regression: {end}/{len(indices)} "
                f"infer={_format_duration(inference_elapsed)} "
                f"post={_format_duration(postprocess_elapsed)} "
                f"write={_format_duration(write_elapsed)} "
                f"rate={rate:.2f} initial-times/s "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(remaining)}",
                flush=True,
            )


def _fill_diffusion_outputs(
    diffusion_file: Path,
    dataset: WindForecastDataset,
    regression_model: torch.nn.Module,
    diffusion_model: torch.nn.Module,
    indices: List[int],
    leads: List[int],
    device: torch.device,
    cfg: DictConfig,
) -> None:
    initial_times = [dataset.time()[index] for index in indices]
    ensembles = int(cfg.num_ensembles)
    sampler_steps = int(cfg.sampler_num_steps)
    if sampler_steps < 2:
        raise ValueError(
            "inference.sampler_num_steps must be >= 2; PhysicsNeMo's "
            "stochastic_sampler divides by (num_steps - 1)"
        )
    sampler_fn = partial(
        stochastic_sampler, num_steps=sampler_steps
    )
    rng = np.random.default_rng(int(cfg.seed))
    chunksizes = (1, 1, ensembles, *dataset.image_shape())
    with WindForecastNetCDFWriter(
        diffusion_file,
        initial_times,
        leads,
        dataset.latitude(),
        dataset.longitude(),
        source="wind-residual-diffusion",
        ensemble_size=ensembles,
        chunksizes=chunksizes,
    ) as writer:
        started_at = time.perf_counter()
        for count, index in enumerate(indices, start=1):
            sample_started = time.perf_counter()
            flat_mean = _regression_batch(regression_model, dataset, [index], device)
            mean_fields = dataset.forecast_fields(flat_mean.cpu().numpy())[0]
            base_condition = dataset.condition(index)
            write_elapsed = 0.0
            for lead_position, lead in enumerate(leads):
                mean_lead = torch.from_numpy(mean_fields[lead - 1]).unsqueeze(0).to(device)
                condition = np.concatenate(
                    [base_condition, _lead_encoding(dataset, lead)], axis=0
                )
                condition = torch.from_numpy(condition).unsqueeze(0).to(device)
                seeds = rng.integers(0, 2**31 - 1, size=ensembles, dtype=np.int64)
                sampling_net = diffusion_model
                if int(getattr(diffusion_model, "augment_dim", 0)) > 0:
                    time_labels = torch.from_numpy(
                        dataset.diffusion_time_features(index, lead)
                    ).unsqueeze(0)
                    sampling_net = WindTimeConditionedModel(
                        diffusion_model, time_labels
                    ).to(device)
                residual = diffusion_step(
                    net=sampling_net,
                    sampler_fn=sampler_fn,
                    img_shape=dataset.image_shape(),
                    img_out_channels=len(dataset.output_variables),
                    rank_batches=[seeds],
                    img_lr=condition.expand(ensembles, -1, -1, -1).contiguous(),
                    rank=0,
                    device=device,
                    mean_hr=mean_lead,
                )
                full = mean_lead + residual
                if not torch.isfinite(full).all():
                    raise FloatingPointError(
                        f"Diffusion sampling produced a non-finite field for "
                        f"initial time {initial_times[count - 1]} and lead {lead}"
                    )
                physical = dataset.denormalize_output(full.cpu().numpy())
                if not np.isfinite(physical).all():
                    raise FloatingPointError(
                        f"Wind denormalization produced a non-finite field for "
                        f"initial time {initial_times[count - 1]} and lead {lead}"
                    )
                write_started = time.perf_counter()
                writer.write_lead(
                    count - 1, lead_position, physical[:, 0], physical[:, 1]
                )
                write_elapsed += time.perf_counter() - write_started
            elapsed = time.perf_counter() - started_at
            sample_elapsed = time.perf_counter() - sample_started
            rate = count / max(elapsed, 1e-9)
            remaining = (len(indices) - count) / rate if rate > 0.0 else 0.0
            print(
                f"diffusion: {count}/{len(indices)} "
                f"sample={_format_duration(sample_elapsed)} "
                f"write={_format_duration(write_elapsed)} "
                f"rate={rate:.3f} initial-times/s "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(remaining)}",
                flush=True,
            )


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_inference_wind_pipeline")
def main(cfg: DictConfig) -> None:
    configure_storage()
    if not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; wind inference will run on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = str(cfg.inference.mode)
    if mode not in ("regression", "diffusion", "all"):
        raise ValueError("inference.mode must be regression, diffusion, or all")
    if int(cfg.inference.get("diffusion_samples_per_month", 0)) < 0:
        raise ValueError("inference.diffusion_samples_per_month must be non-negative")
    dataset = _dataset(cfg.dataset)
    regression_indices = _selected_indices(dataset, cfg.inference)
    if not regression_indices:
        raise RuntimeError("No initial times selected for wind inference")
    diffusion_indices = (
        _diffusion_indices(dataset, cfg.inference)
        if mode in ("diffusion", "all")
        else regression_indices
    )
    if not diffusion_indices:
        raise RuntimeError("No initial times selected for diffusion inference")

    regression_model = None
    if mode in ("regression", "all", "diffusion"):
        regression_model = Module.from_checkpoint(cfg.checkpoints.regression, {})
        regression_model = regression_model.eval().requires_grad_(False).to(device)
    if mode in ("regression", "all"):
        _fill_regression_outputs(
            Path(cfg.outputs.regression_file),
            dataset,
            regression_model,
            regression_indices,
            device,
            int(cfg.inference.regression_batch_size),
        )
    if mode in ("diffusion", "all"):
        diffusion_model = Module.from_checkpoint(cfg.checkpoints.diffusion, {})
        diffusion_model = diffusion_model.eval().requires_grad_(False).to(device)
        leads = [int(lead) for lead in cfg.inference.diffusion_leads]
        _fill_diffusion_outputs(
            Path(cfg.outputs.diffusion_file),
            dataset,
            regression_model,
            diffusion_model,
            diffusion_indices,
            leads,
            device,
            cfg.inference,
        )


if __name__ == "__main__":
    main()
