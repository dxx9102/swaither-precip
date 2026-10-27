#!/usr/bin/env python3
"""Run deterministic regression and residual-diffusion wind inference."""

from __future__ import annotations

import sys
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
from swaither.models import WindEDMPrecondSuperResolution  # noqa: F401 - checkpoint class registry
from swaither.wind_helpers import WindForecastNetCDFWriter


def _dataset(cfg: DictConfig) -> WindForecastDataset:
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs.pop("type", None)
    return WindForecastDataset(**kwargs)


def _selected_indices(dataset: WindForecastDataset, cfg: DictConfig) -> List[int]:
    initial_hour = cfg.get("initial_hour")
    indices = [
        index
        for index, timestamp in enumerate(dataset.time())
        if initial_hour is None or timestamp.hour == int(initial_hour)
    ]
    max_initial_times = int(cfg.get("max_initial_times", 0))
    if max_initial_times > 0:
        indices = indices[:max_initial_times]
    return indices


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
    with torch.inference_mode():
        return model(zeros, x).float()


def _lead_encoding(dataset: WindForecastDataset, lead: int) -> np.ndarray:
    phase = 2.0 * np.pi * float(lead - 1) / float(dataset.forecast_horizon_hours)
    return np.asarray(
        [
            np.full(dataset.image_shape(), np.sin(phase), np.float32),
            np.full(dataset.image_shape(), np.cos(phase), np.float32),
        ]
    )


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
    with WindForecastNetCDFWriter(
        regression_file,
        initial_times,
        horizons,
        dataset.latitude(),
        dataset.longitude(),
        source="wind-regression",
    ) as writer:
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            flat = _regression_batch(model, dataset, indices[start:end], device)
            fields = dataset.forecast_fields(flat.cpu().numpy())
            physical = dataset.denormalize_output(fields)
            for local_index in range(end - start):
                writer.write_initial(
                    start + local_index,
                    physical[local_index, :, 0],
                    physical[local_index, :, 1],
                )
            print(f"regression: {end}/{len(indices)}")


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
    with WindForecastNetCDFWriter(
        diffusion_file,
        initial_times,
        leads,
        dataset.latitude(),
        dataset.longitude(),
        source="wind-residual-diffusion",
        ensemble_size=ensembles,
    ) as writer:
        for count, index in enumerate(indices, start=1):
            flat_mean = _regression_batch(regression_model, dataset, [index], device)
            mean_fields = dataset.forecast_fields(flat_mean.cpu().numpy())[0]
            base_condition = dataset.condition(index)
            for lead_position, lead in enumerate(leads):
                mean_lead = torch.from_numpy(mean_fields[lead - 1]).unsqueeze(0).to(device)
                condition = np.concatenate(
                    [base_condition, _lead_encoding(dataset, lead)], axis=0
                )
                condition = torch.from_numpy(condition).unsqueeze(0).to(device)
                seeds = rng.integers(0, 2**31 - 1, size=ensembles, dtype=np.int64)
                residual = diffusion_step(
                    net=diffusion_model,
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
                writer.write_lead(count - 1, lead_position, physical[:, 0], physical[:, 1])
            print(f"diffusion: {count}/{len(indices)}")


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_inference_wind_pipeline")
def main(cfg: DictConfig) -> None:
    configure_storage()
    if not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; wind inference will run on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _dataset(cfg.dataset)
    indices = _selected_indices(dataset, cfg.inference)
    if not indices:
        raise RuntimeError("No initial times selected for wind inference")
    mode = str(cfg.inference.mode)
    if mode not in ("regression", "diffusion", "all"):
        raise ValueError("inference.mode must be regression, diffusion, or all")

    regression_model = None
    if mode in ("regression", "all", "diffusion"):
        regression_model = Module.from_checkpoint(cfg.checkpoints.regression, {})
        regression_model = regression_model.eval().requires_grad_(False).to(device)
    if mode in ("regression", "all"):
        _fill_regression_outputs(
            Path(cfg.outputs.regression_file),
            dataset,
            regression_model,
            indices,
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
            indices,
            leads,
            device,
            cfg.inference,
        )


if __name__ == "__main__":
    main()
