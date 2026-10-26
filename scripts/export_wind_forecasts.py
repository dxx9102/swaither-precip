#!/usr/bin/env python3
"""Run the frozen Step-1 residual model once per batch and export forecasts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.wind_helpers import configure_storage

configure_storage()

from physicsnemo import Module

import swaither.models as legacy_models
from swaither.datasets.wind import WindForecastDataset
from swaither.wind_helpers import write_forecast_netcdf

sys.modules.setdefault("models", legacy_models)


def _dataset(cfg: DictConfig) -> WindForecastDataset:
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs.pop("type", None)
    return WindForecastDataset(**kwargs)


def _initial_positions(dataset: WindForecastDataset, years: Sequence[int]) -> List[int]:
    selected = set(int(year) for year in years)
    positions = []
    for position in range(dataset.history_hours - 1, len(dataset.low_index.times)):
        history = dataset.low_index.times[position - dataset.history_hours + 1 : position + 1]
        if history[-1].year not in selected:
            continue
        if len({timestamp.year for timestamp in history}) != 1:
            continue
        if any((history[i + 1] - history[i]) != pd.Timedelta(hours=1) for i in range(len(history) - 1)):
            continue
        positions.append(position)
    return positions


def forecast(
    model: torch.nn.Module,
    dataset: WindForecastDataset,
    initial_positions: Sequence[int],
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Predict all configured lead hours with one forward pass per batch."""
    horizon = dataset.forecast_horizon_hours
    outputs = {
        "u": np.empty((len(initial_positions), horizon, *dataset.image_shape()), np.float32),
        "v": np.empty((len(initial_positions), horizon, *dataset.image_shape()), np.float32),
    }
    initial_times = [dataset.low_index.times[position] for position in initial_positions]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(initial_positions), batch_size):
            end = min(start + batch_size, len(initial_positions))
            histories = np.stack(
                [
                    dataset.read_physical_times(
                        range(position - dataset.history_hours + 1, position + 1)
                    )
                    for position in initial_positions[start:end]
                ],
                axis=0,
            )
            x = np.stack(
                [dataset.build_input_from_history(fields) for fields in histories],
                axis=0,
            )
            x = dataset.normalize_input(x)
            x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
            residual_channels = horizon * len(dataset.output_channels())
            zeros = torch.zeros(
                (end - start, residual_channels, *dataset.image_shape()), device=device
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                residual = model(zeros, x_tensor).float()
            physical = dataset.absolute_output_from_prediction(
                residual.cpu().numpy(), x
            )
            outputs["u"][start:end] = physical[..., 0, :, :]
            outputs["v"][start:end] = physical[..., 1, :, :]
            print(f"Completed initial times {end}/{len(initial_positions)}")
    return outputs["u"], outputs["v"], initial_times


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_export_wind_forecasts")
def main(cfg: DictConfig) -> None:
    configure_storage()
    if not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; forecast export will run on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _dataset(cfg.dataset)
    positions = _initial_positions(dataset, cfg.years)
    if cfg.get("max_initial_times"):
        positions = positions[: int(cfg.max_initial_times)]
    if not positions:
        raise RuntimeError("No initial times remain for export")
    forecast_hours = int(cfg.forecast_hours)
    if forecast_hours != dataset.forecast_horizon_hours:
        raise ValueError(
            "Export forecast_hours must match dataset.forecast_horizon_hours "
            f"({forecast_hours} != {dataset.forecast_horizon_hours})"
        )
    model = Module.from_checkpoint(cfg.checkpoint, {})
    if getattr(model, "forecast_hours", None) != forecast_hours:
        raise ValueError(
            "Checkpoint forecast horizon does not match export configuration: "
            f"{getattr(model, 'forecast_hours', None)} != {forecast_hours}"
        )
    model = model.eval().requires_grad_(False).to(device)
    u, v, initial_times = forecast(
        model,
        dataset,
        positions,
        batch_size=int(cfg.batch_size),
        device=device,
        amp=bool(cfg.amp) and device.type == "cuda",
    )
    write_forecast_netcdf(
        Path(cfg.output_file),
        u,
        v,
        initial_times,
        list(range(1, forecast_hours + 1)),
        dataset.latitude(),
        dataset.longitude(),
        source="wind-step1",
    )
    print(f"Wrote {len(initial_times)} initial times to {cfg.output_file}")


if __name__ == "__main__":
    main()
