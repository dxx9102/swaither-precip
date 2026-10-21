#!/usr/bin/env python3
"""Run the frozen Step-1 model autoregressively and export forecast conditions."""

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


def rollout(
    model: torch.nn.Module,
    dataset: WindForecastDataset,
    initial_positions: Sequence[int],
    max_lead: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, List[object]]:
    outputs = {
        "u": np.empty((len(initial_positions), max_lead, *dataset.image_shape()), np.float32),
        "v": np.empty((len(initial_positions), max_lead, *dataset.image_shape()), np.float32),
    }
    histories = np.stack(
        [
            dataset.read_physical_times(
                range(position - dataset.history_hours + 1, position + 1)
            )
            for position in initial_positions
        ],
        axis=0,
    )
    initial_times = [dataset.low_index.times[position] for position in initial_positions]
    model.eval()
    with torch.no_grad():
        for lead in range(1, max_lead + 1):
            for start in range(0, len(initial_positions), batch_size):
                end = min(start + batch_size, len(initial_positions))
                history_batch = histories[start:end]
                x = np.stack(
                    [dataset.build_input_from_history(fields) for fields in history_batch],
                    axis=0,
                )
                x = dataset.normalize_input(x)
                x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
                zeros = torch.zeros((end - start, 2, *dataset.image_shape()), device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    prediction = model(zeros, x_tensor).float()
                physical = dataset.denormalize_output(prediction.cpu().numpy())
                outputs["u"][start:end, lead - 1] = physical[:, 0]
                outputs["v"][start:end, lead - 1] = physical[:, 1]
                if lead < max_lead:
                    histories[start:end, :-1] = history_batch[:, 1:]
                    histories[start:end, -1, 0] = physical[:, 0]
                    histories[start:end, -1, 1] = physical[:, 1]
            print(f"Completed lead {lead}/{max_lead}")
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
    model = Module.from_checkpoint(cfg.checkpoint, {})
    model = model.eval().requires_grad_(False).to(device)
    u, v, initial_times = rollout(
        model,
        dataset,
        positions,
        max_lead=int(cfg.rollout_hours),
        batch_size=int(cfg.batch_size),
        device=device,
        amp=bool(cfg.amp) and device.type == "cuda",
    )
    write_forecast_netcdf(
        Path(cfg.output_file),
        u,
        v,
        initial_times,
        list(range(1, int(cfg.rollout_hours) + 1)),
        dataset.latitude(),
        dataset.longitude(),
        source="wind-step1",
    )
    print(f"Wrote {len(initial_times)} initial times to {cfg.output_file}")


if __name__ == "__main__":
    main()
