#!/usr/bin/env python3
"""Compute train-only robust min/max statistics for the wind pipeline."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.datasets.wind import WindForecastDataset, WindSuperResolutionDataset
from swaither.wind_helpers import configure_storage


configure_storage()


def _dataset(cls, cfg: DictConfig):
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs.pop("type", None)
    kwargs["normalize"] = False
    if cls is WindForecastDataset and "low_res_files" not in kwargs and "low_res_file" in kwargs:
        kwargs["low_res_files"] = kwargs.pop("low_res_file")
    if cls is WindSuperResolutionDataset and "condition_files" not in kwargs and "low_res_file" in kwargs:
        kwargs["condition_files"] = kwargs.pop("low_res_file")
    if cls is WindSuperResolutionDataset and "high_res_files" not in kwargs and "high_res_file" in kwargs:
        kwargs["high_res_files"] = kwargs.pop("high_res_file")
    return cls(**kwargs)


def _reservoir_update(store: Dict[str, np.ndarray], names: List[str], array: np.ndarray, limit: int) -> None:
    values = np.asarray(array, dtype=np.float32)
    for channel, name in enumerate(names):
        flat = values[channel].ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            raise ValueError(f"Channel {name} has no finite values")
        if finite.size > limit:
            positions = np.linspace(0, finite.size - 1, limit, dtype=np.int64)
            finite = finite[positions]
        previous = store.get(name)
        store[name] = finite if previous is None else np.concatenate([previous, finite])
        # Amortize compaction. Copying a full one-million-value reservoir for
        # every hourly sample is prohibitively expensive at 35k training times.
        if store[name].size > 2 * limit:
            positions = np.linspace(0, store[name].size - 1, limit, dtype=np.int64)
            store[name] = store[name][positions]


def _stats_from_store(store: Dict[str, np.ndarray], low: float, high: float) -> Dict[str, Dict[str, float]]:
    result = {}
    for name, values in store.items():
        mn, mx = np.quantile(values, [low, high]).tolist()
        if not np.isfinite(mn) or not np.isfinite(mx) or mx < mn:
            raise ValueError(f"Invalid robust range for {name}: {(mn, mx)}")
        # A constant invariant channel (for example, an all-ocean mask tile)
        # is valid input.  Give it a unit denominator so normalization remains
        # well-defined while mapping the constant field to zero.
        if mx == mn:
            mx = mn + 1.0
        result[name] = {"min": float(mn), "max": float(mx)}
    return result


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_compute_wind_stats")
def main(cfg: DictConfig) -> None:
    configure_storage()
    mode = str(cfg.mode)
    if mode == "step1":
        data = _dataset(WindForecastDataset, cfg.dataset)
        input_names = data.physical_input_names
        output_names = data.output_variables
    elif mode == "step2":
        data = _dataset(WindSuperResolutionDataset, cfg.dataset)
        input_names = data.physical_input_names
        output_names = data.output_variables
    else:
        raise ValueError("mode must be step1 or step2")

    if len(data) == 0:
        raise RuntimeError("Stats dataset is empty")
    max_samples = int(cfg.max_samples)
    indices = np.arange(len(data)) if max_samples <= 0 or max_samples >= len(data) else np.linspace(0, len(data) - 1, max_samples, dtype=np.int64)
    inputs: Dict[str, np.ndarray] = {}
    outputs: Dict[str, np.ndarray] = {}
    value_limit = int(cfg.values_per_channel)
    print(f"Computing {mode} statistics from {len(indices)} samples out of {len(data)}")
    for count, index in enumerate(indices, start=1):
        y, x = data[int(index)]
        _reservoir_update(inputs, input_names, x[: len(input_names)].numpy(), value_limit)
        if mode == "step1":
            # WindForecastDataset returns normalized residuals. Statistics must
            # describe absolute target fields because reconstruction adds them
            # to the normalized final observation.
            absolute_targets = data.read_physical_targets(int(index))
            channel_first = np.moveaxis(absolute_targets, 1, 0)
            _reservoir_update(outputs, output_names, channel_first, value_limit)
        else:
            _reservoir_update(outputs, output_names, y.numpy(), value_limit)
        if count % 100 == 0 or count == len(indices):
            print(f"  processed {count}/{len(indices)}")

    payload = {
        "input": _stats_from_store(inputs, float(cfg.quantile_low), float(cfg.quantile_high)),
        "output": _stats_from_store(outputs, float(cfg.quantile_low), float(cfg.quantile_high)),
        "metadata": {
            "mode": mode,
            "train_years": list(cfg.train_years),
            "sample_count": len(indices),
            "quantile_low": float(cfg.quantile_low),
            "quantile_high": float(cfg.quantile_high),
        },
    }
    destination = Path(cfg.output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
