#!/usr/bin/env python3
"""Compute train-only robust statistics for the two-model wind pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import hydra
import numpy as np
import xarray as xr
from omegaconf import DictConfig

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from swaither.datasets.wind import FileTimeIndex
from swaither.wind_helpers import configure_storage

configure_storage()


def _reservoir_update(
    store: Dict[str, np.ndarray],
    names: Sequence[str],
    array: np.ndarray,
    limit: int,
) -> None:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("Reservoir statistics require an array with at least two dimensions")
    if values.ndim == 2:
        values = values[None, ...]
    if values.shape[0] != len(names):
        raise ValueError(
            f"Expected {len(names)} channel-first fields, got shape {values.shape}"
        )
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
        if store[name].size > 2 * limit:
            positions = np.linspace(0, store[name].size - 1, limit, dtype=np.int64)
            store[name] = store[name][positions]


def _stats_from_store(
    store: Dict[str, np.ndarray], low: float, high: float
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for name, values in store.items():
        mn, mx = np.quantile(values, [low, high]).tolist()
        if not np.isfinite(mn) or not np.isfinite(mx) or mx < mn:
            raise ValueError(f"Invalid robust range for {name}: {(mn, mx)}")
        if mx == mn:
            mx = mn + 1.0
        result[name] = {"min": float(mn), "max": float(mx)}
    return result


def _sample_positions(length: int, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, max_samples, dtype=np.int64)


def _dynamic_stats(
    index: FileTimeIndex,
    variables: Sequence[str],
    positions: np.ndarray,
    store: Dict[str, np.ndarray],
    limit: int,
    engine: str,
) -> None:
    chunks = list(range(0, len(positions), 128))
    label = ",".join(variables)
    for chunk_index, offset in enumerate(chunks, start=1):
        chunk = positions[offset : offset + 128]
        print(
            f"[stats] {label}: chunk {chunk_index}/{len(chunks)} "
            f"({len(chunk)} fields)",
            flush=True,
        )
        values = index.read_variables_many(
            variables, chunk, index.lat, index.lon, engine
        )
        # ``read_variables_many`` returns (time, channel, lat, lon), while the
        # reservoir helper uses a channel-first contract.  Transposing preserves
        # every sampled time for each channel instead of silently keeping only
        # the first field in each chunk.
        _reservoir_update(store, variables, values.transpose(1, 0, 2, 3), limit)


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_compute_wind_stats")
def main(cfg: DictConfig) -> None:
    configure_storage()
    low_index = FileTimeIndex(cfg.dataset.low_res_files, engine=cfg.dataset.engine)
    high_index = FileTimeIndex(cfg.dataset.high_res_files, engine=cfg.dataset.engine)
    low_positions = _sample_positions(len(low_index), int(cfg.max_samples))
    high_positions = _sample_positions(len(high_index), int(cfg.max_samples))
    limit = int(cfg.values_per_channel)
    inputs: Dict[str, np.ndarray] = {}
    outputs: Dict[str, np.ndarray] = {}

    _dynamic_stats(
        low_index,
        cfg.dataset.low_res_variables,
        low_positions,
        inputs,
        limit,
        cfg.dataset.engine,
    )
    with xr.open_dataset(cfg.dataset.static_high_file, engine=cfg.dataset.engine) as ds:
        for name in cfg.dataset.invariant_variables:
            print(f"[stats] static: {name}", flush=True)
            _reservoir_update(inputs, [name], np.asarray(ds[name].values), limit)
    _dynamic_stats(
        high_index,
        cfg.dataset.high_res_variables,
        high_positions,
        outputs,
        limit,
        cfg.dataset.engine,
    )

    output_named = {
        output_name: outputs[source_name]
        for source_name, output_name in zip(
            cfg.dataset.high_res_variables, cfg.dataset.output_variables
        )
    }
    payload = {
        "input": _stats_from_store(
            inputs, float(cfg.quantile_low), float(cfg.quantile_high)
        ),
        "output": _stats_from_store(
            output_named, float(cfg.quantile_low), float(cfg.quantile_high)
        ),
        "metadata": {
            "task": "wind-joint",
            "train_years": list(cfg.train_years),
            "sample_count": len(low_positions),
            "high_res_sample_count": len(high_positions),
            "values_per_channel": limit,
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
