#!/usr/bin/env python3
"""Generate CorrDiff wind super-resolution from exported Step-1 conditions."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import Any, List, Sequence

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
from physicsnemo.utils.corrdiff import diffusion_step, regression_step
from physicsnemo.utils.diffusion import stochastic_sampler

from swaither.datasets.wind import WindSuperResolutionDataset


def _dataset(cfg: DictConfig) -> WindSuperResolutionDataset:
    kwargs = OmegaConf.to_container(cfg, resolve=True)
    kwargs.pop("type", None)
    return WindSuperResolutionDataset(**kwargs)


def _create_output(
    path: Path,
    initial_times: Sequence[pd.Timestamp],
    leads: Sequence[int],
    lat: np.ndarray,
    lon: np.ndarray,
    ensemble: int | None = None,
) -> tuple[Any, Any, Any]:
    import netCDF4 as nc

    path.parent.mkdir(parents=True, exist_ok=True)
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("initial_time", len(initial_times))
    ds.createDimension("lead_hour", len(leads))
    if ensemble is not None:
        ds.createDimension("ensemble", ensemble)
    ds.createDimension("lat", len(lat))
    ds.createDimension("lon", len(lon))
    initial_var = ds.createVariable("initial_time", str, ("initial_time",))
    initial_var[:] = np.asarray(
        [pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ") for value in initial_times],
        dtype=object,
    )
    lead_var = ds.createVariable("lead_hour", "i4", ("lead_hour",))
    lead_var[:] = np.asarray(leads, dtype=np.int32)
    lead_var.units = "hours"
    valid_var = ds.createVariable("valid_time", str, ("initial_time", "lead_hour"))
    valid_var[:] = np.asarray(
        [
            [
                (pd.Timestamp(initial) + pd.Timedelta(hours=int(lead))).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                for lead in leads
            ]
            for initial in initial_times
        ],
        dtype=object,
    )
    valid_var.units = "ISO 8601 UTC"
    lat_var = ds.createVariable("lat", "f4", ("lat",))
    lat_var[:] = lat
    lat_var.units = "degrees_north"
    lon_var = ds.createVariable("lon", "f4", ("lon",))
    lon_var[:] = lon
    lon_var.units = "degrees_east"
    dims = (
        ("initial_time", "lead_hour", "ensemble", "lat", "lon")
        if ensemble is not None
        else ("initial_time", "lead_hour", "lat", "lon")
    )
    variables = {}
    for name in ("u10", "v10"):
        var = ds.createVariable(name, "f4", dims, zlib=True, complevel=2, shuffle=True)
        var.units = "m s-1"
        var.coordinates = "valid_time lat lon"
        var.long_name = "10 metre eastward wind" if name == "u10" else "10 metre northward wind"
        variables[name] = var
    return ds, variables, (initial_var, lead_var, lat_var, lon_var)


def _selected_indices(dataset: WindSuperResolutionDataset, mode: str, cfg: DictConfig) -> List[int]:
    selected = []
    diffusion_leads = set(int(value) for value in cfg.inference.diffusion_leads)
    for index in range(len(dataset)):
        metadata = dataset.sample_metadata(index)
        if mode == "diffusion":
            if metadata["initial_time"].hour != int(cfg.inference.diffusion_initial_hour):
                continue
            if metadata["lead_hour"] not in diffusion_leads:
                continue
        selected.append(index)
    return selected


@hydra.main(version_base="1.2", config_path="../conf", config_name="config_inference_wind_pipeline")
def main(cfg: DictConfig) -> None:
    configure_storage()
    if not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; CorrDiff inference will run on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = _dataset(cfg.dataset)
    mode = str(cfg.inference.mode)
    if mode not in ("regression", "diffusion", "all"):
        raise ValueError("inference.mode must be regression, diffusion, or all")

    net_reg = None
    if mode in ("regression", "all", "diffusion"):
        net_reg = Module.from_checkpoint(cfg.checkpoints.regression, {}).eval().to(device)
        net_reg.requires_grad_(False)
    net_dif = None
    if mode in ("diffusion", "all"):
        net_dif = Module.from_checkpoint(cfg.checkpoints.diffusion, {}).eval().to(device)
        net_dif.requires_grad_(False)

    img_shape = dataset.image_shape()
    num_ensembles = int(cfg.inference.num_ensembles)
    sampler_fn = partial(
        stochastic_sampler,
        num_steps=int(cfg.inference.sampler_num_steps),
    )
    rng = np.random.default_rng(int(cfg.inference.seed))

    def run_mode(run_mode: str) -> None:
        indices = _selected_indices(dataset, run_mode, cfg)
        if not indices:
            print(f"No samples selected for {run_mode}; skipping")
            return
        metadata = [dataset.sample_metadata(index) for index in indices]
        initial_times = sorted({item["initial_time"] for item in metadata})
        leads = sorted({int(item["lead_hour"]) for item in metadata})
        initial_to_index = {value: i for i, value in enumerate(initial_times)}
        lead_to_index = {value: i for i, value in enumerate(leads)}
        output_path = Path(
            cfg.outputs.regression_file if run_mode == "regression" else cfg.outputs.diffusion_file
        )
        ds, variables, _ = _create_output(
            output_path,
            initial_times,
            leads,
            dataset.latitude(),
            dataset.longitude(),
            num_ensembles if run_mode == "diffusion" else None,
        )
        try:
            for count, index in enumerate(indices, start=1):
                item = metadata[count - 1]
                _y, x = dataset[index]
                x = x.unsqueeze(0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                i = initial_to_index[item["initial_time"]]
                lead = lead_to_index[int(item["lead_hour"])]
                mean = regression_step(
                    net_reg,
                    x,
                    latents_shape=(1, 2, img_shape[0], img_shape[1]),
                )
                if run_mode == "regression":
                    physical = dataset.denormalize_output(mean.cpu().numpy())[0]
                    variables["u10"][i, lead] = physical[0]
                    variables["v10"][i, lead] = physical[1]
                else:
                    seeds = [int(value) for value in rng.integers(0, 2**31 - 1, size=num_ensembles, dtype=np.int64)]
                    residual = diffusion_step(
                        net=net_dif,
                        sampler_fn=sampler_fn,
                        img_shape=img_shape,
                        img_out_channels=2,
                        rank_batches=[seeds],
                        img_lr=x.expand(num_ensembles, -1, -1, -1).contiguous(),
                        rank=0,
                        device=device,
                        mean_hr=mean,
                    )
                    full = mean.repeat(num_ensembles, 1, 1, 1) + residual
                    physical = dataset.denormalize_output(full.cpu().numpy())
                    variables["u10"][i, lead] = physical[:, 0]
                    variables["v10"][i, lead] = physical[:, 1]
                if count % int(cfg.inference.print_every) == 0 or count == len(indices):
                    print(f"{run_mode}: {count}/{len(indices)}")
        finally:
            ds.close()
        print(f"Wrote {output_path}")

    if mode in ("regression", "all"):
        run_mode("regression")
    if mode in ("diffusion", "all"):
        run_mode("diffusion")


if __name__ == "__main__":
    main()
