# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Dan Assouline and contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Originally based on physicsnemo/examples/generative/corrdiff/datasets/dataset.py.
# Built-in dataset registrations for cwb / hrrr_mini / gefs_hrrr have been
# removed since SwAIther-Precip does not use them; only user-defined datasets
# loaded via `register_dataset(...)` are supported.

from typing import Iterable, Tuple, Union
import copy
import torch
import importlib.util
from pathlib import Path

from physicsnemo.utils.diffusion import InfiniteSampler
from physicsnemo.distributed import DistributedManager

from . import base


# Mapping from dataset specs to initializer callables. Populated dynamically by
# `register_dataset(...)` when a custom dataset is loaded from a file path.
known_datasets = {}


def register_dataset(dataset_spec: str) -> None:
    """
    Register a new dataset class from a file path specification.

    Parameters
    ----------
    dataset_spec : str
        String specification in the format "path_to_file.py::dataset_class"

    Raises
    ------
    ValueError
        If the dataset_spec format is invalid or if the file doesn't exist
    ImportError
        If the dataset class cannot be imported
    """
    if dataset_spec in known_datasets:
        return  # Dataset already registered
    try:
        file_path, class_name = dataset_spec.split("::")
    except ValueError:
        raise ValueError(
            "Invalid dataset specification. Expected format: "
            "'path_to_file.py::dataset_class'"
        )
    if class_name in known_datasets:
        return  # Dataset already registered

    # Convert to Path and validate
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Dataset file not found: {file_path}")
    if not file_path.suffix == ".py":
        raise ValueError(f"Dataset file must be a Python file: {file_path}")

    # Import the module and get the class
    spec = importlib.util.spec_from_file_location(file_path.stem, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        dataset_class = getattr(module, class_name)
    except AttributeError:
        raise ImportError(
            f"Could not find dataset class '{class_name}' in {file_path}"
        )

    # Register the dataset
    known_datasets[dataset_spec] = dataset_class
    return


def init_train_valid_datasets_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
    validation_dataset_cfg: Union[dict, None] = None,
    validation: bool = True,
) -> Tuple[
    base.DownscalingDataset,
    Iterable,
    Union[base.DownscalingDataset, None],
    Union[Iterable, None],
]:
    """
    Build a training (and optionally validation) dataset from a config dict.

    Parameters
    ----------
    dataset_cfg : dict
        Dataset configuration. Must contain a `type` key with a dataset spec
        (typically `path/to/file.py::ClassName`).
    dataloader_cfg : dict, optional
        Additional keyword arguments forwarded to `torch.utils.data.DataLoader`.
    batch_size : int, default 1
        Mini-batch size.
    seed : int, default 0
        Random seed for the InfiniteSampler.
    validation_dataset_cfg : dict, optional
        Overrides for the validation dataset. Inherits unspecified keys from
        `dataset_cfg`.
    validation : bool, default True
        Whether to create a paired validation dataset.

    Returns
    -------
    Tuple of (train_dataset, train_iterator, valid_dataset_or_None,
    valid_iterator_or_None).
    """
    config = copy.deepcopy(dataset_cfg)
    (dataset, dataset_iter) = init_dataset_from_config(
        config, dataloader_cfg, batch_size=batch_size, seed=seed
    )
    if validation:
        valid_dataset_cfg = copy.deepcopy(config)
        if validation_dataset_cfg:
            valid_dataset_cfg.update(validation_dataset_cfg)
        (valid_dataset, valid_dataset_iter) = init_dataset_from_config(
            valid_dataset_cfg, dataloader_cfg, batch_size=batch_size, seed=seed
        )
    else:
        valid_dataset = valid_dataset_iter = None

    return dataset, dataset_iter, valid_dataset, valid_dataset_iter


def init_dataset_from_config(
    dataset_cfg: dict,
    dataloader_cfg: Union[dict, None] = None,
    batch_size: int = 1,
    seed: int = 0,
) -> Tuple[base.DownscalingDataset, Iterable]:
    """Instantiate a dataset and its DataLoader iterator from a config dict."""
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type")
    if "validation" in dataset_cfg:
        # handled by init_train_valid_datasets_from_config
        del dataset_cfg["validation"]
    if dataset_type not in known_datasets:
        raise KeyError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Did you forget to call register_dataset('{dataset_type}') first? "
            f"Known datasets: {list(known_datasets.keys())}"
        )
    dataset_init_func = known_datasets[dataset_type]

    dataset_obj = dataset_init_func(**dataset_cfg)
    if dataloader_cfg is None:
        dataloader_cfg = {}

    dist = DistributedManager()
    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj, rank=dist.rank, num_replicas=dist.world_size, seed=seed
    )

    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_size,
            worker_init_fn=None,
            **dataloader_cfg,
        )
    )

    return (dataset_obj, dataset_iterator)