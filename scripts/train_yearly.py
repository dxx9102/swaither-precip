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
# Originally based on physicsnemo/examples/generative/corrdiff/train.py.

import os
import sys

# Make the swaither/ package importable regardless of where this script is launched from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- Backwards-compatibility alias ---
# Some pre-existing checkpoints record their model class's import path as
# `models.X` (the package was named `models` at the repo root before being
# moved under `swaither/`). PhysicsNeMo's `Module.from_checkpoint` looks up
# the class by importing that module path; this alias makes such lookups
# resolve to `swaither.models.X` instead. New checkpoints record the
# correct path and don't need this alias.
import swaither.models as _legacy_models_alias
sys.modules.setdefault("models", _legacy_models_alias)

import time, psutil, hydra, torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import wandb
from hydra.core.hydra_config import HydraConfig
from contextlib import nullcontext

from physicsnemo import Module
from physicsnemo.models.diffusion import UNet, EDMPrecondSuperResolution
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.diffusion import RegressionLoss, ResidualLoss, RegressionLossCE
from physicsnemo.utils.patching import RandomPatching2D

from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)

from swaither.datasets.dataset import init_train_valid_datasets_from_config, init_dataset_from_config, register_dataset

from swaither.helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)
import nvtx

import math
# at the top, after your other imports
from torch.utils.data import random_split, DataLoader
from physicsnemo.utils.diffusion import InfiniteSampler
from torch.utils.data import Subset
from physicsnemo.launch.utils import load_checkpoint
import matplotlib.pyplot as plt

# for the epoch progress bar
from tqdm.auto import tqdm

# for the custom cmap
from swaither.helpers.viz_helpers import (
    ColormapConfig,
)
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

# import the custom bias corrector
from swaither.models import UNetResidualCorrectorClamped3, UNetResidualCorrectorClampedFiLM

# and the mass constraints wrappers
#from swaither.models import EDMPrecondSuperResolutionWithGlobalMass, EDMPrecondSuperResolutionWithGlobalMassScheduled

# ViT model
from swaither.models import ViTBiasCorrectorFiLM   

# for the custom loss with temporal regularization 
from swaither.losses.custom_losses import RegressionLossTemporalSmooth, RegressionLossTimeConsistency, RegressionHuberLossTimeConsistency


import numpy as np
# ------ my changes: show visible gpus ---------
print("=== Environment ===")
print(" CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print(" torch.cuda.is_available() =", torch.cuda.is_available())
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f" torch sees {n} GPU(s):")
    for i in range(n):
        print(f"   · GPU {i}: {torch.cuda.get_device_name(i)}")

# Pre-load the Swiss border geometry once
shp_fname = shpreader.natural_earth(
    resolution='50m', category='cultural', name='admin_0_countries'
)
reader     = shpreader.Reader(shp_fname)
switz_geom = next(rec.geometry for rec in reader.records()
                  if rec.attributes['NAME_LONG'] == "Switzerland")
# ------ 

torch._dynamo.reset()
# Increase the cache size limit
torch._dynamo.config.cache_size_limit = 264  # Set to a higher value
torch._dynamo.config.verbose = True  # Enable verbose logging
torch._dynamo.config.suppress_errors = False  # Forces the error to show all details
torch._logging.set_logs(recompiles=True, graph_breaks=True)


def checkpoint_list(path, suffix=".mdlus"):
    """Helper function to return sorted list, in ascending order, of checkpoints in a path"""
    checkpoints = []
    for file in os.listdir(path):
        if file.endswith(suffix):
            # Split the filename and extract the index
            try:
                index = int(file.split(".")[-2])
                checkpoints.append((index, file))
            except ValueError:
                continue

    # Sort by index and return filenames
    checkpoints.sort(key=lambda x: x[0])
    return [file for _, file in checkpoints]


# Define safe CUDA profiler tools that fallback to no-ops when CUDA is not available
def cuda_profiler():
    if torch.cuda.is_available():
        return torch.cuda.profiler.profile()
    else:
        return nullcontext()


def cuda_profiler_start():
    if torch.cuda.is_available():
        torch.cuda.profiler.start()


def cuda_profiler_stop():
    if torch.cuda.is_available():
        torch.cuda.profiler.stop()


def profiler_emit_nvtx():
    if torch.cuda.is_available():
        return torch.autograd.profiler.emit_nvtx()
    else:
        return nullcontext()


# Train the CorrDiff model using the configurations in "conf/config_training.yaml"
@hydra.main(version_base="1.2", config_path="../conf", config_name="config_training")
def main(cfg: DictConfig) -> None:

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    
    # ----------- Define project and entity ------
    initialize_wandb(
        project=cfg.wandb.project,     # comes from your YAML or CLI override
        entity=cfg.wandb.entity,       # ditto
        name=f"CorrDiff-Training-{HydraConfig.get().job.name}",
        group="CorrDiff-DDP-Group",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg),
        results_dir=cfg.wandb.results_dir,
    )

    # Resolve and parse configs
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)  # TODO needs better handling

    # Register custom dataset if specified in config
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    if hasattr(cfg, "validation"):
        validation = True
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation)
    else:
        validation = False
        validation_dataset_cfg = None
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16
    logger.info(f"Saving the outputs in {os.getcwd()}")
    checkpoint_dir = get_checkpoint_dir(
        str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name
    )

    # ---  directory for pure PyTorch checkpoints (no Modulus / .mdlus) ---
    pure_ckpt_dir = os.path.join(checkpoint_dir, "pure_torch")
    os.makedirs(pure_ckpt_dir, exist_ok=True)

    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )

    # Set seeds and configure CUDA and cuDNN settings to ensure consistent precision
    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2 if cfg.training.perf.dataloader_workers > 0 else None,
    }


    # # # ---- YEARLY split ────────────────────────────────────
    # # 1) Instantiate full dataset (same as before)
    base_dataset, _ = init_dataset_from_config(
                    dataset_cfg,
                    data_loader_kwargs,
                    batch_size=None,   # dataset-level only, no batching here
                    seed=0,
                    )

    N = len(base_dataset)
    print(f"[INFO] Full dataset length (samples): {N}")

    # 2) Get times ONCE from the dataset
    times_list = base_dataset.time()   # length N, one timestamp per sample (time,lead)
    if len(times_list) != N:
        print(f"[WARN] base_dataset.time() length {len(times_list)} != len(base_dataset) {N}")

    # 3) Extract year from each timestamp.
    #    Handles: numpy datetime64, Python datetime, or any object with a .year attribute.
    import pandas as pd

    def _get_year(t):
        if isinstance(t, np.datetime64):
            return pd.Timestamp(t).year
        elif hasattr(t, "year"):        # datetime.datetime / pandas Timestamp
            return t.year
        else:                           # fallback: convert via pandas
            return pd.Timestamp(t).year

    years = np.array([_get_year(t) for t in times_list], dtype=np.int32)

    all_years = np.unique(years)
    val_years = set(all_years[-2:].tolist()) # last 2 years → val
    train_years = set(all_years[:-2].tolist())  # everything before → train
    print(f"[INFO] Years in dataset : {all_years.tolist()}")
    print(f"[INFO] Val years        : {sorted(val_years)}  (validation)")
    print(f"[INFO] Training years   : {sorted(train_years)}")

    # 4) Yearly split: last 2 years → val, all previous → train
    train_idx = [i for i, y in enumerate(years) if y in train_years]
    val_idx   = [i for i, y in enumerate(years) if y in val_years]

    print(f"[INFO] Train samples (yearly): {len(train_idx)}")
    print(f"[INFO] Val samples   (yearly): {len(val_idx)}")
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Yearly split produced empty train or val set. Check your timestamps!")

    # ------------------------


    # 5) Build Subsets
    train_ds = Subset(base_dataset, train_idx)
    val_ds   = Subset(base_dataset, val_idx)

    # 6) Create samplers and dataloaders (unchanged)
    dist = DistributedManager()
    train_sampler = InfiniteSampler(
        train_ds,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=0,
    )
    val_sampler = InfiniteSampler(
        val_ds,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=0,
    )

    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        **data_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        sampler=val_sampler,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        **data_loader_kwargs,
    )

    # 7) Expose downstream
    dataset                     = train_ds
    dataset_iterator            = iter(train_loader)
    validation_dataset          = val_ds
    validation_dataset_iterator = iter(val_loader)

    # Optional: unwrap Subset for code that expects the base dataset object
    dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

    # print(f"[FINAL] 6-on/1-off split → Train size = {len(train_idx)}, Val size = {len(val_idx)}")
    
    print(f"[FINAL] Yearly split → Train size = {len(train_idx)}, Val size = {len(val_idx)}")

    # -------- initialize epoch for epoch tracking
    # keep base_dataset as-is for stats, but use train_ds for epoch sizing
    dataset_for_stats = base_dataset

    # ---------------- For Curriculum learning --------------------
    # ─────────────────────────────────────────────────────────────
    # Curriculum learning on lead times
    # cfg.training.hp.curriculum.enabled: bool
    # cfg.training.hp.curriculum.epochs_per_step: int (e.g. 1)
    # cfg.training.hp.curriculum.leads_per_step: int (e.g. 1)
    # cfg.training.hp.curriculum.min_lead_label: optional (default = smallest lead_values)
    # ─────────────────────────────────────────────────────────────
    curriculum_cfg = getattr(cfg.training.hp, "curriculum", None)
    if curriculum_cfg is not None and getattr(curriculum_cfg, "enabled", False):
        use_curriculum = True

        # Infer available lead labels from dataset if possible
        if hasattr(base_dataset, "lead_values"):
            lv = base_dataset.lead_values
            # handle attribute vs. method
            if callable(lv):
                lv = lv()
            lead_values = np.array(lv)
            lead_values_sorted = np.sort(lead_values)
        else:
            # fallback: assume 0..n_leads-1
            n_leads = int(cfg.model.model_args.get("n_leads", 24))
            lead_values_sorted = np.arange(n_leads, dtype=np.int64)

        min_lead_label = int(getattr(curriculum_cfg, "min_lead_label", int(lead_values_sorted[0])))
        max_lead_label = int(lead_values_sorted[-1])
        epochs_per_step = int(getattr(curriculum_cfg, "epochs_per_step", 1))
        leads_per_step  = int(getattr(curriculum_cfg, "leads_per_step", 1))

        if dist.rank == 0:
            logger0.info(
                f"[Curriculum] ENABLED: min_lead_label={min_lead_label}, "
                f"max_lead_label={max_lead_label}, epochs_per_step={epochs_per_step}, "
                f"leads_per_step={leads_per_step}"
            )
    else:
        use_curriculum = False
        min_lead_label = None
        max_lead_label = None
        epochs_per_step = None
        leads_per_step  = None


    lead_levels = None  # will hold sorted unique lead labels from data

    # ---------------- For Multi-Step Fine tuning --------------------
    # ---  fixed lead filter for staged fine-tuning ---
    lead_filter_cfg = getattr(cfg.training.hp, "lead_filter", None)
    if lead_filter_cfg is not None and getattr(lead_filter_cfg, "enabled", False):
        use_lead_filter = True
        lead_filter_min = int(getattr(lead_filter_cfg, "min_lead", 0))
        lead_filter_max = int(getattr(lead_filter_cfg, "max_lead", 23))
        if dist.rank == 0:
            logger0.info(
                f"[LeadFilter] ENABLED: keeping leads in "
                f"[{lead_filter_min}, {lead_filter_max}] for this run."
            )
    else:
        use_lead_filter = False
        lead_filter_min = None
        lead_filter_max = None

    # ----------------# ----------------# ---------------# ----------------
    # # create a tqdm bar for epoch 0
    # if dist.rank == 0:
    #     pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", unit="batch")
    # ────────────────────────────────────────────────────────────────────────────

    # Parse image configuration & update model args
    dataset_channels = len(dataset_for_stats.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset_for_stats.image_shape()
    img_out_channels = len(dataset_for_stats.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    # Handle patch shape
    if cfg.model.name in [ ]:
        prob_channels = dataset.get_prob_channel_index()
    else:
        prob_channels = None
    # Parse the patch shape
    if (
        cfg.model.name == "patched_diffusion"
        or cfg.model.name == "lt_aware_patched_diffusion"
    ):
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None
    if (
        patch_shape_x
        and patch_shape_y
        and patch_shape_y >= img_shape[0]
        and patch_shape_x >= img_shape[1]
    ):
        logger0.warning(
            f"Patch shape {patch_shape_y}x{patch_shape_x} is larger than \
            the image shape {img_shape[0]}x{img_shape[1]}. Patching will not be used."
        )
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        # Utility to perform patches extraction and batching
        patching = RandomPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            patch_num=getattr(cfg.training.hp, "patch_num", 1),
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
        logger0.info("Patch-based training disabled")
    # interpolate global channel if patch-based model is used
    if use_patching:
        img_in_channels += dataset_channels

    # Instantiate the model and move to device.
    model_args = {  # default parameters for all networks
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
        "checkpoint_level": songunet_checkpoint_level,
    }
    if cfg.model.name in [  ]:
        model_args["prob_channels"] = prob_channels
    if hasattr(cfg.model, "model_args"):  # override defaults from config file
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    # ────────────────────────────────────────────────────────────────────────────
    # Sanitize all args to plain Python types so Module.save() can json-dump them:
    import numpy as _np
    def _sanitize(v):
        if isinstance(v, _np.generic):
            return v.item()
        if isinstance(v, (list, tuple)):
            return type(v)(_sanitize_i for _sanitize_i in (_sanitize(i) for i in v))
        return v
    model_args = {k: _sanitize(v) for k, v in model_args.items()}
    # ────────────────────────────────────────────────────────────────────────────

    use_torch_compile = False
    use_apex_gn = False
    profile_mode = False

    if hasattr(cfg.training.perf, "torch_compile"):
        use_torch_compile = cfg.training.perf.torch_compile
    if hasattr(cfg.training.perf, "use_apex_gn"):
        use_apex_gn = cfg.training.perf.use_apex_gn
        model_args["use_apex_gn"] = use_apex_gn

    if hasattr(cfg.training.perf, "profile_mode"):
        profile_mode = cfg.training.perf.profile_mode
        model_args["profile_mode"] = profile_mode

    if enable_amp:
        model_args["amp_mode"] = enable_amp

    if cfg.model.name == "regression_bias_correction":
        
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )

    elif cfg.model.name == "regression":
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )


    elif cfg.model.name == "lt_aware_ce_regression":
        model = UNet(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )

    elif cfg.model.name == "lt_aware_ce_regression_bias_correction":
        model = UNet(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )

    elif cfg.model.name == "lt_aware_FiLM_regression_bias_correction":

        ###### UNetResidualCorrectorClampedFiLM model ########
        model_args = cfg.model.model_args

        # ------------------------------------------------------------------
        # 1) Physical channels only = what the dataset gives you
        #    (AIFS vars + altitude). This is exactly C in img_lr.shape[1].
        # ------------------------------------------------------------------
        in_ch_phys = img_in_channels      # <- from dataset / init_dataset_from_config
        in_ch_total = in_ch_phys          # DO *NOT* add N_grid_channels here

        # We FiLM all physical channels; no grid channels at this level
        n_film_channels = in_ch_total

        # ------------------------------------------------------------------
        # 2) Build the underlying UNetResidualCorrectorClamped3
        #    Use SongUNet (no built-in lead embeddings, FiLM handles lead)
        # ------------------------------------------------------------------
        unet_kwargs = {
           "img_in_channels": in_ch_total,
           "img_out_channels": img_out_channels,           # 1 precip channel
           "img_resolution": list(img_shape),
           "use_fp16": fp16,
           "model_type": "SongUNet",                       # <--- important
           "channel_mult": list(model_args.get("channel_mult", [1, 2, 2])),
           "attn_resolutions": list(model_args.get("attn_resolutions", [16, 8, 4])),
           "amp_mode": True,
           # add any extra SongUNet args you normally use:
           # "num_res_blocks": model_args.get("num_res_blocks", 2),
           # "dropout": model_args.get("dropout", 0.0),
        }

        base_model_kwargs = {
           "img_in_channels": in_ch_total,
           "img_out_channels": img_out_channels,
           "unet_kwargs": unet_kwargs,
           "clamp_min": float(model_args.get("clamp_min", 0.0)),
           "clamp_max": float(model_args.get("clamp_max", 20.0)),
        }

        # ------------------------------------------------------------------
        # 3) Wrap with FiLM
        # ------------------------------------------------------------------
        model = UNetResidualCorrectorClampedFiLM(
           base_model_kwargs=base_model_kwargs,
           total_in_channels=in_ch_total,          # MUST match img_lr.shape[1] = 12
           n_film_channels=n_film_channels,        # FiLM all phys channels
           n_leads=int(model_args.get("n_leads", 24)),
           emb_dim=int(model_args.get("emb_dim", 32)),
           mlp_hidden=int(model_args.get("mlp_hidden", 128)),
        )
       
        # ----------------------------------------------------------------------------------------------------
        # To train a ViT model instead of a UNet: run the following part, comment the previous UNet parameters
        # ----------------------------------------------------------------------------------------------------
        # model_args = cfg.model.model_args
        
        # in_ch_phys = img_in_channels   # your physical channels only  (no grid)
        # img_res = list(img_shape)   
        
        # model = ViTBiasCorrectorFiLM(
        #      img_in_channels=in_ch_phys,
        #      img_out_channels=img_out_channels,
        #      img_resolution=img_res,
        #      patch_size=int(model_args.get("patch_size", 4)),
        #      d_model=int(model_args.get("d_model", 128)),
        #      depth=int(model_args.get("depth", 6)),
        #      n_heads=int(model_args.get("n_heads", 4)),
        #      mlp_hidden=256,
        #      n_leads=int(model_args.get("n_leads", 24)),
        #      emb_dim=int(model_args.get("emb_dim", 64)),
        # )


    elif cfg.model.name == "lt_aware_patched_diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )

        
    elif cfg.model.name == "diffusion":

        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )


    elif cfg.model.name == "patched_diffusion":
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    else:
        raise ValueError(f"Invalid model: {cfg.model.name}")

    model.train().requires_grad_(True).to(dist.device)

    if use_apex_gn:
        model.to(memory_format=torch.channels_last)

    
    if cfg.model.name in ("regression", "lt_aware_ce_regression", "regression_bias_correction", "lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"):
        regression_net = model
    else:
        regression_net = None
    
    # Check if regression model is used with patching
    if (
        cfg.model.name in ["regression", "lt_aware_ce_regression", "regression_bias_correction","lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"]
        and patching is not None
    ):
        raise ValueError(
            f"Regression model ({cfg.model.name}) cannot be used with patch-based training. "
        )

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,  # dist.find_unused_parameters,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
        )
    if cfg.wandb.watch_model and dist.rank == 0:
        wandb.watch(model)

    # Load the regression checkpoint if applicable
    if (
        hasattr(cfg.training.io, "regression_checkpoint_path")
        and cfg.training.io.regression_checkpoint_path is not None
    ):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        regression_net = Module.from_checkpoint(
            regression_checkpoint_path, {}
        )
        ## ------ Physicsnemo changes
        #regression_net.amp_mode = enable_amp
        #regression_net.profile_mode = profile_mode

        # manually configure
        regression_net.use_apex_gn = use_apex_gn
        if hasattr(regression_net, "amp_mode"):
            regression_net.amp_mode = enable_amp
        if hasattr(regression_net, "profile_mode"):
            regression_net.profile_mode = profile_mode

        # ------
        
        regression_net.eval().requires_grad_(False).to(dist.device)
        if use_apex_gn:
            regression_net.to(memory_format=torch.channels_last)
        logger0.success("Loaded the pre-trained regression model")

    # Compile the model and regression net if applicable
    if use_torch_compile:
        model = torch.compile(model)
        if regression_net:
            regression_net = torch.compile(regression_net)

    # Compute the number of required gradient accumulation rounds
    # It is automatically used if batch_size_per_gpu * dist.world_size < total_batch_size
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    max_patch_per_gpu = getattr(cfg.training.hp, "max_patch_per_gpu", 1)

    # calculate patch per iter
    if hasattr(cfg.training.hp, "max_patch_per_gpu") and max_patch_per_gpu > 1:
        max_patch_num_per_iter = min(
            patch_num, (max_patch_per_gpu // batch_size_per_gpu)
        )  # Ensure at least 1 patch per iter
        patch_iterations = (
            patch_num + max_patch_num_per_iter - 1
        ) // max_patch_num_per_iter
        patch_nums_iter = [
            min(max_patch_num_per_iter, patch_num - i * max_patch_num_per_iter)
            for i in range(patch_iterations)
        ]
        print(
            f"max_patch_num_per_iter is {max_patch_num_per_iter}, patch_iterations is {patch_iterations}, patch_nums_iter is {patch_nums_iter}"
        )
    else:
        patch_nums_iter = [patch_num]

    # Set patch gradient accumulation only for patched diffusion models
    if cfg.model.name in {
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    }:
        if len(patch_nums_iter) > 1:
            if not patching:
                logger0.info(
                    "Patching is not enabled: patch gradient accumulation automatically disabled."
                )
                use_patch_grad_acc = False
            else:
                use_patch_grad_acc = True
        else:
            use_patch_grad_acc = False
    # Automatically disable patch gradient accumulation for non-patched models
    else:
        logger0.info(
            "Training a non-patched model: patch gradient accumulation automatically disabled."
        )
        use_patch_grad_acc = None

    # Instantiate the loss function
    if cfg.model.name in (
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    ):
        loss_fn = ResidualLoss(
            regression_net=regression_net,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name in ["regression", "regression_bias_correction"]:
        loss_fn = RegressionLoss()

    elif cfg.model.name in ["lt_aware_ce_regression"]:
        #loss_fn = RegressionLossCE(prob_channels=prob_channels)
        loss_fn = RegressionLoss()

    # for the bias correction with lead time we keep a normal regression loss
    elif cfg.model.name in ["lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"]:

        # for a huber loss with time consistency
        lambda_time = cfg.training.hp.get("lambda_time", 1.0)
        huber_delta = cfg.training.hp.get("huber_delta", 1.0)
        loss_fn = RegressionHuberLossTimeConsistency(
            lambda_time=lambda_time,
            huber_delta=huber_delta,
        )
    
    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.training.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=True,
    )

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

    # MY CHANGES: combine load of checkpoint
    # --- Load checkpoint (model + optimizer) if present ---
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
            models=model,
            optimizer=optimizer,
            device=dist.device,
        )
        logger0.info(f"Resuming from checkpoint at cur_nimg={cur_nimg}")
    except Exception:
        cur_nimg = 0
        logger0.info("No previous checkpoint found, starting from scratch.")


    # # MY CHANGES --- Epoch / step bookkeeping ---------------------------------
    steps_per_epoch = math.ceil(len(train_ds) / cfg.training.hp.total_batch_size)
    samples_per_epoch = steps_per_epoch * cfg.training.hp.total_batch_size

    if cur_nimg > 0:
        # how many full epochs we already saw, in terms of images
        epoch = cur_nimg // samples_per_epoch
        # how many steps we already took within this epoch
        steps_done_in_epoch = (cur_nimg % samples_per_epoch) // cfg.training.hp.total_batch_size
        train_step = epoch * steps_per_epoch + steps_done_in_epoch
    else:
        epoch = 0
        train_step = 0
        steps_done_in_epoch = 0

    # --- tqdm progress bar starting at the right place ------------
    if dist.rank == 0:
        pbar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch}",
            unit="batch",
            initial=steps_done_in_epoch,   # <--- resume within epoch
        )


    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    # -------  MY CHANGES ------------------------------------------
    # keep track of best validation loss for best-model saving
    best_valid_loss = float('inf')
    # -------

    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False

    # init variables to monitor running mean of average loss since last periodic
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg
    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16

    # enable profiler:
    with cuda_profiler():
        with profiler_emit_nvtx():

            while not done:
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()

                # ---- Curriculum: compute current max allowed lead label ----
                if use_curriculum:
                    # pseudo_epoch is 0-based
                    pseudo_epoch = cur_nimg // samples_per_epoch
                    stage = pseudo_epoch // epochs_per_step  # which curriculum stage we’re in
                    current_max_lead_label = min(
                        max_lead_label,
                        min_lead_label + (stage + 1) * leads_per_step - 1,
                    )
                    if dist.rank == 0 and cur_nimg % 2048 == 0:
                        wandb.log({
                            "train/curriculum_max_lead": int(current_max_lead_label),
                            "train/curriculum_epoch_est": int(pseudo_epoch),
                        }, step=cur_nimg)
                else:
                    current_max_lead_label = None
                # -------

                if cur_nimg - start_nimg == 24 * cfg.training.hp.total_batch_size:
                    logger0.info(f"Starting Profiler at {cur_nimg}")
                    cuda_profiler_start()

                if cur_nimg - start_nimg == 25 * cfg.training.hp.total_batch_size:
                    logger0.info(f"Stopping Profiler at {cur_nimg}")
                    cuda_profiler_stop()

                with nvtx.annotate("Training iteration", color="green"):
                    # Compute & accumulate gradients
                    optimizer.zero_grad(set_to_none=True)
                    loss_accum = 0
                    for n_i in range(num_accumulation_rounds):
                        with nvtx.annotate(
                            f"accumulation round {n_i}", color="Magenta"
                        ):

                            with nvtx.annotate("loading data", color="green"):
                                img_clean, img_lr, *extra = next(dataset_iterator)


                                if dist.rank == 0 and cur_nimg == 0:
                                    logger0.info(
                                        f"[one-shot] use_curriculum={use_curriculum}, "
                                        f"model={cfg.model.name}, "
                                        f"len(extra)={len(extra)}, "
                                        f"current_max_lead_label={current_max_lead_label}"
                                    )


                                # ---- Curriculum filtering (only for lead-aware models) ----
                                if (
                                    use_curriculum
                                    and cfg.model.name in ["lt_aware_ce_regression",
                                                        "lt_aware_ce_regression_bias_correction",
                                                        "lt_aware_FiLM_regression_bias_correction"]
                                    and len(extra) >= 2
                                    and current_max_lead_label is not None
                                ):
                                    horizon_cpu = extra[0].squeeze(1)
                                    time_id_cpu = extra[1].squeeze(1)

                                    # Initialize lead_levels once from the data if needed
                                    if lead_levels is None:
                                        # distinct horizons in increasing order (CPU numpy)
                                        lead_levels = torch.sort(horizon_cpu.unique()).values.cpu().numpy()
                                        if dist.rank == 0:
                                            logger0.info(f"[curriculum] detected lead_levels={lead_levels.tolist()}")

                                    # Which curriculum "stage" we are in (same as before)
                                    # pseudo_epoch = cur_nimg // samples_per_epoch  (already computed above)
                                    stage = pseudo_epoch // epochs_per_step

                                    # How many distinct leads to allow at this stage
                                    n_levels = len(lead_levels)
                                    k = min((stage + 1) * leads_per_step, n_levels)

                                    # Allowed leads for this stage (smallest k horizon labels)
                                    allowed = lead_levels[:k]
                                    allowed_t = torch.from_numpy(allowed).to(horizon_cpu.device)

                                    # Build mask: horizon ∈ allowed
                                    mask = torch.isin(horizon_cpu, allowed_t)

                                    # DEBUG
                                    if dist.rank == 0 and cur_nimg % (10 * cfg.training.hp.total_batch_size) == 0:
                                        uniq_h, cnt_h = horizon_cpu.unique(return_counts=True)
                                        logger0.info(
                                            f"[curri-debug] cur_nimg={cur_nimg}, "
                                            f"stage={stage}, allowed={allowed.tolist()}, "
                                            f"batch leads: "
                                            + ", ".join(f"{int(u.item())}×{int(c.item())}" for u, c in zip(uniq_h, cnt_h))
                                            + f" | mask_true={int(mask.sum().item())}/{mask.numel()}"
                                        )

                                    if mask.any():
                                        img_clean   = img_clean[mask]
                                        img_lr      = img_lr[mask]
                                        horizon_cpu = horizon_cpu[mask]
                                        time_id_cpu = time_id_cpu[mask]

                                        if dist.rank == 0 and cur_nimg % (10 * cfg.training.hp.total_batch_size) == 0:
                                            uniq1, cnt1 = horizon_cpu.unique(return_counts=True)
                                            logger0.info(
                                                f"[curri-debug] AFTER mask: cur_nimg={cur_nimg}, "
                                                f"stage={stage}, allowed={allowed.tolist()}, "
                                                f"kept leads: "
                                                + ", ".join(f"{int(u.item())}×{int(c.item())}" for u, c in zip(uniq1, cnt1))
                                            )
                                    else:
                                        # fallback: keep all
                                        if dist.rank == 0:
                                            logger0.info(
                                                f"[curriculum] mask.empty at cur_nimg={cur_nimg}, "
                                                f"keeping full batch to avoid zero-size"
                                            )
                                        horizon_cpu = extra[0].squeeze(1)
                                        time_id_cpu = extra[1].squeeze(1)

                                    extra = [horizon_cpu.unsqueeze(1), time_id_cpu.unsqueeze(1)]


                                # --------- For Multi-Step fine tuning ---------
                                elif (
                                    use_lead_filter
                                    and cfg.model.name in ["lt_aware_ce_regression",
                                                        "lt_aware_ce_regression_bias_correction",
                                                        "lt_aware_FiLM_regression_bias_correction"]
                                    and len(extra) >= 2
                                ):
                                    horizon_cpu = extra[0].squeeze(1)  # still on CPU
                                    time_id_cpu = extra[1].squeeze(1)

                                    # static mask for this run
                                    mask = (horizon_cpu >= lead_filter_min) & (horizon_cpu <= lead_filter_max)

                                    if mask.any():
                                        img_clean = img_clean[mask]
                                        img_lr    = img_lr[mask]
                                        horizon_cpu = horizon_cpu[mask]
                                        time_id_cpu = time_id_cpu[mask]
                                    else:
                                        # fallback: keep full batch to avoid empty batch
                                        if dist.rank == 0:
                                            logger0.info(
                                                f"[LeadFilter] Empty batch for leads "
                                                f"[{lead_filter_min},{lead_filter_max}] → keeping all samples."
                                            )
                                        horizon_cpu = extra[0].squeeze(1)
                                        time_id_cpu = extra[1].squeeze(1)

                                    extra = [horizon_cpu.unsqueeze(1), time_id_cpu.unsqueeze(1)]

                                    # debug
                                    if dist.rank == 0 and cur_nimg % (10 * cfg.training.hp.total_batch_size) == 0:
                                        uniq, cnt = horizon_cpu.unique(return_counts=True)
                                        lead_str = ", ".join(f"{int(u)}×{int(c)}" for u, c in zip(uniq, cnt))
                                        logger0.info(
                                            f"[lead-filter-debug] cur_nimg={cur_nimg}, "
                                            f"allowed=[{lead_filter_min},{lead_filter_max}], leads in batch: {lead_str}"
    )


                                # ---- Move to device after curriculum filtering ----
                                if use_apex_gn:
                                    img_clean = img_clean.to(
                                        dist.device,
                                        dtype=input_dtype,
                                        non_blocking=True,
                                    ).to(memory_format=torch.channels_last)
                                    img_lr = img_lr.to(
                                        dist.device,
                                        dtype=input_dtype,
                                        non_blocking=True,
                                    ).to(memory_format=torch.channels_last)
                                else:
                                    img_clean = (
                                        img_clean.to(dist.device)
                                        .to(input_dtype)
                                        .contiguous()
                                    )
                                    img_lr = (
                                        img_lr.to(dist.device)
                                        .to(input_dtype)
                                        .contiguous()
                                    )


                            if cfg.model.name in ["lt_aware_ce_regression",
                                                  "lt_aware_ce_regression_bias_correction",
                                                  "lt_aware_FiLM_regression_bias_correction"]:
                                # img_lr is purely your meteorology channels
                                horizon = extra[0].squeeze(1).to(dist.device)  # (B,)
                                time_id = extra[1].squeeze(1).to(dist.device)  # (B,)

                                loss_fn_kwargs = {
                                    "net"            : model,
                                    "img_clean"      : img_clean,
                                    "img_lr"         : img_lr,
                                    "lead_time_label": horizon,
                                    "time_id_label"  : time_id,
                                    "augment_pipe"   : None,
                                }
                            else:
                                loss_fn_kwargs = {
                                    "net": model,
                                    "img_clean": img_clean,
                                    "img_lr": img_lr,
                                    "augment_pipe": None,
                                }

                            # effective batch size after curriculum filtering
                            B_eff = img_clean.shape[0]
                            if B_eff == 0:
                                # extremely unlikely (we avoided empty mask), but guard anyway
                                continue
                
                            # --------# --------# --------# --------# --------# --------# --------



                            # Only pass `use_patch_grad_acc` when the loss actually supports it
                            if cfg.model.name in ("patched_diffusion", "lt_aware_patched_diffusion"):
                                loss_fn_kwargs["use_patch_grad_acc"] = use_patch_grad_acc
                            # --------
                            
                            if use_patch_grad_acc is not None:
                                loss_fn_kwargs[
                                    "use_patch_grad_acc"
                                ] = use_patch_grad_acc


                            if use_patch_grad_acc:
                                loss_fn.y_mean = None

                            for patch_num_per_iter in patch_nums_iter:
                                if patching is not None:
                                    patching.set_patch_num(patch_num_per_iter)
                                    loss_fn_kwargs.update({"patching": patching})
                                with nvtx.annotate(f"loss forward", color="green"):
                                    with torch.autocast(
                                        "cuda", dtype=amp_dtype, enabled=enable_amp
                                    ):

                                        # -----------------------------
                                        # Lambda warmup for time penalty
                                        # -----------------------------
                                        base_lambda = float(cfg.training.hp.get("lambda_time", 0.0))
                                        warmup_epochs = int(cfg.training.hp.get("lambda_time_warmup_epochs", 0))

                                        pseudo_epoch = cur_nimg // samples_per_epoch

                                        if  warmup_epochs > 0:
                                            warm_factor = min(pseudo_epoch / warmup_epochs, 1.0)
                                        else:
                                            warm_factor = 1.0

                                        effective_lambda = base_lambda * warm_factor
                                        loss_fn.lambda_time = effective_lambda

                                        # ----------------------
                                        # Compute training loss
                                        # ----------------------
                                        loss = loss_fn(**loss_fn_kwargs)

                                # normalize by effective batch size (after curriculum filtering)
                                #loss = loss.sum() / batch_size_per_gpu
                                loss = loss.sum() / max(B_eff, 1)

                                loss_accum += loss / num_accumulation_rounds
                                with nvtx.annotate(f"loss backward", color="yellow"):
                                    loss.backward()

                    with nvtx.annotate(f"loss aggregate", color="green"):
                        loss_sum = torch.tensor([loss_accum], device=dist.device)
                        if dist.world_size > 1:
                            torch.distributed.barrier()
                            torch.distributed.all_reduce(
                                loss_sum, op=torch.distributed.ReduceOp.SUM
                            )
                        average_loss = (loss_sum / dist.world_size).cpu().item()

                    # update running mean of average loss since last periodic task
                    average_loss_running_mean += (
                        average_loss - average_loss_running_mean
                    ) / n_average_loss_running_mean
                    n_average_loss_running_mean += 1

                    if dist.rank == 0:
                        writer.add_scalar("training_loss", average_loss, cur_nimg)
                        writer.add_scalar(
                            "training_loss_running_mean",
                            average_loss_running_mean,
                            cur_nimg,
                        )
                        # -------- MY CHANGES ---------
                        # → Log loss on wandb
                        wandb.log({
                            "train/loss": average_loss,
                            "train/loss_running_mean": average_loss_running_mean,
                            }, step=cur_nimg)

                        # in case of RegressionLossTimeConsistency use, log the base mse and the time penalty separated as well
                        if isinstance(loss_fn, (RegressionLossTimeConsistency, RegressionHuberLossTimeConsistency)):
                            wandb.log({
                                "train/base_loss_only": float(loss_fn.last_base_loss),
                                "train/time_penalty_only": float(loss_fn.last_time_penalty),
                            }, step=cur_nimg)

                        # --- Per-lead train MSE logging (current batch) ---
                        if isinstance(loss_fn, (RegressionLossTimeConsistency, RegressionHuberLossTimeConsistency)) and loss_fn.last_mse_per_lead:
                            
                            # 1) Log as a WandB table
                            if cur_nimg % 1000 == 0 and getattr(loss_fn, "last_mse_per_lead", None):
                                # Compact one-row table: columns = L00, L01, ..., values = rounded MSE
                                leads_mse = sorted(loss_fn.last_mse_per_lead.items())  # list of (lead, mse)

                                cols = [f"L{lead:02d}" for lead, _ in leads_mse]
                                vals = [round(float(mse), 4) for _, mse in leads_mse]

                                table = wandb.Table(columns=cols)
                                table.add_data(*vals)

                                wandb.log({"train/per_lead_mse": table}, step=cur_nimg)

                            # 2) Also log per-lead scalars for line plots
                            wandb.log(
                                {
                                    f"train/mse_lead_{int(lead):02d}": float(mse)
                                    for lead, mse in loss_fn.last_mse_per_lead.items()
                                },
                                step=cur_nimg,
                            )



                    ptt = is_time_for_periodic_task(
                        cur_nimg,
                        cfg.training.io.print_progress_freq,
                        done,
                        cfg.training.hp.total_batch_size,
                        dist.rank,
                        rank_0_only=True,
                    )
                    if ptt:
                        # reset running mean of average loss
                        average_loss_running_mean = 0
                        n_average_loss_running_mean = 1

                    # Update weights.
                    with nvtx.annotate("update weights", color="blue"):
                        lr_rampup = (
                            cfg.training.hp.lr_rampup
                        )  # ramp up the learning rate
                        for g in optimizer.param_groups:
                            if lr_rampup > 0:
                                g["lr"] = cfg.training.hp.lr * min(
                                    cur_nimg / lr_rampup, 1
                                )
                            if cur_nimg >= lr_rampup:
                                g["lr"] *= cfg.training.hp.lr_decay ** (
                                    (cur_nimg - lr_rampup)
                                    // cfg.training.hp.lr_decay_rate
                                )
                            current_lr = g["lr"]
                            if dist.rank == 0:
                                writer.add_scalar("learning_rate", current_lr, cur_nimg)
                        handle_and_clip_gradients(
                            model,
                            grad_clip_threshold=cfg.training.hp.grad_clip_threshold,
                        )
                    with nvtx.annotate("optimizer step", color="blue"):
                        optimizer.step()

                    if dist.rank == 0:
                        train_step += 1
                        # advance the bar
                        pbar.update(1)

                        # end of epoch?
                        if train_step % steps_per_epoch == 0:
                            epoch += 1
                            # reset & relabel bar for the next epoch
                            pbar.reset()
                            pbar.set_description(f"Epoch {epoch}")
                            # still log epoch to wandb if you like
                            wandb.log({"epoch": epoch}, step=cur_nimg)

                            # ------- val pass and save best model ------------------------------------------------
                            # ── run one full validation pass ────────────────
                            model.eval()
                            valid_loss = 0.0
                            with torch.no_grad():
                                for _ in range(cfg.training.io.validation_steps):
                                    img_clean_v, img_lr_v, *extra_v = next(validation_dataset_iterator)
                                    img_clean_v = img_clean_v.to(dist.device).to(input_dtype)
                                    img_lr_v    = img_lr_v.to(dist.device).to(input_dtype)

                                    if cfg.model.name in ("lt_aware_ce_regression", "lt_aware_patched_diffusion", "lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"):

                                        horizon_v = extra_v[0].squeeze(1).to(dist.device) # (B,)
                                        time_id_v = extra_v[1].squeeze(1).to(dist.device)   # (B,)

                                        kwargs = {
                                            "net": model,
                                            "img_clean": img_clean_v,
                                            "img_lr": img_lr_v,
                                            "lead_time_label": horizon_v,
                                            "time_id_label": time_id_v,
                                            "augment_pipe": None,
                                        }

                                    else:
                                        
                                        kwargs = {
                                            "net": model,
                                            "img_clean": img_clean_v,
                                            "img_lr": img_lr_v,
                                            "augment_pipe": None,
                                        }

                                    loss_v = loss_fn(**kwargs).sum().item() / batch_size_per_gpu
                                    valid_loss += loss_v
                            valid_loss /= cfg.training.io.validation_steps
                    
                            # if distributed, average across ranks
                            loss_tensor = torch.tensor([valid_loss], device=dist.device)
                            # only do an all_reduce if we really are in DDP (world_size>1)
                            if dist.world_size > 1 and torch.distributed.is_initialized():
                                torch.distributed.barrier()
                                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                            average_valid_loss = (loss_tensor / dist.world_size).item()
                    
                            # ── save best model only ────────────────────────
                            if dist.rank == 0:
                                logger0.info(
                                    f"Epoch {epoch} validation loss: {average_valid_loss:.4f} "
                                    f"(best {best_valid_loss:.4f})"
                                )
                                if average_valid_loss < best_valid_loss:
                                    best_valid_loss = average_valid_loss

                                    # IMPORTANT: use cur_nimg as index, not epoch
                                    save_checkpoint(
                                        path=checkpoint_dir,
                                        models=model,
                                        optimizer=optimizer,
                                        epoch=cur_nimg,      # <--- use global progress
                                    )
                                    logger0.info(
                                        f"→ New best model saved at cur_nimg={cur_nimg} "
                                        f"(loss = {best_valid_loss:.4f})"
                                    )                            
                                    

                            model.train()
                            # -------

                   # # -------

                    cur_nimg += cfg.training.hp.total_batch_size
                    done = cur_nimg >= cfg.training.hp.training_duration


                    # ----------  add visualizations of regression results in wandb in TRAIN set ---------
                    if dist.rank == 0 and cfg.model.name in ("regression", "lt_aware_ce_regression", "regression_bias_correction", "lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"):

                        # TO VIZ ONLY EVERY 5000 steps otherwise it makes the training slow!
                        if cur_nimg % 5000 == 0:

                            if cfg.model.name in ("lt_aware_ce_regression", "lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"):
                                # grab TRAIN examples (dataset always returns 3 items)
                                img_clean_train, img_lr_train, lead_time, time_id = next(dataset_iterator)
                                img_clean_train = img_clean_train.to(dist.device).to(input_dtype)
                                img_lr_train    = img_lr_train.to(dist.device).to(input_dtype)
                                lead_time       = lead_time.to(dist.device).contiguous()  # shape (B, 1)
                                time_id         = time_id.to(dist.device).contiguous()  # shape (B, 1)

                            if cfg.model.name in ("regression", "regression_bias_correction"):
                                # grab TRAIN examples (dataset always returns 2 items)
                                img_clean_train, img_lr_train = next(dataset_iterator)
                                img_clean_train = img_clean_train.to(dist.device).to(input_dtype)
                                img_lr_train    = img_lr_train.to(dist.device).to(input_dtype)


                            # build the zero‐init input and run
                            B, _, H, W = img_lr_train.shape
                            c_hr = regression_net.img_out_channels
                            x0 = torch.zeros((B, c_hr, H, W), device=dist.device, dtype=input_dtype)
                            
                            with torch.no_grad():
                                # account for the lead time in these versions
                                if cfg.model.name in ("lt_aware_ce_regression", "lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"):
                                    pred_norm = regression_net(x0, img_lr_train, lead_time_label=lead_time)
                                if cfg.model.name in ("regression", "regression_bias_correction"):
                                    pred_norm = regression_net(x0, img_lr_train)
    
    
                            # --- UNDO MIN–MAX SCALING INSTEAD OF Z-SCORE ---
                            # pull in your saved mins/maxs per channel
                            out_min = torch.from_numpy(dataset_for_stats.output_min[:, None, None]).to(dist.device)
                            out_max = torch.from_numpy(dataset_for_stats.output_max[:, None, None]).to(dist.device)
                            
                            # invert (x - min)/(max - min)  →  x = norm*(max-min) + min
                            range_out = out_max - out_min
                            pred      = (pred_norm * range_out + out_min)
                            gt        = (img_clean_train * range_out + out_min)
                            low_res   = (img_lr_train     * range_out + out_min)
    
                            # if we log‐transformed precip in the dataset, invert it here
                            if getattr(cfg.dataset, "log_precip", False):
                                # high‐res precipitation is channel 0
                                pred     = pred.clone()
                                gt       = gt.clone()
                                pred[:, 0, :, :] = torch.expm1(pred[:, 0, :, :])
                                gt[:,   0, :, :] = torch.expm1(gt[:,   0, :, :])
                            
                                # low‐res precipitation is channel 0
                                low_res  = low_res.clone()
                                low_res[:, 0, :, :] = torch.expm1(low_res[:, 0, :, :])
                            
                            # move to CPU / numpy
                            pred    = pred.cpu().numpy()
                            gt      = gt.cpu().numpy()
                            low_res = low_res.cpu().numpy()
                        
                            # pick channel slices: high-res precip=0, low-res precip=0
                            gt_np   = gt[:,   0, :, :]
                            pred_np = pred[:, 0, :, :]
                            low_np  = low_res[:, 0, :, :]
                        
                            # shared color limits
                            #vmin = float(min(gt_np.min(), pred_np.min(), low_np.min()))
                            #vmax = float(max(gt_np.max(), pred_np.max(), low_np.max()))
                        
                            # custom colormap
                            config      = ColormapConfig()
                            cmap_custom = config.cmap
                            norm_custom = config.norm
                            clevs       = config.clevs
                        
                            # number of examples to show
                            N = 2
    
                            # --- before the plotting loop: get real coords ---
                            lon_vals = base_dataset.longitude()  # array of length W
                            lat_vals = base_dataset.latitude()   # array of length H
                            
                            # min/max in degrees
                            lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
                            lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
                            
                            # Build extent tuple once
                            extent = [lon_min, lon_max, lat_min, lat_max]
    
                            # build cartopy figure: 3 rows × N cols
                            fig, axes = plt.subplots(
                                3, N,
                                figsize=(3 * N, 9),
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                constrained_layout=False
                            )
                            # adjust for shared colorbar on right
                            fig.subplots_adjust(right=0.88, wspace=0.05, hspace=0.03)
                        
                            for row, data_arr in enumerate((gt_np, pred_np, low_np)):
                                title_row = ("GT", "Pred", "Low-res")[row]
                                for i in range(N):
                                    ax = axes[row, i]
                                    im = ax.imshow(
                                        data_arr[i],
                                        origin='lower',
                                        extent=extent,
                                        cmap=cmap_custom,
                                        norm=norm_custom,
                                        #vmin=vmin,
                                        #vmax=vmax,
                                        transform=ccrs.PlateCarree()
                                    )
                                    # draw CH border
                                    ax.add_geometries(
                                        [switz_geom],
                                        crs=ccrs.PlateCarree(),
                                        facecolor='none',
                                        edgecolor='black',
                                        linewidth=1.0
                                    )
                                    ax.set_title(f"{title_row} #{i+1}", fontsize=10)
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                        
                            # single shared colorbar
                            cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.8])
                            fig.colorbar(
                                im, cax=cbar_ax,
                                boundaries=clevs,
                                ticks=clevs,
                                extend='both',
                                label="6-h Cumulative Precipitation (mm/hr)"
                            )
                        
                            wandb.log({"train/examples_grid": wandb.Image(fig)}, step=cur_nimg)
                            plt.close(fig)                                    


                
                with nvtx.annotate("validation", color="red"):
                    # Validation
                    if validation_dataset_iterator is not None:
                        valid_loss_accum = 0
                        if is_time_for_periodic_task(
                            cur_nimg,
                            cfg.training.io.validation_freq,
                            done,
                            cfg.training.hp.total_batch_size,
                            dist.rank,
                        ):
                            with torch.no_grad():
                                for _ in range(cfg.training.io.validation_steps):
                                    (
                                        img_clean_valid,
                                        img_lr_valid,
                                        *extra_valid
                                    ) = next(validation_dataset_iterator)

                                    if use_apex_gn:
                                        img_clean_valid = img_clean_valid.to(
                                            dist.device,
                                            dtype=input_dtype,
                                            non_blocking=True,
                                        ).to(memory_format=torch.channels_last)
                                        img_lr_valid = img_lr_valid.to(
                                            dist.device,
                                            dtype=input_dtype,
                                            non_blocking=True,
                                        ).to(memory_format=torch.channels_last)

                                    else:
                                        img_clean_valid = (
                                            img_clean_valid.to(dist.device)
                                            .to(input_dtype)
                                            .contiguous()
                                        )
                                        img_lr_valid = (
                                            img_lr_valid.to(dist.device)
                                            .to(input_dtype)
                                            .contiguous()
                                        )

                                    # Only attach lead_time_label for the lead‐time‐aware models
                                    if cfg.model.name in ("lt_aware_ce_regression", "lt_aware_patched_diffusion", "lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"):

                                        lead_valid = extra_valid[0].squeeze(1).to(dist.device)
                                        time_id_valid = extra_valid[1].squeeze(1).to(dist.device)

                                        loss_valid_kwargs = {
                                            "net": model,
                                            "img_clean": img_clean_valid,
                                            "img_lr": img_lr_valid,
                                            "lead_time_label": lead_valid,
                                            "time_id_label": time_id_valid,
                                            "augment_pipe": None,
                                        }
                                    else:
                                        loss_valid_kwargs = {
                                            "net": model,
                                            "img_clean": img_clean_valid,
                                            "img_lr": img_lr_valid,
                                            "augment_pipe": None,
                                            }

                                    # ---------------------------------------

                                    # Only pass `use_patch_grad_acc` when the loss actually supports it
                                    if cfg.model.name in ("patched_diffusion", "lt_aware_patched_diffusion"):
                                        loss_valid_kwargs["use_patch_grad_acc"] = use_patch_grad_acc
                                    
                                    if use_patch_grad_acc is not None:
                                        loss_valid_kwargs[
                                            "use_patch_grad_acc"
                                        ] = use_patch_grad_acc


                                    if use_patch_grad_acc:
                                        loss_fn.y_mean = None

                                    for patch_num_per_iter in patch_nums_iter:
                                        if patching is not None:
                                            patching.set_patch_num(patch_num_per_iter)
                                            loss_valid_kwargs.update(
                                                {"patching": patching}
                                            )
                                        with torch.autocast(
                                            "cuda", dtype=amp_dtype, enabled=enable_amp
                                        ):
                                            loss_valid = loss_fn(**loss_valid_kwargs)

                                        loss_valid = (
                                            (loss_valid.sum() / batch_size_per_gpu)
                                            .cpu()
                                            .item()
                                        )
                                        valid_loss_accum += (
                                            loss_valid
                                            / cfg.training.io.validation_steps
                                        )
                                valid_loss_sum = torch.tensor(
                                    [valid_loss_accum], device=dist.device
                                )
                                if dist.world_size > 1:
                                    torch.distributed.barrier()
                                    torch.distributed.all_reduce(
                                        valid_loss_sum,
                                        op=torch.distributed.ReduceOp.SUM,
                                    )
                                average_valid_loss = valid_loss_sum / dist.world_size
                                if dist.rank == 0:
                                    writer.add_scalar(
                                        "validation_loss", average_valid_loss, cur_nimg
                                    )

                                    # → Log loss in wandb
                                    wandb.log({
                                        "val/loss": average_valid_loss,
                                            }, step=cur_nimg)

                                    # log time penalty separated from main loss with RegressionLossTimeConsistency loss
                                    if isinstance(loss_fn, (RegressionLossTimeConsistency, RegressionHuberLossTimeConsistency)):

                                        wandb.log(
                                            {
                                                "val/base_loss": float(loss_fn.last_base_loss),
                                                "val/time_penalty": float(loss_fn.last_time_penalty),
                                            },
                                            step=cur_nimg,
                                        )

                                    # log per lead mse, with RegressionLossTimeConsistency loss
                                    if cur_nimg % 1000 == 0 and getattr(loss_fn, "last_mse_per_lead", None):
                                        leads_mse = sorted(loss_fn.last_mse_per_lead.items())

                                        cols = [f"L{lead:02d}" for lead, _ in leads_mse]
                                        vals = [round(float(mse), 4) for _, mse in leads_mse]

                                        table = wandb.Table(columns=cols)
                                        table.add_data(*vals)

                                        wandb.log({"val/per_lead_mse": table}, step=cur_nimg)

                                    if isinstance(loss_fn, (RegressionLossTimeConsistency, RegressionHuberLossTimeConsistency)):
                                        # 2) Also log per-lead scalars for line plots
                                        wandb.log(
                                            {
                                                f"val/mse_lead_{int(lead):02d}": float(mse)
                                                for lead, mse in loss_fn.last_mse_per_lead.items()
                                            },
                                            step=cur_nimg,
                                        )



                                # ---------- add visualizations of regression results in wandb, in val set ---------
                                if dist.rank == 0 and cfg.model.name in ("regression", "lt_aware_ce_regression", "regression_bias_correction", "lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"):

                                    # TO VIZ ONLY EVERY 5000 steps otherwise it makes the training slow!
                                    if cur_nimg % 5000 == 0:

                                        if cfg.model.name in ("lt_aware_ce_regression", "lt_aware_ce_regression_bias_correction", "lt_aware_FiLM_regression_bias_correction"):
                                            # grab VAL examples
                                            img_clean_val, img_lr_val, lead_time_val, time_id = next(validation_dataset_iterator)
                                            img_clean_val = img_clean_val.to(dist.device).to(input_dtype)
                                            img_lr_val    = img_lr_val.to(dist.device).to(input_dtype)
                                            lead_time_val = lead_time_val.to(dist.device).contiguous()  # shape (B, 1)   
                                            time_id       = time_id.to(dist.device).contiguous()  # shape (B, 1)   


                                        if cfg.model.name in ("regression", "regression_bias_correction"):         
                                            # grab VAL examples
                                            img_clean_val, img_lr_val = next(validation_dataset_iterator)
                                            img_clean_val = img_clean_val.to(dist.device).to(input_dtype)
                                            img_lr_val    = img_lr_val.to(dist.device).to(input_dtype)


                                        B, _, H, W = img_lr_val.shape
                                        c_hr = regression_net.img_out_channels
                                        x0 = torch.zeros((B, c_hr, H, W), device=dist.device, dtype=input_dtype)
                                        
                                        with torch.no_grad():
                                            # account for lead time in these versions
                                            if cfg.model.name in ("lt_aware_ce_regression", "lt_aware_ce_regression_bias_regression", "lt_aware_FiLM_regression_bias_correction"):
                                                pred_norm = regression_net(x0, img_lr_val, lead_time_label=lead_time_val)
                                            if cfg.model.name in ("regression", "regression_bias_correction"):
                                                pred_norm = regression_net(x0, img_lr_val)
    
                                        # --- UNDO MIN–MAX SCALING INSTEAD OF Z-SCORE ---
                                        # pull in your saved mins/maxs per channel
                                        out_min = torch.from_numpy(dataset_for_stats.output_min[:, None, None]).to(dist.device)
                                        out_max = torch.from_numpy(dataset_for_stats.output_max[:, None, None]).to(dist.device)
                                        
                                        # invert (x - min)/(max - min)  →  x = norm*(max-min) + min
                                        range_out = out_max - out_min
                                        pred      = (pred_norm * range_out + out_min)
                                        gt        = (img_clean_val * range_out + out_min)
                                        low_res   = (img_lr_val     * range_out + out_min)
    
                                        # if we log‐transformed precip in the dataset, invert it here
                                        if getattr(cfg.dataset, "log_precip", False):
                                            # high‐res precipitation is channel 0
                                            pred     = pred.clone()
                                            gt       = gt.clone()
                                            pred[:, 0, :, :] = torch.expm1(pred[:, 0, :, :])
                                            gt[:,   0, :, :] = torch.expm1(gt[:,   0, :, :])
                                        
                                            # low‐res precipitation is channel 1
                                            low_res  = low_res.clone()
                                            low_res[:, 0, :, :] = torch.expm1(low_res[:, 0, :, :])
                                        
                                        # move to CPU / numpy
                                        pred    = pred.cpu().numpy()
                                        gt      = gt.cpu().numpy()
                                        low_res = low_res.cpu().numpy()
                                    
                                        # pick channel slices: high-res precip=0, low-res precip=0
                                        gt_np   = gt[:,   0, :, :]
                                        pred_np = pred[:, 0, :, :]
                                        low_np  = low_res[:, 0, :, :]
                                    
                                        # shared color limits
                                        #vmin = float(min(gt_np.min(), pred_np.min(), low_np.min()))
                                        #vmax = float(max(gt_np.max(), pred_np.max(), low_np.max()))
                                    
                                        # custom colormap
                                        config      = ColormapConfig()
                                        cmap_custom = config.cmap
                                        norm_custom = config.norm
                                        clevs       = config.clevs
                                    
                                        # number of examples to show
                                        N = 2
    
                                        # --- before the plotting loop: get real coords ---
                                        lon_vals = base_dataset.longitude()  # array of length W
                                        lat_vals = base_dataset.latitude()   # array of length H
                                        
                                        # min/max in degrees
                                        lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
                                        lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
                                        
                                        # Build extent tuple once
                                        extent = [lon_min, lon_max, lat_min, lat_max]
    
                                        # build cartopy figure: 3 rows × N cols
                                        fig, axes = plt.subplots(
                                            3, N,
                                            figsize=(3 * N, 9),
                                            subplot_kw={'projection': ccrs.PlateCarree()},
                                            constrained_layout=False
                                        )
                                        # adjust for shared colorbar on right
                                        fig.subplots_adjust(right=0.88, wspace=0.05, hspace=0.03)
                                    
                                        for row, data_arr in enumerate((gt_np, pred_np, low_np)):
                                            title_row = ("GT", "Pred", "Low-res")[row]
                                            for i in range(N):
                                                ax = axes[row, i]
                                                im = ax.imshow(
                                                    data_arr[i],
                                                    origin='lower',
                                                    extent=extent,
                                                    cmap=cmap_custom,
                                                    norm=norm_custom,
                                                    #vmin=vmin,
                                                    #vmax=vmax,
                                                    transform=ccrs.PlateCarree()
                                                )
                                                # draw CH border
                                                ax.add_geometries(
                                                    [switz_geom],
                                                    crs=ccrs.PlateCarree(),
                                                    facecolor='none',
                                                    edgecolor='black',
                                                    linewidth=1.0
                                                )
                                                ax.set_title(f"{title_row} #{i+1}", fontsize=10)
                                                ax.set_xticks([])
                                                ax.set_yticks([])
                                    
                                        # single shared colorbar
                                        cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.8])
                                        fig.colorbar(
                                            im, cax=cbar_ax,
                                            boundaries=clevs,
                                            ticks=clevs,
                                            extend='both',
                                            label="6-h Cumulative Precipitation (mm/hr)"
                                        )
                                    
                                        wandb.log({"val/examples_grid": wandb.Image(fig)}, step=cur_nimg)
                                        plt.close(fig)                                    


            # Retain only the recent n checkpoints, if desired
            if cfg.training.io.save_n_recent_checkpoints > 0:
                for suffix in [".mdlus", ".pt"]:
                    ckpts = checkpoint_list(checkpoint_dir, suffix=suffix)
                    while len(ckpts) > cfg.training.io.save_n_recent_checkpoints:
                        os.remove(os.path.join(checkpoint_dir, ckpts[0]))
                        ckpts = ckpts[1:]


    # Save the best validation loss for Optuna (or future use).
    # Only rank 0 writes to disk.
    if dist.rank == 0:
        try:
            metric_path = os.path.join(checkpoint_dir, "best_valid_loss.txt")
            with open(metric_path, "w") as f:
                f.write(f"{best_valid_loss:.8f}\n")
            logger0.info(f"Saved best_valid_loss={best_valid_loss:.4f} to {metric_path}")
        except Exception as e:
            logger0.info(f"Could not save best_valid_loss.txt: {e}")
    # ------------------------------------------------------------------


    # Done.
    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
