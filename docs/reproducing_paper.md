# Reproducing the paper

This document maps the main tables and figures in the paper to the
specific training configurations, generation runs, and evaluation
notebooks needed to reproduce them.

> **GPU budget**: a full reproduction requires roughly 5–10 GPU-days on a
> recent NVIDIA GPU (H100 / GH200 / A100). The most expensive parts are
> the Step-2 regression and diffusion training (~2 days each); a single
> Step-1 model takes ~1 day. Generation and evaluation are fast (hours).

---

## Prerequisites

- The repository installed (see [README](../README.md))
- Data prepared via `notebooks/01_data_preparation.ipynb` (see [docs/data_preparation.md](data_preparation.md))
- A symlink or copy of your data at `./data/`

All commands below should be run from the repository root.

---

## 1. Step 1: bias-correction models

The paper compares several Step-1 variants. The headline model is the
multi-lead FiLM-conditioned U-Net with λ=2 temporal regularization.

### 1.1 Main model — `UNet, λ=2` (final published model)

```bash
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml
```

Default behaviour: trains on the 0–24 h horizon range with the rolling
weekly split, λ_time=2, no curriculum.

### 1.2 Variant — `UNet, λ=0` (no temporal regularization)

Same config as above, but override λ_time at the command line:

```bash
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml \
    ++training.hp.lambda_time=0
```

### 1.3 Variant — `UNet MultiFineTuned`

A 5-stage fine-tuning schedule that progressively specializes on long
lead times. Each stage starts from the previous stage's checkpoint,
with the lead range, learning rate, and λ_time updated per stage.
See the paper's Appendix for the schedule rationale; the commands are:

```bash
# Stage 1: short leads (≤1 day), no temporal regularization
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml \
    ++training.hp.lead_filter.enabled=True \
    ++training.hp.lead_filter.min_lead=0 \
    ++training.hp.lead_filter.max_lead=7 \
    ++training.hp.lr=2e-4 \
    ++training.hp.lambda_time=0

# Stage 2: medium leads (~2–4 days)
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml \
    ++training.hp.lead_filter.enabled=True \
    ++training.hp.lead_filter.min_lead=8 \
    ++training.hp.lead_filter.max_lead=15 \
    ++training.hp.lr=2e-4 \
    ++training.hp.lambda_time=0

# Stage 3: full range, time-coherence on
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml \
    ++training.hp.lead_filter.enabled=True \
    ++training.hp.lead_filter.min_lead=0 \
    ++training.hp.lead_filter.max_lead=23 \
    ++training.hp.lr=2e-5 \
    ++training.hp.lambda_time=1.0 \
    ++training.hp.lambda_time_warmup_epochs=10

# Stage 4: long leads (~3–6 days)
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml \
    ++training.hp.lead_filter.enabled=True \
    ++training.hp.lead_filter.min_lead=12 \
    ++training.hp.lead_filter.max_lead=23 \
    ++training.hp.lr=1e-5 \
    ++training.hp.lambda_time=0.3 \
    ++training.hp.lambda_time_warmup_epochs=3

# Stage 5: very long leads (≥4 days)
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml \
    ++training.hp.lead_filter.enabled=True \
    ++training.hp.lead_filter.min_lead=16 \
    ++training.hp.lead_filter.max_lead=23 \
    ++training.hp.lr=2e-5 \
    ++training.hp.lambda_time=0.0 \
    ++training.hp.lambda_time_warmup_epochs=3
```

Between stages, point the training script to the previous stage's checkpoint
via a Hydra override (or by editing the config). See your existing training
scripts for the exact mechanism — typically a `checkpoint_dir` override
pointing to a directory containing a checkpoint the trainer will warm-start from.

### 1.4 Baseline — Single-lead specialists (UNet Single 6h / 3day / 6day)

Train one model per lead horizon, using the single-horizon dataset variant.
The `config_training_aifs_single_regression_STEP1.yaml` config defaults to
one of the single-horizon files; switch between horizons by overriding the
dataset paths:

```bash
# UNet Single 6h
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_regression_STEP1.yaml \
    ++dataset.stats_path=./data/stats/stats_aifs_single_6h_biasCorrection.json \
    ++dataset.low_res_file=./data/AIFS_Single_pred_concat_2019_2023_SingleHori_6h_NoNeg_andLowResPrecipWetEnough_gamma.nc \
    ++dataset.high_res_file=./data/AIFS_Single_pred_concat_2019_2023_SingleHori_6h_NoNeg_andLowResPrecipWetEnough_gamma.nc

# UNet Single 3day — override with the 3day stats + file
# UNet Single 6day — override with the 6day stats + file
```

### 1.5 Baseline — Strategy 2 ablation (no Step 1, direct multivariate regression)

A regression U-Net trained directly on the full multivariate AIFS state to
high-res CombiPrecip, bypassing the explicit bias-correction stage.

```bash
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_regression_DirectStrat2.yaml
```

### 1.6 Baseline — B4 per-cell linear regression

This baseline does not use neural networks; it is computed inline in the
evaluation notebook from precomputed bias-corrected fields. See cells in
`notebooks/02_step1_evaluation.ipynb` that compute the B4 baseline from
the multivariate AIFS inputs.

---

## 2. Step 2: super-resolution

Trained once and used as the common downstream stage for all Step-1
variants in the comparison tables.

### 2.1 Regression U-Net (deterministic component)

```bash
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_regression_STEP2.yaml
```

### 2.2 Diffusion U-Net (stochastic component)

Requires the regression checkpoint from the previous step. After
regression training, find the latest `.mdlus` file under
`./outputs/<step2_regression_run>/checkpoints/` and pass its path:

```bash
python scripts/train_weekly.py \
    --config-name=config_training_aifs_single_diffusion.yaml \
    ++training.io.regression_checkpoint_path=./outputs/<step2_regression_run>/checkpoints/UNet.0.<iter>.mdlus
```

---

## 3. Generation

Once Step-1 and Step-2 models are trained, generate ensemble samples for
the validation set.

### 3.1 Full pipeline inference (recommended)

The full pipeline takes raw AIFS forecasts as input and runs Step 1 +
Step 2 together. This is the canonical inference path:

```bash
python scripts/inference_full_pipeline.py \
    --config-name=config_inference_full_pipeline.yaml
```

Override paths and parameters as needed via Hydra:

```bash
python scripts/inference_full_pipeline.py \
    --config-name=config_inference_full_pipeline.yaml \
    inference.checkpoints.bc=./outputs/<step1_run>/checkpoints/UNetResidualCorrectorClampedFiLM.0.<iter>.mdlus \
    inference.checkpoints.reg=./outputs/<step2_reg_run>/checkpoints/UNet.0.<iter>.mdlus \
    inference.checkpoints.dif=./outputs/<step2_dif_run>/checkpoints/EDMPrecondSuperResolution.0.<iter>.mdlus \
    inference.output.file=./outputs/generations/UNet_lam2/samples.nc \
    inference.output.save_bc=True
```

The `save_bc=True` option writes the intermediate bias-corrected output
alongside the final super-resolved samples; the evaluation notebooks use
this to compute "before super-resolution" metrics.

### 3.2 Super-resolution only (Step 2 only, from pre-computed bias-corrected inputs)

If you already have bias-corrected fields on disk (e.g., to compare
multiple Step-1 variants against a single Step-2 model), run only the
super-resolution stage:

```bash
python scripts/generate_super_resolution.py \
    --config-name=config_generate_aifs_single.yaml
```

For each Step-1 variant you want to compare, update the `low_res_file` in
the generation config (or pass via Hydra override) to point to the
corresponding bias-corrected file.

---

## 4. Evaluation

All evaluation is done in the Jupyter notebooks.

### 4.1 Step 1 (bias correction) metrics

`notebooks/02_step1_evaluation.ipynb`

Computes CSI, FSS, MSE on the low-res bias-corrected fields for each
Step-1 variant. Produces the Step 1 metric tables and per-lead metric
heatmaps shown in the paper.

Required inputs: trained Step-1 checkpoints (see Section 1 above) plus
the corresponding bias-corrected `.nc` files produced by inference.

### 4.2 Step 2 (super-resolution + end-to-end) metrics

`notebooks/03_step2_evaluation.ipynb`

Computes CRPS, FSS, CSI, MSE, eS / eA / eL spectral metrics on the
super-resolved ensemble outputs. Produces the end-to-end metric tables,
CRPS-by-lead plots, spectral comparison figures, and sample
visualizations shown in the paper.

Required inputs: generated `.nc` files from Section 3 above, plus the
high-res CombiPrecip target.

### 4.3 Supplemental sanity checks

`notebooks/04_supplemental_sanity_checks.ipynb`

Produces the supplemental-material sanity-check plots:
- Skill by hour of day (using the 6h/18h test set)
- Spatial autocorrelation function of CombiPrecip (motivates the patch size)

---

## 5. Notes on the published checkpoints

The exact checkpoints used to produce the paper's numbers are not
redistributed (see README — *Model checkpoints*). For academic
reproducibility access, please contact the corresponding author.

Reproducing from scratch should yield numbers within the run-to-run
variability of the training procedure (different random seeds, machine
precision, etc.). The high-level conclusions of the paper are robust
to this variability, but individual table entries may differ by a few
percent.

---

## Troubleshooting

**Training fails with a checkpoint-loading error.** Verify your
installed PhysicsNeMo version is `1.2.0` (`pip show nvidia-physicsnemo`).
Newer versions may have an incompatible model serialization format.

**Out-of-memory errors during Step-2 training.** Reduce
`training.hp.patch_num` or `training.hp.batch_size_per_gpu` in the
config (or via Hydra override). Step-2 diffusion is the most memory-
intensive component due to patch-based training.

**Generation produces all-zero or all-NaN outputs.** Check that the
normalization statistics file (`bc_stats` / `sr_stats` in the inference
config) matches the dataset variant the model was trained on. A
mismatch between training and inference normalization is a common cause
of degenerate outputs.

**The Hydra `${hydra:run.dir}` path resolves to something unexpected.**
This is set by the `hydra.run.dir` key in each config. By default we
use `./outputs/<job-name>/`; you can override on the command line with
`hydra.run.dir=./my_custom_dir`.

**My config filename is too long to type comfortably.** Hydra lets you
shorten config invocations by setting up a shell alias:

```bash
alias swa-step1='python scripts/train_weekly.py --config-name=config_training_aifs_single_lt_aware_FiLM_Time_regression_STEP1.yaml'
alias swa-step2-reg='python scripts/train_weekly.py --config-name=config_training_aifs_single_regression_STEP2.yaml'
alias swa-step2-dif='python scripts/train_weekly.py --config-name=config_training_aifs_single_diffusion.yaml'
alias swa-infer='python scripts/inference_full_pipeline.py --config-name=config_inference_full_pipeline.yaml'
```

Then just `swa-step1`, `swa-step2-reg`, etc.
