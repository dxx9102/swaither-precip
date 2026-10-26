# Hourly wind forecasting and CorrDiff super-resolution

This pipeline is independent of the original precipitation workflow. It keeps
the precipitation code and configurations intact while adding dedicated wind
datasets, models, training entry points, inference, and evaluation.

## Pipeline

1. A deterministic PhysicsNeMo U-Net uses the latest six hourly low-resolution
   `u10/v10` fields, two low-resolution static fields, and four coordinate
   channels to predict normalized persistence residuals for leads 1-72.
2. The residual at each lead is relative to the final history observation at
   `t`; inference reconstructs all 72 absolute hourly fields in one forward
   pass per batch without feeding predictions back into the history.
3. The frozen Step-1 forecast library is paired by valid time with ICON
   high-resolution targets.
4. The existing CorrDiff regression U-Net and conditional EDM residual model
   are trained on two wind channels. The diffusion model continues to consume
   `hr_mean_conditioning`, `RegressionLoss`, `ResidualLoss`, and the original
   EDM sampler logic.

Year splits are `2019-2022` train, `2023` validation, and `2024` test. Robust
0.5%/99.5% normalization statistics are computed from training years only.

## Environment and storage

The project uses the Anaconda environment `swaither-precip`.

Before launching, source the storage setup:

```bash
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
cd /path/to/swaither-precip
source scripts/wind_env.sh
```

The Python entry points also call `configure_storage()`. Checkpoints, statistics,
conditions, inference output, TensorBoard/W&B data, and caches are placed below
the project root. Prepared wind data defaults to `../../data/uv10/` and is read
only. The data directory is configured by `paths.raw_data_root` in every wind
YAML file; for example, override it at launch with
`paths.raw_data_root=/path/to/uv10`.

Verify GPU availability before production training:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 1. Training statistics

Compute absolute-target Step-1 statistics from the 2019-2022 low-resolution
records. The statistics script deliberately ignores the residual output of the
training dataset and summarizes the physical `u10/v10` targets instead:

```bash
python scripts/compute_wind_stats.py \
  --config-name config_compute_wind_stats.yaml
```

Train Step 1:

```bash
python scripts/train_wind_step1.py \
  --config-name config_training_wind_step1_unet.yaml
```

The best checkpoint is:

```text
outputs/wind_step1_unet_72h/checkpoints/WindForecastUNet.best.mdlus
```

The 72-hour residual model uses a new output contract and is intentionally not
compatible with the old one-step direct-regression checkpoints.

## 2. Frozen Step-1 conditions

Generate one-to-72-hour conditions for all initial times that have a complete
same-year six-hour history:

```bash
python scripts/export_wind_forecasts.py \
  --config-name config_export_wind_forecasts.yaml
```

The default output is:

```text
data/wind_generated/step1_conditions_2019_2024_72h.nc
```

It has dimensions `(initial_time, lead_hour, lat, lon)`, two variables
`u10/v10` in `m s-1`, a `valid_time` coordinate, and `lead_hour=1..72`.
Step-2 never updates the Step-1 model. The current Step-2 configurations still
consume the existing one-to-six-hour condition file and are unchanged by this
Step-1-only migration.

Compute Step-2 statistics using the generated conditions and ICON targets from
2019-2022 only:

```bash
python scripts/compute_wind_stats.py \
  --config-name config_compute_wind_stats_step2.yaml
```

## 3. CorrDiff training

On the Slurm cluster, the two training stages can also be submitted directly
with `yhbatch`. This cluster only accepts scheduler options on the `yhbatch`
command line; do not add `#SBATCH` directives to the job scripts. The forecast
job trains Step 1:

```bash
yhbatch -p a800x -G 1 scripts/yhbatch_train_wind_forecast.sh

# Any Hydra override is forwarded to train_wind_step1.py.
yhbatch -p a800x -G 1 scripts/yhbatch_train_wind_forecast.sh \
  paths.raw_data_root=/path/to/uv10
```

After Step 1 finishes, submit Step 2. The super-resolution job exports frozen
Step-1 conditions, recomputes the Step-2 statistics, trains regression, then
finds the newest regression checkpoint and passes it to diffusion:

```bash
yhbatch -p a800x -G 1 scripts/yhbatch_train_wind_super_resolution.sh

# Use a non-default Step-1 checkpoint or data location.
WIND_STEP1_CHECKPOINT=/path/to/WindForecastUNet.best.mdlus \
WIND_RAW_DATA_ROOT=/path/to/uv10 \
yhbatch -p a800x -G 1 scripts/yhbatch_train_wind_super_resolution.sh
```

The scripts first `cd` to the absolute repository path
`/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip`, which is also
available through `~/HDD_POOL`. All subsequent project paths remain relative
to that directory. Keep the absolute path synchronized if the repository
moves. Required and optional scheduler resources are passed directly to
`yhbatch`, for example `-p a800x -G 1 -c 16 -t 48:00:00`.

Optional Step-2 path overrides are `WIND_STEP1_STATS`,
`WIND_CONDITIONS_FILE`, `WIND_STEP2_STATS`, `WIND_REGRESSION_RUN_DIR`, and
`WIND_DIFFUSION_RUN_DIR`. Existing regression checkpoints are reused by the
training entry point automatically. When changing the Step-1 checkpoint or
input data, set both run directories to new paths so that regression and
diffusion do not resume stale models.

Train the deterministic regression component:

```bash
python scripts/train_yearly.py \
  --config-name config_training_wind_step2_regression.yaml
```

PhysicsNeMo writes model files below
`outputs/wind_step2_regression/checkpoints_regression/`. Set
`training.io.regression_checkpoint_path` in the diffusion configuration to the
chosen regression checkpoint, then train diffusion:

```bash
python scripts/train_yearly.py \
  --config-name config_training_wind_step2_diffusion.yaml \
  training.io.regression_checkpoint_path=/absolute/path/to/UNet.0.ITER.mdlus
```

The defaults use BF16 AMP. Step 1 uses batch 32; CorrDiff regression uses batch
8; diffusion uses micro-batch 2 with four gradient-accumulation rounds and
activation checkpointing. Each wind training run executes one batch before the
main loop and prints allocated, reserved, and peak GPU memory. Adjust only the
micro-batch or accumulation settings after reviewing that probe.

High-resolution fields are cropped from `281 x 321` to `272 x 320`, which is
compatible with the network downsampling factors. The wind diffusion run uses
the complete cropped field rather than a patch larger than the image.

Step-2 hybrid sampling contains all valid one-hour examples and daily 00 UTC
examples for leads two through six. The same valid-time sampling is used for
training and inference selection.

## 4. Regression, diffusion, and evaluation

Run both deterministic and ensemble inference:

```bash
python scripts/inference_wind_pipeline.py \
  --config-name config_inference_wind_pipeline.yaml
```

Regression output covers every paired test sample. Diffusion selects daily 00
UTC initial times and leads 1, 3, and 6. Output dimensions are:

```text
regression: (initial_time, lead_hour, lat, lon)
diffusion:  (initial_time, lead_hour, ensemble, lat, lon)
```

Evaluate deterministic metrics, persistence and bilinear baselines, CRPS,
spread-skill, and radial spectra:

```bash
python scripts/evaluate_wind.py \
  --forecast outputs/wind_corr_diff_inference/wind_regression_2024.nc \
  --truth ../../data/uv10/uv10_2024_clipped_HR.nc \
  --truth-u u_10m --truth-v v_10m \
  --ensemble outputs/wind_corr_diff_inference/wind_diffusion_daily00_2024.nc \
  --low-condition data/wind_generated/step1_conditions_2019_2024.nc \
  --persistence \
  --output outputs/wind_corr_diff_inference/metrics_2024.json
```

## Verification

```bash
pytest -q tests
python -m py_compile swaither/datasets/wind.py swaither/models/wind_forecast.py \
  scripts/train_wind_step1.py scripts/export_wind_forecasts.py \
  scripts/inference_wind_pipeline.py scripts/evaluate_wind.py
```

Tests cover exact-hour time parsing (including leap years and ICON day
fractions), same-year 6-to-72-hour windows, residual channel order, signed wind
values, normalization and absolute-field round trips, valid-time alignment,
explicit high-resolution gaps, checkpoint round trips, two-channel
regression/diffusion forward passes, one-shot 72-hour export, NetCDF dimensions,
CRPS, and the year split.

## Known data limitation

The supplied 2024 ICON high-resolution record ends at `2024-10-31 00:00`.
Step-2 therefore evaluates only its overlap with that record. Step 1 can still
generate 2024 low-resolution conditions through the end of the year. The ERA5
low-resolution input and ICON high-resolution target are different products;
the CorrDiff stage consequently learns both cross-product bias correction and
fine-scale spatial detail.
