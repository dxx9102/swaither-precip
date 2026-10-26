# Hourly wind forecasting and CorrDiff super-resolution

This pipeline is independent of the original precipitation workflow. It keeps
the precipitation code and configurations intact while adding dedicated wind
datasets, models, training entry points, inference, and evaluation.

## Pipeline

1. A deterministic PhysicsNeMo U-Net uses the latest six hourly low-resolution
   `u10/v10` fields, two low-resolution static fields, and four coordinate
   channels to predict wind one hour ahead.
2. Inference feeds each prediction back into the six-hour history for six
   autoregressive steps.
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

Compute Step-1 statistics from the 2019-2022 low-resolution records:

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
outputs/wind_step1_unet/checkpoints/WindForecastUNet.best.mdlus
```

## 2. Frozen Step-1 conditions

Generate one-to-six-hour conditions for all initial times that have a complete
same-year six-hour history:

```bash
python scripts/export_wind_forecasts.py \
  --config-name config_export_wind_forecasts.yaml
```

The default output is:

```text
data/wind_generated/step1_conditions_2019_2024.nc
```

It has dimensions `(initial_time, lead_hour, lat, lon)`, two variables
`u10/v10` in `m s-1`, and a `valid_time` coordinate. Step-2 never updates the
Step-1 model.

Compute Step-2 statistics using the generated conditions and ICON targets from
2019-2022 only:

```bash
python scripts/compute_wind_stats.py \
  --config-name config_compute_wind_stats_step2.yaml
```

## 3. CorrDiff training

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
fractions), same-year history windows, channel order, signed wind values,
normalization round trips, valid-time alignment, explicit high-resolution gaps,
checkpoint round trips, two-channel regression/diffusion forward passes,
autoregressive rollout, NetCDF dimensions, CRPS, and the year split.

## Known data limitation

The supplied 2024 ICON high-resolution record ends at `2024-10-31 00:00`.
Step-2 therefore evaluates only its overlap with that record. Step 1 can still
generate 2024 low-resolution conditions through the end of the year. The ERA5
low-resolution input and ICON high-resolution target are different products;
the CorrDiff stage consequently learns both cross-product bias correction and
fine-scale spatial detail.
