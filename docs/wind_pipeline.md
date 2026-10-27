# Hourly wind regression and residual diffusion

The wind workflow uses two networks.  A deterministic U-Net maps six hours of
hourly low-resolution `u10/v10` history directly to 72 hours of full-domain
`281 x 321` high-resolution absolute wind.  A conditional EDM model then learns
the two-channel residual between the high-resolution truth and the frozen
U-Net prediction for each forecast lead.

The workflow uses train-only robust min/max statistics.  The year split is
`2019-2022` for training, `2023` for validation, and `2024` for testing.  All
samples require contiguous hourly history and a complete same-year 72-hour
high-resolution target window.

## Data contract

The regression dataset returns:

```text
y: (144, 281, 321) normalized HR wind, [h1_u, h1_v, ..., h72_u, h72_v]
x: (18, 281, 321) = 6 * [u10, v10] + [elevation, land_sea_mask] + 4 grid channels
```

Low-resolution history is bilinearly interpolated to the complete HR grid.
Terrain and land-sea mask come only from `fj_HR.nc`; `fj_LRes.nc` is not used.
The diffusion view expands each initial time into one sample per lead, adds
`sin/cos` lead-encoding channels, and trains on residuals relative to the frozen
regression output.

## Environment

```bash
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
cd /XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip
source scripts/wind_env.sh
```

Data paths default to `../../data/uv10` and can be overridden with
`paths.raw_data_root=...`.

## Training

Compute training-only statistics:

```bash
python scripts/compute_wind_stats.py --config-name=config_compute_wind_stats.yaml
```

The shipped statistics config samples 512 training-year time steps with up to
1,000,000 values per channel.  Increase `max_samples` for a production
statistics pass when the additional I/O cost is acceptable.

Train the 72-hour high-resolution regression U-Net:

```bash
python scripts/train_wind_regression.py \
  --config-name=config_training_wind_regression.yaml
```

The default checkpoint is:

```text
outputs/wind_regression/checkpoints/WindForecastUNet.best.mdlus
```

Train residual diffusion with that U-Net frozen:

```bash
python scripts/train_wind_diffusion.py \
  --config-name=config_training_wind_diffusion.yaml
```

The default diffusion checkpoint is:

```text
outputs/wind_diffusion/checkpoints/WindEDMPrecondSuperResolution.best.mdlus
```

On Slurm, use the job scripts.  They explicitly change to the repository
directory and activate the required environment:

```bash
yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_prepare.out \
  scripts/yhbatch_wind_prepare_data.sh
yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_regression_train.out \
  scripts/yhbatch_train_wind_regression.sh
yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_diffusion_train.out \
  scripts/yhbatch_train_wind_diffusion.sh
```

## Inference

The inference pipeline performs one regression forward pass per initial time
to obtain all 72 leads, then calls the diffusion model separately for each
configured lead.  It does not read a separate low-resolution forecast-condition
file.

Both NetCDF outputs are written one initial time or one diffusion lead at a
time.  This streaming writer avoids allocating a full test-year array in memory.
`inference.sampler_num_steps` must be at least 2 because PhysicsNeMo's stochastic
sampler requires a non-degenerate sigma-step grid.

```bash
python scripts/inference_wind_pipeline.py \
  --config-name=config_inference_wind_pipeline.yaml
```

Deterministic output has dimensions
`(initial_time, lead_hour, lat, lon)`.  Diffusion output adds an ensemble
dimension.  By default inference uses daily 00 UTC initial times and all leads
1-72; use Hydra overrides such as `inference.initial_hour=null`,
`inference.max_initial_times=1`, or `inference.diffusion_leads=[1,6,24]` for a
smaller run.

For Slurm:

```bash
yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_inference.out \
  scripts/yhbatch_inference_wind.sh
```

Evaluate against HR truth with the existing generic evaluator:

```bash
python scripts/evaluate_wind.py \
  --forecast outputs/wind_inference/wind_regression_2024.nc \
  --truth ../../data/uv10/uv10_2024_clipped_HR.nc \
  --truth-u u_10m --truth-v v_10m \
  --ensemble outputs/wind_inference/wind_diffusion_daily00_2024.nc \
  --output outputs/wind_inference/metrics_2024.json
```

## Verification

```bash
pytest -q tests/test_wind_data.py tests/test_wind_models_and_io.py \
  tests/test_wind_configs.py
yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_joint_test.out \
  scripts/yhbatch_test_wind_joint.sh
yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_pipeline_smoke.out \
  scripts/yhbatch_smoke_wind_pipeline.sh
```

The joint job covers the wind tests plus real-shape U-Net and residual-diffusion
forward/backward memory probes.  The smoke job trains one regression and one
diffusion sample, reloads both checkpoints, runs inference, and evaluates both
NetCDF outputs.

Old Step-1/Step-2 checkpoints are not compatible with the new 144-channel
regression output or the 20-channel-plus-mean diffusion condition.
