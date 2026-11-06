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

On Slurm, submit through the timestamped wrappers.  Each wrapper generates one
submission timestamp, uses it for both the Slurm log and training artifacts, and
marks the timestamped run as `latest` only after training succeeds:

```bash
yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_prepare.out \
  scripts/yhbatch_wind_prepare_data.sh
scripts/submit_yhbatch_train_wind_regression.sh
scripts/submit_yhbatch_train_wind_diffusion.sh
```

For example, a submission at `2026-11-04 16:30:00` writes
`outputs/slurm/wind_regression_train_2026-11-04_16-30-00.out` and places the
run under `outputs/wind_regression/2026-11-04_16-30-00/`.  Set `RUN_STAMP` to
override the generated value.  The timestamped wrappers manage
`hydra.run.dir`, `training.run_dir`, `training.checkpoint_dir`, and
`training.tensorboard_dir`; passing different values for those four keys is
rejected.  Existing fixed paths under
`outputs/wind_regression/checkpoints/` and
`outputs/wind_diffusion/checkpoints/` remain the defaults for direct Python
training, inference, and checkpoint evaluation.

To opt a timestamped diffusion run into the newest successfully completed
regression run:

```bash
scripts/submit_yhbatch_train_wind_diffusion.sh \
  training.regression_checkpoint=outputs/wind_regression/latest/checkpoints/WindForecastUNet.best.mdlus
```

## Inference

The inference pipeline performs one regression forward pass per initial time
to obtain all 72 leads, then calls the diffusion model separately for each
configured lead.  It does not read a separate low-resolution forecast-condition
file.

Both NetCDF outputs are written one initial time or one diffusion lead at a
time.  This streaming writer avoids allocating a full test-year array in memory.
Inference explicitly chunks deterministic variables as `(1, 1, lat, lon)` and
ensemble variables as `(1, 1, ensemble, lat, lon)`, so each streamed write maps
to complete HDF5 chunks instead of repeatedly decompressing large spatial tiles.
Regression progress reports separate load/forward, host post-processing, and
NetCDF write time together with throughput and ETA.  Outside inference, the
writer retains its previous automatic chunking behavior unless `chunksizes` is
provided explicitly.
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

For the 2024 test window, use staged inference and evaluation.  The configured
daily 00 UTC range is `2024-01-02` through `2024-10-28`: the earlier date needs
the 2024 low-resolution file's missing 2024-01-01 history, and later dates do not
have a complete 72-hour high-resolution target window.  This produces 301
initial times.  Run the deterministic regression stage first:

```bash
yhbatch --export=ALL,EVAL_STAGE=regression -p a800x_595 -G 1 \
  -o ./outputs/slurm/wind_evaluate_regression_2024.out \
  scripts/yhbatch_evaluate_wind_test.sh
```

Then run diffusion using the same regression checkpoint for its deterministic
mean and residual ensemble:

```bash
yhbatch --export=ALL,EVAL_STAGE=diffusion -p a800x_595 -G 1 \
  -o ./outputs/slurm/wind_evaluate_diffusion_2024.out \
  scripts/yhbatch_evaluate_wind_test.sh
```

`EVAL_STAGE=all` runs both stages in one GPU job.  The wrapper defaults to
`outputs/wind_regression/checkpoints/WindForecastUNet.best.mdlus` and
`outputs/wind_diffusion/latest/checkpoints/WindEDMPrecondSuperResolution.best.mdlus`;
override them with `REGRESSION_CHECKPOINT` and `DIFFUSION_CHECKPOINT`.  Set
`EVAL_OUTPUT_DIR` to isolate a run.  Stage outputs are:

```text
outputs/wind_inference/test_2024/{regression.nc,diffusion.nc}
outputs/wind_inference/test_2024/metrics/regression.json
outputs/wind_inference/test_2024/metrics/diffusion.json
outputs/wind_inference/test_2024/metrics/comparison.json
```

The comparison uses `diffusion ensemble mean - regression` for deterministic
metrics and retains CRPS, spread, skill, and spread/skill for the diffusion
ensemble.  The lower-level evaluator remains available for custom runs:

```bash
python scripts/evaluate_wind.py \
  --forecast outputs/wind_inference/test_2024/regression.nc \
  --ensemble outputs/wind_inference/test_2024/diffusion.nc \
  --truth ../../data/uv10/uv10_2024_clipped_HR.nc \
  --truth-u u_10m --truth-v v_10m \
  --regression-output outputs/wind_inference/test_2024/metrics/regression.json \
  --diffusion-output outputs/wind_inference/test_2024/metrics/diffusion.json \
  --comparison-output outputs/wind_inference/test_2024/metrics/comparison.json
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

Training validation subsets are selected by a fixed-seed uniform random draw
across the complete validation-year dataset.  To compare the saved best
checkpoint with the final checkpoint on the same 512 uniformly selected 2023
initial times:

```bash
yhbatch -p a800x_595 -G 1 -c 8 \
  -o ./outputs/slurm/wind_regression_epochs_eval.out \
  scripts/yhbatch_evaluate_wind_regression_epochs.sh
```

The comparison is written to
`outputs/wind_regression/uniform_2023_checkpoint_comparison.json`.  Pass
`--samples 0` to evaluate every valid 2023 initial time instead of a 512-sample
subset.

Old Step-1/Step-2 checkpoints are not compatible with the new 144-channel
regression output or the 20-channel-plus-mean diffusion condition.
