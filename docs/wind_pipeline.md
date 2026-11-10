# Hourly wind regression and residual diffusion

The wind workflow uses two networks.  A deterministic U-Net maps six hours of
hourly low-resolution `u10/v10` history directly to 72 hours of full-domain
`281 x 321` high-resolution absolute wind.  A conditional EDM model then learns
the two-channel residual between the high-resolution truth and the frozen
U-Net prediction for each forecast lead.

Both networks use SongUNet's native global augmentation embedding for time
conditioning.  The regression model receives six cyclic initial-time features;
the residual diffusion model receives six initial-time and six valid-time
features.  These labels are not added as spatial `H x W` channels.

The workflow uses train-only robust min/max statistics.  The year split is
`2019-2022` for training, `2023` for validation, and `2024` for testing.  All
samples require contiguous hourly history and a complete 72-hour
high-resolution target window.  Training samples start at 00, 06, 12, or 18
UTC and may cross year boundaries inside 2019-2022.  Validation and test
samples start at daily 00 UTC.  With cross-year sampling enabled, every history
and target timestamp must remain inside the configured dataset year set, so a
sample can never cross from training into validation or from validation into
test.

## Data contract

The regression dataset returns:

```text
y: (144, 281, 321) normalized HR wind, [h1_u, h1_v, ..., h72_u, h72_v]
x: (18, 281, 321) = 6 * [u10, v10] + [elevation, land_sea_mask] + 4 grid channels
time6: (6,) = [doy_sin, doy_cos, month_sin, month_cos, hour_sin, hour_cos]
```

Low-resolution history is bilinearly interpolated to the complete HR grid.
Terrain and land-sea mask come only from `fj_HR.nc`; `fj_LRes.nc` is not used.
The diffusion view expands each initial time into one sample per lead, adds
`sin/cos` lead-encoding channels to make the spatial condition 20 channels, and
trains on residuals relative to the frozen regression output.  It also returns
the 12-dimensional label `[initial_time_6, valid_time_6]`, where
`valid_time = initial_time + lead`.

Time conditioning enters `SongUNet` through `augment_dim` and
`augment_labels`.  The diffusion time labels and noise level are combined in
the existing global embedding MLP.  Multi-patch residual training keeps the
spatial contract unchanged and expands labels in P-major order to match
`RandomPatching2D`; the frozen regression adapter receives the first six
initial-time features.  With `augment_dim: 0`, callers omit time labels and the
model remains compatible with pre-time-conditioning checkpoints.

`WindForecastDataset` accepts optional `initial_hours` and `allow_cross_year`
arguments.  Their defaults retain the legacy all-hour, same-year behavior.
`allow_cross_year=True` requires an explicit `years` set and permits crossing
only between years in that set.  The hourly restriction is applied while the
dataset index is built; it is independent of `training.max_samples`.

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

Regression training supports a weighted composite loss.  The configured terms
are summed without automatic renormalization:

```text
L_total = sum_i weight_i * L_i
```

The default keeps the previous behavior by enabling only `normalized_mse`.
Available terms are:

- `normalized_mse`: the existing normalized-space MSE.
- `physical_vector_mse`: `mean(du^2 + dv^2) / S`, where `du` and `dv` are
  residuals converted back to m/s and `S` is the mean squared output range.
- `physical_vector_charbonnier`: smooth vector-magnitude error with a
  configurable `epsilon`, scaled by `sqrt(S)`.
- `physical_speed_mae`: absolute wind-speed error scaled by `sqrt(S)`.
- `direction_cosine`: `1 - cos(angle_delta)`, evaluated only where the truth
  speed is at least `min_speed`.

Terms with zero weight are not evaluated.  Enable a multi-term experiment with
Hydra overrides while retaining the four default terms as disabled entries:

```bash
python scripts/train_wind_regression.py \
  --config-name=config_training_wind_regression.yaml \
  training.loss.terms.normalized_mse.weight=0.2 \
  training.loss.terms.vector_mse.weight=1.0 \
  training.loss.terms.vector_charbonnier.weight=0.1 \
  training.loss.terms.speed_mae.weight=0.05 \
  training.loss.terms.direction.weight=0.01
```

Every term accepts its own `type`, `weight`, and type-specific parameters.
New named terms may also be added through Hydra, which is useful when comparing
two direction losses with different calm-wind thresholds.  Regression
checkpoint selection defaults to `training.checkpoint_metric=vector_rmse`; set
it to `validation_loss` to select by the configured composite objective.

The one-step multi-loss GPU smoke job can be submitted manually with:

```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
yhbatch -p a800x_595 -G 1 \
  --export=ALL,RUN_STAMP=${RUN_STAMP} \
  -o ./outputs/slurm/wind_regression_loss_smoke_${RUN_STAMP}.out \
  scripts/yhbatch_smoke_wind_regression_loss.sh
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

Pass the Hydra override `paths.outputs=<root>` to place both training products
and Slurm logs below one custom root:

```bash
scripts/submit_yhbatch_train_wind_regression.sh paths.outputs=output/xx
scripts/submit_yhbatch_train_wind_diffusion.sh \
  paths.outputs=output/xx
```

Relative roots are resolved from the repository root; absolute roots are also
accepted.  Omitting `paths.outputs` preserves the existing `<repo>/outputs`
layout.  For a root `<root>` and submission timestamp `<RUN_STAMP>`, the
wrappers create:

```text
<root>/slurm/wind_regression_train_<RUN_STAMP>.out
<root>/slurm/wind_diffusion_train_<RUN_STAMP>.out
<root>/wind_regression/<RUN_STAMP>/checkpoints/
<root>/wind_regression/<RUN_STAMP>/tensorboard/
<root>/wind_diffusion/<RUN_STAMP>/checkpoints/
<root>/wind_diffusion/<RUN_STAMP>/tensorboard/
<root>/wind_regression/latest
<root>/wind_diffusion/latest
```

The diffusion submission automatically uses the newest successfully completed
regression run below the same output root:

```text
<root>/wind_regression/latest/checkpoints/WindForecastUNet.best.mdlus
```

The submit wrapper resolves this symlink to the concrete timestamped
checkpoint before calling `yhbatch`, so changing `latest` while the diffusion
job is pending does not change which regression model it loads.  A missing or
dangling `latest`, or a missing best checkpoint, is an error; the wrapper does
not fall back to a legacy path or scan timestamped directories.  An explicit
`training.regression_checkpoint=<path>` override takes precedence.

Set `RUN_STAMP` to override the generated timestamp.  The timestamped wrappers
manage `hydra.run.dir`, `training.run_dir`, `training.checkpoint_dir`, and
`training.tensorboard_dir`; passing different values for those four keys is
rejected.  Existing fixed paths under
`outputs/wind_regression/checkpoints/` and
`outputs/wind_diffusion/checkpoints/` remain the defaults for direct Python
training, inference, and checkpoint evaluation.

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

Regression training validates every complete 2023 daily 00 UTC initial-time
window (361 samples).  Diffusion validation uses a fixed-seed uniform random
draw of 32 base initial times across those 361 windows and evaluates every
configured validation lead (`1, 6, 24, 48, 72`), for 160 expanded pairs.  When
`validation.sample_by_initial_time` is false, the legacy behavior draws
`validation.max_samples` individual `(initial_time, lead)` pairs instead.

To compare the saved best checkpoint with the final checkpoint on the same 512
uniformly selected 2023 initial times:

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
regression output or the 20-channel-plus-mean diffusion condition.  Version-3
`WindForecastUNet` checkpoints are supported through the explicit backward
compatibility path and load with time conditioning disabled.
