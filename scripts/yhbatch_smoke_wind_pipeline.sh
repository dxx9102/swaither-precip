#!/usr/bin/env bash
# Submit with:
#   yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_pipeline_smoke.out scripts/yhbatch_smoke_wind_pipeline.sh
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

pytest -q \
  tests/test_wind_models_and_io.py::test_wind_netcdf_streaming_writer_accepts_lead_slices \
  tests/test_wind_evaluate.py

regression_dir="outputs/wind_e2e/regression"
diffusion_dir="outputs/wind_e2e/diffusion"
regression_checkpoint="${regression_dir}/checkpoints/WindForecastUNet.best.mdlus"
diffusion_checkpoint="${diffusion_dir}/checkpoints/WindEDMPrecondSuperResolution.best.mdlus"
inference_dir="outputs/wind_e2e/inference"

python -u scripts/train_wind_regression.py \
  --config-name=config_training_wind_regression.yaml \
  dataset.preload_low=false \
  validation.dataset.preload_low=false \
  model.unet.model_channels=4 \
  'model.unet.channel_mult=[1]' \
  training.epochs=1 \
  training.max_samples=1 \
  training.gradient_accumulation_steps=1 \
  training.num_workers=0 \
  training.memory_probe=false \
  validation.max_samples=1 \
  training.run_dir="${regression_dir}" \
  training.checkpoint_dir="${regression_dir}/checkpoints" \
  training.tensorboard_dir="${regression_dir}/tensorboard"

python -u scripts/train_wind_diffusion.py \
  --config-name=config_training_wind_diffusion.yaml \
  dataset.preload_low=false \
  validation.dataset.preload_low=false \
  'dataset.leads=[1]' \
  'validation.leads=[1]' \
  model.model_args.model_channels=4 \
  'model.model_args.channel_mult=[1,2,2]' \
  training.epochs=1 \
  training.max_samples=1 \
  training.gradient_accumulation_steps=1 \
  training.num_workers=0 \
  training.memory_probe=false \
  training.patch_shape_y=64 \
  training.patch_shape_x=64 \
  validation.max_samples=1 \
  training.regression_checkpoint="${regression_checkpoint}" \
  training.run_dir="${diffusion_dir}" \
  training.checkpoint_dir="${diffusion_dir}/checkpoints" \
  training.tensorboard_dir="${diffusion_dir}/tensorboard"

python -u scripts/inference_wind_pipeline.py \
  --config-name=config_inference_wind_pipeline.yaml \
  dataset.preload_low=false \
  inference.mode=all \
  inference.max_initial_times=1 \
  'inference.diffusion_leads=[1]' \
  inference.num_ensembles=1 \
  inference.sampler_num_steps=2 \
  checkpoints.regression="${regression_checkpoint}" \
  checkpoints.diffusion="${diffusion_checkpoint}" \
  outputs.regression_file="${inference_dir}/regression.nc" \
  outputs.diffusion_file="${inference_dir}/diffusion.nc"

python -u scripts/evaluate_wind.py \
  --forecast "${inference_dir}/regression.nc" \
  --truth ../../data/uv10/uv10_2024_clipped_HR.nc \
  --truth-u u_10m \
  --truth-v v_10m \
  --ensemble "${inference_dir}/diffusion.nc" \
  --output "${inference_dir}/metrics.json"

echo "Wind end-to-end GPU smoke passed"
