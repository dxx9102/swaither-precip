#!/usr/bin/env bash
# Submit with:
# yhbatch -p a800x -G 1 scripts/yhbatch_train_wind_super_resolution.sh
#
# This job consumes the Step-1 checkpoint, exports frozen forecast conditions,
# computes Step-2 statistics, trains regression, and then trains diffusion with
# the newest regression checkpoint from this job.

set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"

step1_checkpoint="${WIND_STEP1_CHECKPOINT:-outputs/wind_step1_unet/checkpoints/WindForecastUNet.best.mdlus}"
step1_stats="${WIND_STEP1_STATS:-data/stats/wind_step1_2019_2022.json}"
conditions_file="${WIND_CONDITIONS_FILE:-data/wind_generated/step1_conditions_2019_2024.nc}"
step2_stats="${WIND_STEP2_STATS:-data/stats/wind_step2_2019_2022.json}"
regression_run_dir="${WIND_REGRESSION_RUN_DIR:-outputs/wind_step2_regression}"
diffusion_run_dir="${WIND_DIFFUSION_RUN_DIR:-outputs/wind_step2_diffusion}"

if [[ ! -f "${step1_checkpoint}" ]]; then
  echo "Step-1 checkpoint not found: ${step1_checkpoint}" >&2
  echo "Set WIND_STEP1_CHECKPOINT to the best Step-1 .mdlus file." >&2
  exit 1
fi
if [[ ! -f "${step1_stats}" ]]; then
  echo "Step-1 normalization statistics not found: ${step1_stats}" >&2
  echo "Compute them first or set WIND_STEP1_STATS to the file used by Step 1." >&2
  exit 1
fi

module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

common_overrides=()
if [[ -n "${WIND_RAW_DATA_ROOT:-}" ]]; then
  common_overrides+=("paths.raw_data_root=${WIND_RAW_DATA_ROOT}")
fi

echo "[$(date '+%F %T')] Wind super-resolution started"
echo "Repository: ${repo_root}"
echo "SLURM job: ${SLURM_JOB_ID:-interactive}"
echo "Step-1 checkpoint: ${step1_checkpoint}"
echo "Step-1 statistics: ${step1_stats}"
echo "Conditions: ${conditions_file}"
echo "Step-2 statistics: ${step2_stats}"

if [[ -d "${regression_run_dir}/checkpoints_regression" ]] && \
   find "${regression_run_dir}/checkpoints_regression" -maxdepth 1 -type f -name 'UNet.*.mdlus' -print -quit | grep -q .; then
  echo "[WARN] Regression checkpoints already exist in ${regression_run_dir}; training will resume from them."
  echo "[WARN] Set WIND_REGRESSION_RUN_DIR and WIND_DIFFUSION_RUN_DIR to new paths when changing Step-1 inputs."
fi

echo "[$(date '+%F %T')] Exporting frozen Step-1 forecast conditions"
python -u scripts/export_wind_forecasts.py \
  --config-name=config_export_wind_forecasts.yaml \
  "checkpoint=${step1_checkpoint}" \
  "output_file=${conditions_file}" \
  "dataset.stats_path=${step1_stats}" \
  "${common_overrides[@]}"

echo "[$(date '+%F %T')] Computing Step-2 normalization statistics"
python -u scripts/compute_wind_stats.py \
  --config-name=config_compute_wind_stats_step2.yaml \
  "dataset.condition_files=['${conditions_file}']" \
  "output_file=${step2_stats}" \
  "${common_overrides[@]}"

echo "[$(date '+%F %T')] Training deterministic wind regression"
python -u scripts/train_yearly.py \
  --config-name=config_training_wind_step2_regression.yaml \
  "dataset.condition_files=['${conditions_file}']" \
  "dataset.stats_path=${step2_stats}" \
  "training.io.checkpoint_dir=${regression_run_dir}" \
  "${common_overrides[@]}"

regression_checkpoint_dir="${regression_run_dir}/checkpoints_regression"
regression_checkpoint=""
if [[ -d "${regression_checkpoint_dir}" ]]; then
  regression_checkpoint="$(find "${regression_checkpoint_dir}" -maxdepth 1 -type f -name 'UNet.*.mdlus' -print | sort -V | tail -n 1)"
fi
if [[ -z "${regression_checkpoint}" ]]; then
  echo "Regression completed without producing a checkpoint in ${regression_checkpoint_dir}" >&2
  exit 1
fi
echo "[$(date '+%F %T')] Using regression checkpoint: ${regression_checkpoint}"

echo "[$(date '+%F %T')] Training conditional wind diffusion"
python -u scripts/train_yearly.py \
  --config-name=config_training_wind_step2_diffusion.yaml \
  "dataset.condition_files=['${conditions_file}']" \
  "dataset.stats_path=${step2_stats}" \
  "training.io.checkpoint_dir=${diffusion_run_dir}" \
  "training.io.regression_checkpoint_path=${regression_checkpoint}" \
  "${common_overrides[@]}"

echo "[$(date '+%F %T')] Wind regression and diffusion training completed"
