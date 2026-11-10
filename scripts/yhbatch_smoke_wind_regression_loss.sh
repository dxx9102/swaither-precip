#!/usr/bin/env bash
# One-step GPU smoke for the configurable weighted regression loss.
# Submit manually with a matching RUN_STAMP:
#   RUN_STAMP=$(date +%Y%m%d_%H%M%S)
#   yhbatch -p a800x_595 -G 1 \
#     --export=ALL,RUN_STAMP=${RUN_STAMP} \
#     -o ./outputs/slurm/wind_regression_loss_smoke_${RUN_STAMP}.out \
#     scripts/yhbatch_smoke_wind_regression_loss.sh
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

run_stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
run_dir="${repo_root}/outputs/wind_regression/loss_smoke_${run_stamp}"
if [[ -e "${run_dir}" ]]; then
  printf 'Smoke run directory already exists: %s\n' "${run_dir}" >&2
  exit 2
fi

python -u scripts/train_wind_regression.py \
  --config-name=config_training_wind_regression.yaml \
  hydra.run.dir="${run_dir}" \
  training.run_dir="${run_dir}" \
  training.checkpoint_dir="${run_dir}/checkpoints" \
  training.tensorboard_dir="${run_dir}/tensorboard" \
  training.epochs=1 \
  training.max_samples=1 \
  training.batch_size=1 \
  training.gradient_accumulation_steps=1 \
  training.num_workers=0 \
  training.memory_probe=false \
  validation.max_samples=1 \
  validation.batch_size=1 \
  training.loss.terms.normalized_mse.weight=0.2 \
  training.loss.terms.vector_mse.weight=1.0 \
  training.loss.terms.vector_charbonnier.weight=0.1 \
  training.loss.terms.speed_mae.weight=0.05 \
  training.loss.terms.direction.weight=0.01

printf 'Wind regression loss smoke artifacts: %s\n' "${run_dir}"
