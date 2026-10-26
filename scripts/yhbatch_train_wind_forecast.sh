#!/usr/bin/env bash
# Submit with:
# yhbatch -p a800x -G 1 scripts/yhbatch_train_wind_forecast.sh [Hydra overrides]

set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"

module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "[$(date '+%F %T')] Wind forecast Step 1 started"
echo "Repository: ${repo_root}"
echo "SLURM job: ${SLURM_JOB_ID:-interactive}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/train_wind_step1.py \
  --config-name=config_training_wind_step1_unet.yaml \
  "$@"

echo "[$(date '+%F %T')] Wind forecast Step 1 completed"
