#!/usr/bin/env bash
# Submit with:
#   yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_inference.out scripts/yhbatch_inference_wind.sh
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -u scripts/inference_wind_pipeline.py \
  --config-name=config_inference_wind_pipeline.yaml \
  "$@"
