#!/usr/bin/env bash
# Submit with:
#   yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_prepare.out scripts/yhbatch_wind_prepare_data.sh
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh
export PYTHONUNBUFFERED=1

python -u scripts/compute_wind_stats.py \
  --config-name=config_compute_wind_stats.yaml \
  "$@"
