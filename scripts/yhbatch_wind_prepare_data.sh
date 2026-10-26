#!/usr/bin/env bash
# Submit with: yhbatch -p a800x -G 1 scripts/yhbatch_wind_prepare_data.sh [Hydra overrides]


set -euo pipefail

cd /XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip/
echo "Repository: $(pwd)"

module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh

echo "[$(date '+%F %T')] Wind data prepare started"

python scripts/compute_wind_stats.py \
  --config-name config_compute_wind_stats.yaml

echo "[$(date '+%F %T')] Wind data prepare completed"
