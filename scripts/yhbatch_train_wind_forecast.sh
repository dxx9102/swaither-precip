#!/usr/bin/env bash
# Submit with: yhbatch scripts/yhbatch_train_wind_forecast.sh [Hydra overrides]
#
# yhbatch is a Slurm sbatch wrapper on this cluster, so the directives below
# deliberately use the standard #SBATCH spelling.
#SBATCH --job-name=wind-forecast
#SBATCH --partition=szjt_idd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=slurm-wind-forecast-%j.out
#SBATCH --error=slurm-wind-forecast-%j.err

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
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
