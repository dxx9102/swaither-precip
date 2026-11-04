#!/usr/bin/env bash
# Submit with:
#   scripts/submit_yhbatch_train_wind_regression.sh [Hydra overrides]
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

source scripts/yhbatch_wind_common.sh
run_stamp="$(wind_resolve_run_stamp)"
wind_reject_path_overrides "$@"
run_dir="$(wind_training_run_dir "${repo_root}" wind_regression "${run_stamp}")"
latest_path="${repo_root}/outputs/wind_regression/latest"
if [[ -e "${run_dir}" ]]; then
  printf 'Timestamped regression run already exists: %s\n' "${run_dir}" >&2
  exit 2
fi
if [[ -e "${latest_path}" && ! -L "${latest_path}" ]]; then
  printf 'Refusing to replace non-symlink latest path: %s\n' "${latest_path}" >&2
  exit 2
fi

python -u scripts/train_wind_regression.py \
  --config-name=config_training_wind_regression.yaml \
  hydra.run.dir="${run_dir}" \
  training.run_dir="${run_dir}" \
  training.checkpoint_dir="${run_dir}/checkpoints" \
  training.tensorboard_dir="${run_dir}/tensorboard" \
  "$@"

wind_mark_latest "${latest_path}" "${run_stamp}"
printf 'Regression artifacts: %s\nLatest link: %s -> %s\n' \
  "${run_dir}" "${latest_path}" "${run_stamp}"
