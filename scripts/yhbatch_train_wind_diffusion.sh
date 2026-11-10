#!/usr/bin/env bash
# Submit with:
#   scripts/submit_yhbatch_train_wind_diffusion.sh [Hydra overrides]
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
source scripts/yhbatch_wind_common.sh

default_output_root="${WIND_OUTPUTS_ROOT:-${repo_root}/outputs}"
wind_parse_outputs "${repo_root}" "${default_output_root}" "$@"
output_root="${WIND_OUTPUTS_ROOT}"
forward_args=("${WIND_FORWARD_ARGS[@]}")
wind_reject_path_overrides "${forward_args[@]}"
wind_find_regression_checkpoint "${repo_root}" "${forward_args[@]}"

module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

run_stamp="$(wind_resolve_run_stamp)"
run_dir="$(wind_training_run_dir "${output_root}" wind_diffusion "${run_stamp}")"
latest_path="$(wind_training_latest_path "${output_root}" wind_diffusion)"
checkpoint_args=()
if ((WIND_HAS_REGRESSION_CHECKPOINT)); then
  regression_checkpoint="${WIND_REGRESSION_CHECKPOINT}"
else
  regression_checkpoint="$(wind_resolve_regression_latest_checkpoint "${output_root}")"
  checkpoint_args=("training.regression_checkpoint=${regression_checkpoint}")
fi
if [[ -e "${run_dir}" ]]; then
  printf 'Timestamped diffusion run already exists: %s\n' "${run_dir}" >&2
  exit 2
fi
if [[ -e "${latest_path}" && ! -L "${latest_path}" ]]; then
  printf 'Refusing to replace non-symlink latest path: %s\n' "${latest_path}" >&2
  exit 2
fi

python -u scripts/train_wind_diffusion.py \
  --config-name=config_training_wind_diffusion.yaml \
  hydra.run.dir="${run_dir}" \
  training.run_dir="${run_dir}" \
  training.checkpoint_dir="${run_dir}/checkpoints" \
  training.tensorboard_dir="${run_dir}/tensorboard" \
  "${forward_args[@]}" "${checkpoint_args[@]}"

wind_mark_latest "${latest_path}" "${run_stamp}"
printf 'Diffusion artifacts: %s\nLatest link: %s -> %s\n' \
  "${run_dir}" "${latest_path}" "${run_stamp}"
