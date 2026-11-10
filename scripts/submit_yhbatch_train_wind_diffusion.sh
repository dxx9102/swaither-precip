#!/usr/bin/env bash
# Submit the timestamped wind-diffusion training job.
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
source scripts/yhbatch_wind_common.sh

wind_parse_outputs "${repo_root}" "${repo_root}/outputs" "$@"
output_root="${WIND_OUTPUTS_ROOT}"
forward_args=("${WIND_FORWARD_ARGS[@]}")
wind_reject_path_overrides "${forward_args[@]}"
wind_find_regression_checkpoint "${repo_root}" "${forward_args[@]}"

run_stamp="$(wind_resolve_run_stamp)"
log_dir="${output_root}/slurm"
log_file="$(wind_slurm_log_path "${output_root}" wind_diffusion_train "${run_stamp}")"
run_dir="$(wind_training_run_dir "${output_root}" wind_diffusion "${run_stamp}")"
checkpoint_args=()
if ((WIND_HAS_REGRESSION_CHECKPOINT)); then
  regression_checkpoint="${WIND_REGRESSION_CHECKPOINT}"
else
  regression_checkpoint="$(wind_resolve_regression_latest_checkpoint "${output_root}")"
  checkpoint_args=("training.regression_checkpoint=${regression_checkpoint}")
fi

mkdir -p "${log_dir}"
if [[ -e "${log_file}" || -e "${run_dir}" ]]; then
  printf 'Submission paths already exist for RUN_STAMP=%s\n' "${run_stamp}" >&2
  printf '  log: %s\n  run: %s\n' "${log_file}" "${run_dir}" >&2
  exit 2
fi

printf 'Submitting wind diffusion: RUN_STAMP=%s\n  output root: %s\n  log: %s\n  run: %s\n  regression checkpoint: %s\n' \
  "${run_stamp}" "${output_root}" "${log_file}" "${run_dir}" "${regression_checkpoint}"
export WIND_OUTPUTS_ROOT="${output_root}"
yhbatch \
  -p a800x_595 \
  -G 1 \
  --export="ALL,RUN_STAMP=${run_stamp}" \
  -o "${log_file}" \
  scripts/yhbatch_train_wind_diffusion.sh \
  "${forward_args[@]}" "${checkpoint_args[@]}"
