#!/usr/bin/env bash
# Submit the timestamped wind-regression training job.
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
source scripts/yhbatch_wind_common.sh

wind_parse_outputs "${repo_root}" "${repo_root}/outputs" "$@"
output_root="${WIND_OUTPUTS_ROOT}"
forward_args=("${WIND_FORWARD_ARGS[@]}")
wind_reject_path_overrides "${forward_args[@]}"

run_stamp="$(wind_resolve_run_stamp)"
log_dir="${output_root}/slurm"
log_file="$(wind_slurm_log_path "${output_root}" wind_regression_train "${run_stamp}")"
run_dir="$(wind_training_run_dir "${output_root}" wind_regression "${run_stamp}")"

mkdir -p "${log_dir}"
if [[ -e "${log_file}" || -e "${run_dir}" ]]; then
  printf 'Submission paths already exist for RUN_STAMP=%s\n' "${run_stamp}" >&2
  printf '  log: %s\n  run: %s\n' "${log_file}" "${run_dir}" >&2
  exit 2
fi

printf 'Submitting wind regression: RUN_STAMP=%s\n  output root: %s\n  log: %s\n  run: %s\n' \
  "${run_stamp}" "${output_root}" "${log_file}" "${run_dir}"
export WIND_OUTPUTS_ROOT="${output_root}"
yhbatch \
  -p a800x_595 \
  -G 1 \
  --export="ALL,RUN_STAMP=${run_stamp}" \
  -o "${log_file}" \
  scripts/yhbatch_train_wind_regression.sh \
  "${forward_args[@]}"
