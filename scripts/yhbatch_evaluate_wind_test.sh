#!/usr/bin/env bash
# Submit with:
#   yhbatch --export=ALL,EVAL_STAGE=all -p a800x_595 -G 1 \
#     -o ./outputs/slurm/wind_evaluate_test.out \
#     scripts/yhbatch_evaluate_wind_test.sh
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

stage="${EVAL_STAGE:-all}"
output_dir="${EVAL_OUTPUT_DIR:-outputs/wind_inference/test_2024}"
metrics_dir="${output_dir}/metrics"
regression_file="${output_dir}/regression.nc"
diffusion_file="${output_dir}/diffusion.nc"
regression_checkpoint="${REGRESSION_CHECKPOINT:-outputs/wind_regression/checkpoints/WindForecastUNet.best.mdlus}"
diffusion_checkpoint="${DIFFUSION_CHECKPOINT:-outputs/wind_diffusion/latest/checkpoints/WindEDMPrecondSuperResolution.best.mdlus}"
evaluation_batch_size="${EVAL_BATCH_SIZE:-8}"

case "${stage}" in
  regression|diffusion|all) ;;
  *)
    printf 'Invalid EVAL_STAGE=%q; expected regression, diffusion, or all\n' "${stage}" >&2
    exit 2
    ;;
esac

mkdir -p "${output_dir}" "${metrics_dir}"

inference_args=(
  --config-name=config_inference_wind_pipeline.yaml
  "checkpoints.regression=${regression_checkpoint}"
  "outputs.regression_file=${regression_file}"
  "outputs.diffusion_file=${diffusion_file}"
)

if [[ "${stage}" == "regression" || "${stage}" == "all" ]]; then
  python -u scripts/inference_wind_pipeline.py \
    "${inference_args[@]}" \
    inference.mode=regression \
    "$@"
fi

if [[ "${stage}" == "diffusion" || "${stage}" == "all" ]]; then
  if [[ ! -f "${regression_file}" && "${stage}" == "diffusion" ]]; then
    printf 'Regression forecast not found for diffusion comparison: %s\n' \
      "${regression_file}" >&2
    exit 2
  fi
  if [[ ! -f "${diffusion_checkpoint}" ]]; then
    printf 'Diffusion checkpoint not found: %s\n' "${diffusion_checkpoint}" >&2
    exit 2
  fi
  python -u scripts/inference_wind_pipeline.py \
    "${inference_args[@]}" \
    inference.mode=diffusion \
    "checkpoints.diffusion=${diffusion_checkpoint}" \
    "$@"
fi

evaluation_args=(
  --truth ../../data/uv10/uv10_2024_clipped_HR.nc
  --truth-u u_10m
  --truth-v v_10m
  --batch-size "${evaluation_batch_size}"
  --regression-checkpoint "${regression_checkpoint}"
  --diffusion-checkpoint "${diffusion_checkpoint}"
)
if [[ "${EVAL_PERSISTENCE:-0}" == "1" ]]; then
  evaluation_args+=(--persistence)
fi
if [[ -n "${EVAL_LOW_CONDITION:-}" ]]; then
  evaluation_args+=(--low-condition "${EVAL_LOW_CONDITION}")
fi

if [[ "${stage}" == "regression" ]]; then
  python -u scripts/evaluate_wind.py \
    --forecast "${regression_file}" \
    --regression-output "${metrics_dir}/regression.json" \
    "${evaluation_args[@]}"
elif [[ "${stage}" == "diffusion" ]]; then
  python -u scripts/evaluate_wind.py \
    --forecast "${regression_file}" \
    --ensemble "${diffusion_file}" \
    --regression-output "${metrics_dir}/regression.json" \
    --diffusion-output "${metrics_dir}/diffusion.json" \
    --comparison-output "${metrics_dir}/comparison.json" \
    "${evaluation_args[@]}"
else
  python -u scripts/evaluate_wind.py \
    --forecast "${regression_file}" \
    --ensemble "${diffusion_file}" \
    --regression-output "${metrics_dir}/regression.json" \
    --diffusion-output "${metrics_dir}/diffusion.json" \
    --comparison-output "${metrics_dir}/comparison.json" \
    "${evaluation_args[@]}"
fi

printf 'Wind 2024 evaluation stage=%s complete; metrics are in %s\n' \
  "${stage}" "${metrics_dir}"
