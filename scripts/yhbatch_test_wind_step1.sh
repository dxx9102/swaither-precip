#!/usr/bin/env bash
# Submit with:
# yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/slurm-step1-test.out \
#   scripts/yhbatch_test_wind_step1.sh

set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"

module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

checkpoint="${WIND_STEP1_CHECKPOINT:-outputs/wind_step1_unet/checkpoints/WindForecastUNet.best.mdlus}"
truth="${WIND_STEP1_TEST_TRUTH:-../../data/uv10/uv10_2024_clipped_LRes.nc}"
output_dir="${WIND_STEP1_TEST_OUTPUT_DIR:-outputs/wind_step1_test}"
forecast="${output_dir}/forecast_2024.nc"
metrics="${output_dir}/metrics_2024.json"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Step-1 checkpoint not found: ${checkpoint}" >&2
  exit 1
fi
if [[ ! -f "${truth}" ]]; then
  echo "Step-1 test truth file not found: ${truth}" >&2
  exit 1
fi

mkdir -p "${output_dir}"

echo "[$(date '+%F %T')] Step-1 test started"
echo "Repository: ${repo_root}"
echo "Checkpoint: ${checkpoint}"
echo "Test truth: ${truth}"
echo "SLURM job: ${SLURM_JOB_ID:-interactive}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u scripts/export_wind_forecasts.py \
  --config-name=config_export_wind_forecasts.yaml \
  "checkpoint=${checkpoint}" \
  "years=[2024]" \
  "output_file=${forecast}"

python -u scripts/evaluate_wind.py \
  --forecast "${forecast}" \
  --truth "${truth}" \
  --truth-u u10 \
  --truth-v v10 \
  --persistence \
  --output "${metrics}"

echo "[$(date '+%F %T')] Step-1 test completed"
echo "Forecast: ${forecast}"
echo "Metrics: ${metrics}"
