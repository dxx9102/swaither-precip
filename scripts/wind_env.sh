#!/usr/bin/env bash
# Source this file before launching the wind pipeline outside the Python
# entry points, which also call the same storage configuration themselves.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_wind_root="$(cd "${script_dir}/.." && pwd)"
wind_root="${SWAITHER_WIND_ROOT:-${default_wind_root}}"
wind_cache="${wind_root}/.cache"

export SWAITHER_WIND_ROOT="${wind_root}"
export WANDB_DIR="${wind_root}/outputs/wandb"
export WANDB_CACHE_DIR="${wind_cache}/wandb"
export XDG_CACHE_HOME="${wind_cache}"
export TORCH_HOME="${wind_cache}/torch"
export HF_HOME="${wind_cache}/huggingface"
export TRITON_CACHE_DIR="${wind_cache}/triton"
export MPLCONFIGDIR="${wind_cache}/matplotlib"
export CARTOPY_DATA_DIR="${wind_cache}/cartopy"
export LOCAL_CACHE="${wind_cache}/physicsnemo"
export NUMBA_CACHE_DIR="${wind_cache}/numba"
export JOBLIB_TEMP_FOLDER="${wind_cache}/joblib"

mkdir -p \
  "${wind_root}/outputs" \
  "${wind_root}/data/wind_generated" \
  "${wind_root}/data/stats" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${HF_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${MPLCONFIGDIR}" \
  "${CARTOPY_DATA_DIR}" \
  "${LOCAL_CACHE}" \
  "${NUMBA_CACHE_DIR}" \
  "${JOBLIB_TEMP_FOLDER}"
