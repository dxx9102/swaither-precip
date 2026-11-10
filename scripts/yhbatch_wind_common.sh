#!/usr/bin/env bash
# Shared helpers for timestamped wind-training yhbatch submissions.

wind_resolve_run_stamp() {
  local stamp="${RUN_STAMP:-}"
  if [[ -z "${stamp}" ]]; then
    stamp="$(TZ=Asia/Shanghai date +%Y-%m-%d_%H-%M-%S)"
  fi
  if [[ ! "${stamp}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
    printf 'Invalid RUN_STAMP %q; expected YYYY-MM-DD_HH-MM-SS\n' "${stamp}" >&2
    return 2
  fi
  printf '%s\n' "${stamp}"
}

wind_parse_outputs() {
  local repo_root="$1"
  local default_root="$2"
  shift 2
  local argument value candidate resolved
  local found=0

  WIND_OUTPUTS_ROOT=""
  WIND_FORWARD_ARGS=()
  for argument in "$@"; do
    case "${argument}" in
      paths.outputs|paths.outputs=)
        printf 'paths.outputs requires a non-empty value\n' >&2
        return 2
        ;;
      paths.outputs=*)
        if ((found)); then
          printf 'paths.outputs may be specified at most once\n' >&2
          return 2
        fi
        value="${argument#paths.outputs=}"
        if [[ "${value}" == *$'\n'* || "${value}" == *$'\r'* ]]; then
          printf 'paths.outputs contains an invalid newline character\n' >&2
          return 2
        fi
        found=1
        ;;
      *)
        WIND_FORWARD_ARGS+=("${argument}")
        ;;
    esac
  done

  candidate="${value:-${default_root}}"
  if [[ "${candidate}" != /* ]]; then
    candidate="${repo_root}/${candidate}"
  fi
  if ! resolved="$(realpath -m -- "${candidate}" 2>/dev/null)" || [[ -z "${resolved}" ]]; then
    printf 'Unable to resolve output root %q\n' "${candidate}" >&2
    return 2
  fi
  if [[ -e "${resolved}" || -L "${resolved}" ]] && [[ ! -d "${resolved}" ]]; then
    printf 'Output root is not a directory: %s\n' "${resolved}" >&2
    return 2
  fi
  WIND_OUTPUTS_ROOT="${resolved}"
}

wind_training_run_dir() {
  printf '%s/%s/%s\n' "$1" "$2" "$3"
}

wind_slurm_log_path() {
  printf '%s/slurm/%s_%s.out\n' "$1" "$2" "$3"
}

wind_training_latest_path() {
  printf '%s/%s/latest\n' "$1" "$2"
}

wind_reject_path_overrides() {
  local argument
  for argument in "$@"; do
    case "${argument}" in
      hydra.run.dir=*|training.run_dir=*|training.checkpoint_dir=*|training.tensorboard_dir=*)
        printf 'Output-directory override is managed by the timestamped wrapper: %s\n' \
          "${argument}" >&2
        return 2
        ;;
    esac
  done
}

wind_find_regression_checkpoint() {
  local repo_root="$1"
  shift
  local argument value candidate
  local found=0

  WIND_REGRESSION_CHECKPOINT=""
  WIND_HAS_REGRESSION_CHECKPOINT=0
  for argument in "$@"; do
    case "${argument}" in
      training.regression_checkpoint|training.regression_checkpoint=)
        printf 'training.regression_checkpoint requires a non-empty path\n' >&2
        return 2
        ;;
      training.regression_checkpoint=*)
        found=1
        value="${argument#training.regression_checkpoint=}"
        ;;
    esac
  done
  if ((found)); then
    candidate="${value}"
    if [[ "${candidate}" != /* ]]; then
      candidate="${repo_root}/${candidate}"
    fi
    if [[ ! -f "${candidate}" ]]; then
      printf 'Regression checkpoint does not exist: %s\n' "${value}" >&2
      return 2
    fi
    WIND_REGRESSION_CHECKPOINT="${value}"
    WIND_HAS_REGRESSION_CHECKPOINT=1
  fi
}

wind_resolve_regression_latest_checkpoint() {
  local output_root="$1"
  local phase_root="${output_root}/wind_regression"
  local latest_path="${phase_root}/latest"
  local resolved_target phase_root_resolved checkpoint resolved_checkpoint

  if [[ ! -e "${latest_path}" && ! -L "${latest_path}" ]]; then
    printf 'Regression latest link not found: %s\n' "${latest_path}" >&2
    return 2
  fi
  if [[ ! -L "${latest_path}" ]]; then
    printf 'Regression latest path is not a symlink: %s\n' "${latest_path}" >&2
    return 2
  fi
  if ! resolved_target="$(realpath -e -- "${latest_path}" 2>/dev/null)" || [[ ! -d "${resolved_target}" ]]; then
    printf 'Regression latest link is dangling or does not target a directory: %s\n' \
      "${latest_path}" >&2
    return 2
  fi

  phase_root_resolved="$(realpath -e -- "${phase_root}")"
  if [[ "$(dirname -- "${resolved_target}")" != "${phase_root_resolved}" ||
        ! "$(basename -- "${resolved_target}")" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
    printf 'Regression latest must target a timestamped run under %s: %s\n' \
      "${phase_root_resolved}" "${latest_path}" >&2
    return 2
  fi

  checkpoint="${resolved_target}/checkpoints/WindForecastUNet.best.mdlus"
  if [[ ! -f "${checkpoint}" ]]; then
    printf 'Latest regression checkpoint does not exist: %s\n' "${checkpoint}" >&2
    return 2
  fi
  resolved_checkpoint="$(realpath -e -- "${checkpoint}")"
  printf '%s\n' "${resolved_checkpoint}"
}

wind_mark_latest() {
  local latest_path="$1"
  local target="$2"
  local parent temporary_dir

  parent="$(dirname -- "${latest_path}")"
  if [[ -e "${latest_path}" && ! -L "${latest_path}" ]]; then
    printf 'Refusing to replace non-symlink latest path: %s\n' "${latest_path}" >&2
    return 2
  fi

  temporary_dir="$(mktemp -d "${parent}/.latest.${target}.XXXXXX")"
  if ! ln -s -- "${target}" "${temporary_dir}/latest"; then
    rm -f -- "${temporary_dir}/latest"
    rmdir -- "${temporary_dir}"
    return 1
  fi
  if ! mv -Tf -- "${temporary_dir}/latest" "${latest_path}"; then
    rm -f -- "${temporary_dir}/latest"
    rmdir -- "${temporary_dir}"
    return 1
  fi
  rmdir -- "${temporary_dir}"
}
