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

wind_training_run_dir() {
  printf '%s/outputs/%s/%s\n' "$1" "$2" "$3"
}

wind_slurm_log_path() {
  printf '%s/outputs/slurm/%s_%s.out\n' "$1" "$2" "$3"
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
