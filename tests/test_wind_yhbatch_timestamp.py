from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts/yhbatch_wind_common.sh"
STAMP = "2026-11-04_16-30-00"


def _run_shell(source: str, *, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", "-c", f"set -euo pipefail\nsource {shlex.quote(str(HELPER))}\n{source}"],
        cwd=REPO,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def _run_script(
    script: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
):
    return subprocess.run(
        [str(script), *args],
        cwd=REPO,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _make_regression_latest(root: Path, *, checkpoint: bool = True) -> Path:
    run_dir = root / "wind_regression" / STAMP
    run_dir.mkdir(parents=True)
    model_path = run_dir / "checkpoints/WindForecastUNet.best.mdlus"
    if checkpoint:
        model_path.parent.mkdir()
        model_path.write_text("regression-model", encoding="utf-8")
    (root / "wind_regression/latest").symlink_to(STAMP)
    return model_path


def _install_yhbatch_stub(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "yhbatch.args"
    _write_executable(
        bin_dir / "yhbatch",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{\n"
        "  printf 'WIND_OUTPUTS_ROOT=%s\\n' \"${WIND_OUTPUTS_ROOT:-}\"\n"
        "  printf 'ARG=%s\\n' \"$@\"\n"
        f"}} > {shlex.quote(str(capture))}\n",
    )
    return bin_dir, capture, {"PATH": f"{bin_dir}:{os.environ['PATH']}"}


def _install_job_stubs(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_capture = tmp_path / "python.args"
    env_root = tmp_path / "environment"

    _write_executable(bin_dir / "module", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        bin_dir / "conda",
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == shell.bash && ${2:-} == hook ]]; then\n"
        "  printf 'conda() { :; }\\n'\n"
        "fi\n",
    )
    _write_executable(
        bin_dir / "python",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "run_dir=\n"
        f"capture={shlex.quote(str(python_capture))}\n"
        ": > \"${capture}\"\n"
        "for argument in \"$@\"; do\n"
        "  printf 'ARG=%s\\n' \"${argument}\" >> \"${capture}\"\n"
        "  case \"${argument}\" in\n"
        "    training.run_dir=*) run_dir=\"${argument#training.run_dir=}\" ;;\n"
        "  esac\n"
        "done\n"
        "if [[ -n \"${run_dir}\" ]]; then\n"
        "  mkdir -p \"${run_dir}/checkpoints\"\n"
        "fi\n",
    )
    return (
        bin_dir,
        python_capture,
        env_root,
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SWAITHER_WIND_ROOT": str(env_root),
        },
    )


def _captured_args(path: Path) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    args: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("WIND_OUTPUTS_ROOT="):
            env["WIND_OUTPUTS_ROOT"] = line.split("=", 1)[1]
        elif line.startswith("ARG="):
            args.append(line[4:])
    return env, args


def test_timestamped_yhbatch_scripts_parse_and_are_executable():
    for name in (
        "yhbatch_wind_common.sh",
        "yhbatch_train_wind_regression.sh",
        "yhbatch_train_wind_diffusion.sh",
        "submit_yhbatch_train_wind_regression.sh",
        "submit_yhbatch_train_wind_diffusion.sh",
        "yhbatch_evaluate_wind_test.sh",
    ):
        path = REPO / "scripts" / name
        subprocess.run(["bash", "-n", str(path)], check=True)
    for name in (
        "yhbatch_train_wind_regression.sh",
        "yhbatch_train_wind_diffusion.sh",
        "submit_yhbatch_train_wind_regression.sh",
        "submit_yhbatch_train_wind_diffusion.sh",
        "yhbatch_evaluate_wind_test.sh",
    ):
        assert os.access(REPO / "scripts" / name, os.X_OK)


def test_evaluate_wrapper_defaults_to_latest_regression_checkpoint():
    script = (REPO / "scripts/yhbatch_evaluate_wind_test.sh").read_text(encoding="utf-8")
    assert (
        'regression_checkpoint="${REGRESSION_CHECKPOINT:-'
        'outputs/wind_regression/latest/checkpoints/WindForecastUNet.best.mdlus}"'
    ) in script
    assert (
        "${REGRESSION_CHECKPOINT:-outputs/wind_regression/checkpoints/"
        not in script
    )


def test_run_stamp_is_validated_and_preserved():
    result = _run_shell("wind_resolve_run_stamp", env={"RUN_STAMP": STAMP})
    assert result.returncode == 0
    assert result.stdout == f"{STAMP}\n"

    invalid = _run_shell("wind_resolve_run_stamp", env={"RUN_STAMP": "../escape"})
    assert invalid.returncode != 0
    assert "Invalid RUN_STAMP" in invalid.stderr


def test_output_root_parser_defaults_relative_and_absolute(tmp_path):
    default = _run_shell(
        "wind_parse_outputs /repo /repo/outputs keep=1\n"
        "printf 'ROOT=%s\\n' \"${WIND_OUTPUTS_ROOT}\"\n"
        "printf 'FWD=%s\\n' \"${WIND_FORWARD_ARGS[@]}\""
    )
    assert default.returncode == 0
    assert default.stdout.splitlines() == ["ROOT=/repo/outputs", "FWD=keep=1"]

    relative = _run_shell(
        "wind_parse_outputs /repo /repo/outputs paths.outputs=output/xx keep=1\n"
        "printf 'ROOT=%s\\n' \"${WIND_OUTPUTS_ROOT}\"\n"
        "printf 'FWD=%s\\n' \"${WIND_FORWARD_ARGS[@]}\""
    )
    assert relative.returncode == 0
    assert relative.stdout.splitlines() == ["ROOT=/repo/output/xx", "FWD=keep=1"]

    absolute_root = tmp_path / "absolute output"
    absolute = _run_shell(
        f"wind_parse_outputs /repo /repo/outputs {shlex.quote(f'paths.outputs={absolute_root}')}\n"
        "printf 'ROOT=%s\\n' \"${WIND_OUTPUTS_ROOT}\""
    )
    assert absolute.returncode == 0
    assert absolute.stdout == f"ROOT={absolute_root}\n"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["paths.outputs"], "requires a non-empty value"),
        (["paths.outputs="], "requires a non-empty value"),
        (["paths.outputs=a", "paths.outputs=b"], "at most once"),
        (["paths.outputs=bad\npath"], "invalid newline"),
    ],
)
def test_output_root_parser_rejects_invalid_values(arguments, message):
    command = " ".join(shlex.quote(argument) for argument in arguments)
    result = _run_shell(f"wind_parse_outputs /repo /repo/outputs {command}")
    assert result.returncode != 0
    assert message in result.stderr


def test_output_root_parser_rejects_an_existing_file(tmp_path):
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("file", encoding="utf-8")
    result = _run_shell(
        f"wind_parse_outputs /repo /repo/outputs {shlex.quote(f'paths.outputs={output_file}')}"
    )
    assert result.returncode != 0
    assert "Output root is not a directory" in result.stderr


def test_training_paths_follow_the_output_root():
    result = _run_shell(
        "wind_slurm_log_path /custom/output wind_regression_train "
        f"{shlex.quote(STAMP)}\n"
        "wind_training_run_dir /custom/output wind_regression "
        f"{shlex.quote(STAMP)}\n"
        "wind_training_latest_path /custom/output wind_regression"
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        f"/custom/output/slurm/wind_regression_train_{STAMP}.out",
        f"/custom/output/wind_regression/{STAMP}",
        "/custom/output/wind_regression/latest",
    ]


@pytest.mark.parametrize(
    "argument",
    [
        "hydra.run.dir=elsewhere",
        "training.run_dir=elsewhere",
        "training.checkpoint_dir=elsewhere",
        "training.tensorboard_dir=elsewhere",
    ],
)
def test_timestamped_jobs_reject_managed_path_overrides(argument):
    result = _run_shell(f"wind_reject_path_overrides {shlex.quote(argument)}")
    assert result.returncode != 0
    assert "managed by the timestamped wrapper" in result.stderr


def test_regression_latest_resolves_to_concrete_checkpoint(tmp_path):
    root = tmp_path / "output/xx"
    model_path = _make_regression_latest(root)
    result = _run_shell(
        f"wind_resolve_regression_latest_checkpoint {shlex.quote(str(root))}"
    )
    assert result.returncode == 0
    assert result.stdout == f"{model_path.resolve()}\n"


def test_regression_latest_rejects_missing_dangling_and_missing_checkpoint(tmp_path):
    missing = _run_shell(
        f"wind_resolve_regression_latest_checkpoint {shlex.quote(str(tmp_path / 'missing'))}"
    )
    assert missing.returncode != 0
    assert "latest link not found" in missing.stderr

    dangling_root = tmp_path / "dangling"
    (dangling_root / "wind_regression").mkdir(parents=True)
    (dangling_root / "wind_regression/latest").symlink_to(STAMP)
    dangling = _run_shell(
        f"wind_resolve_regression_latest_checkpoint {shlex.quote(str(dangling_root))}"
    )
    assert dangling.returncode != 0
    assert "dangling" in dangling.stderr

    incomplete_root = tmp_path / "incomplete"
    _make_regression_latest(incomplete_root, checkpoint=False)
    incomplete = _run_shell(
        f"wind_resolve_regression_latest_checkpoint {shlex.quote(str(incomplete_root))}"
    )
    assert incomplete.returncode != 0
    assert "Latest regression checkpoint does not exist" in incomplete.stderr


def test_explicit_regression_checkpoint_is_detected_and_validated(tmp_path):
    checkpoint = tmp_path / "checkpoints/model.mdlus"
    checkpoint.parent.mkdir()
    checkpoint.write_text("explicit-model", encoding="utf-8")
    argument = f"training.regression_checkpoint={checkpoint}"
    result = _run_shell(
        f"wind_find_regression_checkpoint /repo {shlex.quote(argument)}\n"
        "printf 'HAS=%s\\nVALUE=%s\\n' \"${WIND_HAS_REGRESSION_CHECKPOINT}\" "
        "\"${WIND_REGRESSION_CHECKPOINT}\""
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["HAS=1", f"VALUE={checkpoint}"]

    absent = _run_shell(
        "wind_find_regression_checkpoint /repo keep=1\n"
        "printf 'HAS=%s\\n' \"${WIND_HAS_REGRESSION_CHECKPOINT}\""
    )
    assert absent.returncode == 0
    assert absent.stdout == "HAS=0\n"

    invalid = _run_shell(
        f"wind_find_regression_checkpoint /repo {shlex.quote('training.regression_checkpoint=/missing/model')}"
    )
    assert invalid.returncode != 0
    assert "Regression checkpoint does not exist" in invalid.stderr


def test_latest_link_is_replaced_only_when_it_is_a_symlink(tmp_path):
    run_root = tmp_path / "wind_regression"
    run_root.mkdir()
    target = run_root / STAMP
    target.mkdir()
    latest = run_root / "latest"
    latest.symlink_to("2026-11-03_10-00-00")

    result = _run_shell(
        f"wind_mark_latest {shlex.quote(str(latest))} {shlex.quote(target.name)}"
    )
    assert result.returncode == 0
    assert latest.is_symlink()
    assert latest.readlink() == Path(target.name)

    unsafe = tmp_path / "unsafe/latest"
    unsafe.mkdir(parents=True)
    result = _run_shell(
        f"wind_mark_latest {shlex.quote(str(unsafe))} {shlex.quote(target.name)}"
    )
    assert result.returncode != 0
    assert "Refusing to replace non-symlink" in result.stderr


def test_regression_submit_uses_custom_root_and_strips_output_argument(tmp_path):
    bin_dir, capture, env = _install_yhbatch_stub(tmp_path)
    root = tmp_path / "output/xx"
    result = _run_script(
        REPO / "scripts/submit_yhbatch_train_wind_regression.sh",
        [f"paths.outputs={root}", "model.batch_size=2", "note=hello world"],
        env={**env, "RUN_STAMP": STAMP},
    )
    assert result.returncode == 0, result.stderr
    captured_env, args = _captured_args(capture)
    assert captured_env["WIND_OUTPUTS_ROOT"] == str(root)
    assert f"{root}/slurm/wind_regression_train_{STAMP}.out" in args
    assert f"paths.outputs={root}" not in args
    assert "model.batch_size=2" in args
    assert "note=hello world" in args
    assert args[-1] == "note=hello world"
    assert bin_dir.is_dir()


def test_diffusion_submit_resolves_latest_to_timestamp_checkpoint(tmp_path):
    bin_dir, capture, env = _install_yhbatch_stub(tmp_path)
    root = tmp_path / "output/xx"
    model_path = _make_regression_latest(root)
    result = _run_script(
        REPO / "scripts/submit_yhbatch_train_wind_diffusion.sh",
        [f"paths.outputs={root}", "model.batch_size=2"],
        env={**env, "RUN_STAMP": STAMP},
    )
    assert result.returncode == 0, result.stderr
    captured_env, args = _captured_args(capture)
    assert captured_env["WIND_OUTPUTS_ROOT"] == str(root)
    assert f"{root}/slurm/wind_diffusion_train_{STAMP}.out" in args
    assert f"paths.outputs={root}" not in args
    assert f"training.regression_checkpoint={model_path.resolve()}" in args
    assert str(model_path) in result.stdout
    assert bin_dir.is_dir()


def test_diffusion_submit_explicit_checkpoint_overrides_missing_latest(tmp_path):
    _, capture, env = _install_yhbatch_stub(tmp_path)
    root = tmp_path / "output/xx"
    checkpoint = tmp_path / "explicit/model.mdlus"
    checkpoint.parent.mkdir()
    checkpoint.write_text("explicit-model", encoding="utf-8")
    result = _run_script(
        REPO / "scripts/submit_yhbatch_train_wind_diffusion.sh",
        [
            f"paths.outputs={root}",
            f"training.regression_checkpoint={checkpoint}",
        ],
        env={**env, "RUN_STAMP": STAMP},
    )
    assert result.returncode == 0, result.stderr
    _, args = _captured_args(capture)
    assert f"training.regression_checkpoint={checkpoint}" in args
    assert sum(arg.startswith("training.regression_checkpoint=") for arg in args) == 1


def test_diffusion_submit_reports_missing_latest(tmp_path):
    _, capture, env = _install_yhbatch_stub(tmp_path)
    root = tmp_path / "output/xx"
    result = _run_script(
        REPO / "scripts/submit_yhbatch_train_wind_diffusion.sh",
        [f"paths.outputs={root}"],
        env={**env, "RUN_STAMP": STAMP},
    )
    assert result.returncode != 0
    assert f"Regression latest link not found: {root}/wind_regression/latest" in result.stderr
    assert not capture.exists()


def test_regression_job_uses_forwarded_root_for_managed_paths(tmp_path):
    _, capture, env_root, env = _install_job_stubs(tmp_path)
    root = tmp_path / "output/xx"
    result = _run_script(
        REPO / "scripts/yhbatch_train_wind_regression.sh",
        ["model.batch_size=2"],
        env={
            **env,
            "WIND_OUTPUTS_ROOT": str(root),
            "RUN_STAMP": STAMP,
            "PYTHON_CAPTURE": str(capture),
        },
    )
    assert result.returncode == 0, result.stderr
    _, args = _captured_args(capture)
    run_dir = root / "wind_regression" / STAMP
    assert f"hydra.run.dir={run_dir}" in args
    assert f"training.run_dir={run_dir}" in args
    assert f"training.checkpoint_dir={run_dir}/checkpoints" in args
    assert f"training.tensorboard_dir={run_dir}/tensorboard" in args
    assert "model.batch_size=2" in args
    latest = root / "wind_regression/latest"
    assert latest.is_symlink()
    assert latest.readlink() == Path(STAMP)
    assert (env_root / "outputs").is_dir()


def test_diffusion_job_resolves_latest_and_updates_diffusion_latest(tmp_path):
    _, capture, _, env = _install_job_stubs(tmp_path)
    root = tmp_path / "output/xx"
    model_path = _make_regression_latest(root)
    result = _run_script(
        REPO / "scripts/yhbatch_train_wind_diffusion.sh",
        ["model.batch_size=2"],
        env={
            **env,
            "WIND_OUTPUTS_ROOT": str(root),
            "RUN_STAMP": STAMP,
            "PYTHON_CAPTURE": str(capture),
        },
    )
    assert result.returncode == 0, result.stderr
    _, args = _captured_args(capture)
    run_dir = root / "wind_diffusion" / STAMP
    assert f"training.regression_checkpoint={model_path.resolve()}" in args
    assert f"hydra.run.dir={run_dir}" in args
    assert f"training.run_dir={run_dir}" in args
    assert f"training.checkpoint_dir={run_dir}/checkpoints" in args
    assert f"training.tensorboard_dir={run_dir}/tensorboard" in args
    latest = root / "wind_diffusion/latest"
    assert latest.is_symlink()
    assert latest.readlink() == Path(STAMP)


def test_consumers_keep_legacy_checkpoint_defaults():
    regression = OmegaConf.load(REPO / "conf/config_training_wind_regression.yaml")
    diffusion = OmegaConf.load(REPO / "conf/config_training_wind_diffusion.yaml")
    inference = OmegaConf.load(REPO / "conf/config_inference_wind_pipeline.yaml")

    assert "outputs/wind_regression/latest" not in OmegaConf.to_yaml(regression)
    assert "outputs/wind_regression/latest" not in OmegaConf.to_yaml(diffusion)
    assert "outputs/wind_regression/latest" not in OmegaConf.to_yaml(inference)
    assert str(diffusion.training.regression_checkpoint) == (
        "outputs/wind_regression/checkpoints/WindForecastUNet.best.mdlus"
    )
    assert str(inference.checkpoints.regression) == (
        "outputs/wind_regression/checkpoints/WindForecastUNet.best.mdlus"
    )
