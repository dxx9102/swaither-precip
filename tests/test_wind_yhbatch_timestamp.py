from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts/yhbatch_wind_common.sh"


def _run_shell(source: str, *, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", "-c", f"set -euo pipefail\nsource {shlex.quote(str(HELPER))}\n{source}"],
        cwd=REPO,
        env={**os.environ, **(env or {})},
        text=True,
        capture_output=True,
        check=False,
    )


def test_timestamped_yhbatch_scripts_parse_and_are_executable():
    for name in (
        "yhbatch_wind_common.sh",
        "yhbatch_train_wind_regression.sh",
        "yhbatch_train_wind_diffusion.sh",
        "submit_yhbatch_train_wind_regression.sh",
        "submit_yhbatch_train_wind_diffusion.sh",
    ):
        path = REPO / "scripts" / name
        subprocess.run(["bash", "-n", str(path)], check=True)
    for name in (
        "yhbatch_train_wind_regression.sh",
        "yhbatch_train_wind_diffusion.sh",
        "submit_yhbatch_train_wind_regression.sh",
        "submit_yhbatch_train_wind_diffusion.sh",
    ):
        assert os.access(REPO / "scripts" / name, os.X_OK)


def test_run_stamp_is_validated_and_preserved():
    stamp = "2026-11-04_16-30-00"
    result = _run_shell("wind_resolve_run_stamp", env={"RUN_STAMP": stamp})
    assert result.returncode == 0
    assert result.stdout == f"{stamp}\n"

    invalid = _run_shell("wind_resolve_run_stamp", env={"RUN_STAMP": "../escape"})
    assert invalid.returncode != 0
    assert "Invalid RUN_STAMP" in invalid.stderr


def test_log_and_training_directory_share_the_same_stamp_contract():
    stamp = "2026-11-04_16-30-00"
    result = _run_shell(
        "wind_slurm_log_path /repo wind_regression_train "
        f"{shlex.quote(stamp)}\n"
        "wind_training_run_dir /repo wind_regression "
        f"{shlex.quote(stamp)}"
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "/repo/outputs/slurm/wind_regression_train_2026-11-04_16-30-00.out",
        "/repo/outputs/wind_regression/2026-11-04_16-30-00",
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
    result = _run_shell(
        f"wind_reject_path_overrides {shlex.quote(argument)}"
    )
    assert result.returncode != 0
    assert "managed by the timestamped wrapper" in result.stderr


def test_latest_link_is_replaced_only_when_it_is_a_symlink(tmp_path):
    run_root = tmp_path / "wind_regression"
    run_root.mkdir()
    target = run_root / "2026-11-04_16-30-00"
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
