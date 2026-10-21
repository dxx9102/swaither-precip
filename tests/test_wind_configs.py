from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data62T/dengxian")


def _assert_storage_path(value) -> None:
    path = Path(str(value)).resolve()
    assert path == DATA_ROOT or DATA_ROOT in path.parents


def test_wind_year_splits_and_storage_paths():
    step1 = OmegaConf.load(REPO / "conf/config_training_wind_step1_unet.yaml")
    assert list(step1.dataset.years) == [2019, 2020, 2021, 2022]
    assert list(step1.validation.dataset.years) == [2023]
    _assert_storage_path(step1.dataset.stats_path)
    _assert_storage_path(step1.training.run_dir)
    _assert_storage_path(step1.training.checkpoint_dir)
    assert step1.dataset.history_hours == 6

    for name in (
        "config_training_wind_step2_regression.yaml",
        "config_training_wind_step2_diffusion.yaml",
    ):
        cfg = OmegaConf.load(REPO / "conf" / name)
        assert list(cfg.training.hp.split.train_years) == [2019, 2020, 2021, 2022]
        assert list(cfg.training.hp.split.validation_years) == [2023]
        assert list(cfg.training.hp.split.test_years) == [2024]
        assert list(cfg.dataset.leads) == [1, 2, 3, 4, 5, 6]
        assert cfg.dataset.crop_mult == 16
        if "diffusion" in name:
            assert int(cfg.training.hp.patch_shape_y) <= 272
            assert int(cfg.training.hp.patch_shape_x) <= 320
        _assert_storage_path(cfg.dataset.stats_path)
        _assert_storage_path(cfg.training.io.checkpoint_dir)
        _assert_storage_path(cfg.training.io.tensorboard_dir)


def test_wind_configs_do_not_write_into_home_directory():
    for path in (REPO / "conf").glob("*wind*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "~" not in text
