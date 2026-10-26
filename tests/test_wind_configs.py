from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = (REPO.parent.parent / "data" / "uv10").resolve()


def _config_path(value) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def _assert_below(value, root: Path) -> None:
    path = _config_path(value)
    assert path == root or root in path.parents


def test_wind_year_splits_and_storage_paths():
    step1 = OmegaConf.load(REPO / "conf/config_training_wind_step1_unet.yaml")
    assert _config_path(step1.paths.raw_data_root) == RAW_DATA_ROOT
    assert list(step1.dataset.years) == [2019, 2020, 2021, 2022]
    assert list(step1.validation.dataset.years) == [2023]
    for value in step1.dataset.low_res_files:
        _assert_below(value, RAW_DATA_ROOT)
    _assert_below(step1.dataset.static_low_file, RAW_DATA_ROOT)
    _assert_below(step1.dataset.stats_path, REPO)
    _assert_below(step1.training.run_dir, REPO)
    _assert_below(step1.training.checkpoint_dir, REPO)
    assert step1.dataset.history_hours == 6
    assert step1.dataset.forecast_horizon_hours == 72
    assert step1.validation.dataset.forecast_horizon_hours == 72
    assert "_72h" in str(step1.dataset.stats_path)
    assert "_72h" in str(step1.training.run_dir)

    step1_stats = OmegaConf.load(REPO / "conf/config_compute_wind_stats.yaml")
    assert step1_stats.dataset.forecast_horizon_hours == 72
    assert "_72h" in str(step1_stats.output_file)

    step1_export = OmegaConf.load(REPO / "conf/config_export_wind_forecasts.yaml")
    assert step1_export.forecast_hours == 72
    assert step1_export.dataset.forecast_horizon_hours == 72

    for name in (
        "config_training_wind_step2_regression.yaml",
        "config_training_wind_step2_diffusion.yaml",
    ):
        cfg = OmegaConf.load(REPO / "conf" / name)
        assert list(cfg.training.hp.split.train_years) == [2019, 2020, 2021, 2022]
        assert list(cfg.training.hp.split.validation_years) == [2023]
        assert list(cfg.training.hp.split.test_years) == [2024]
        assert _config_path(cfg.paths.raw_data_root) == RAW_DATA_ROOT
        for value in cfg.dataset.high_res_files:
            _assert_below(value, RAW_DATA_ROOT)
        _assert_below(cfg.dataset.static_high_file, RAW_DATA_ROOT)
        assert list(cfg.dataset.leads) == [1, 2, 3, 4, 5, 6]
        assert cfg.dataset.crop_mult == 16
        if "diffusion" in name:
            assert int(cfg.training.hp.patch_shape_y) <= 272
            assert int(cfg.training.hp.patch_shape_x) <= 320
        _assert_below(cfg.dataset.stats_path, REPO)
        _assert_below(cfg.training.io.checkpoint_dir, REPO)
        _assert_below(cfg.training.io.tensorboard_dir, REPO)


def test_wind_raw_data_root_is_configurable():
    cfg = OmegaConf.load(REPO / "conf/config_training_wind_step1_unet.yaml")
    cfg.paths.raw_data_root = "/datasets/uv10"
    assert cfg.dataset.low_res_files[0] == "/datasets/uv10/uv10_2019_2022_clipped_LRes.nc"
    assert cfg.validation.dataset.static_low_file == "/datasets/uv10/fj_LRes.nc"


def test_wind_configs_do_not_write_into_home_directory():
    for path in (REPO / "conf").glob("*wind*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "~" not in text
        assert "/data62T/" not in text
