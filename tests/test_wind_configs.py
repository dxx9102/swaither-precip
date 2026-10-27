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


def test_wind_joint_configs_use_train_validation_test_year_splits():
    for name in (
        "config_training_wind_regression.yaml",
        "config_training_wind_diffusion.yaml",
    ):
        cfg = OmegaConf.load(REPO / "conf" / name)
        assert list(cfg.dataset.years) == [2019, 2020, 2021, 2022]
        assert list(cfg.validation.dataset.years) == [2023]
        assert cfg.dataset.history_hours == 6
        assert cfg.dataset.forecast_horizon_hours == 72
        assert cfg.dataset.grid_channels == 4
        assert "static_low_file" not in cfg.dataset
        assert "crop_mult" not in cfg.dataset
        assert all("_HR.nc" in str(path) for path in cfg.dataset.high_res_files)
        _assert_below(cfg.dataset.static_high_file, RAW_DATA_ROOT)
        _assert_below(cfg.dataset.stats_path, REPO)
        _assert_below(cfg.training.checkpoint_dir, REPO)
        _assert_below(cfg.training.tensorboard_dir, REPO)

    inference = OmegaConf.load(REPO / "conf/config_inference_wind_pipeline.yaml")
    assert list(inference.dataset.years) == [2024]
    assert list(inference.inference.diffusion_leads) == list(range(1, 73))
    assert inference.inference.initial_hour == 0
    assert inference.inference.sampler_num_steps >= 2
    assert "condition_files" not in inference.dataset
    assert "static_low_file" not in inference.dataset


def test_stats_and_diffusion_configs_cover_complete_lead_range():
    stats = OmegaConf.load(REPO / "conf/config_compute_wind_stats.yaml")
    assert list(stats.train_years) == [2019, 2020, 2021, 2022]
    assert stats.max_samples == 512
    assert "_HR.nc" in str(stats.dataset.static_high_file)
    assert "static_low_file" not in stats.dataset

    diffusion = OmegaConf.load(REPO / "conf/config_training_wind_diffusion.yaml")
    assert list(diffusion.dataset.leads) == list(range(1, 73))
    assert diffusion.model.hr_mean_conditioning is True
    assert diffusion.training.patch_shape_y <= 272
    assert diffusion.training.patch_shape_x <= 320


def test_old_wind_entrypoints_and_condition_files_are_removed():
    assert not (REPO / "conf/config_training_wind_step1_unet.yaml").exists()
    assert not (REPO / "conf/config_training_wind_step2_regression.yaml").exists()
    assert not (REPO / "conf/config_training_wind_step2_diffusion.yaml").exists()
    assert not (REPO / "conf/config_compute_wind_stats_step2.yaml").exists()
    assert not (REPO / "conf/config_export_wind_forecasts.yaml").exists()
    assert not (REPO / "scripts/train_wind_step1.py").exists()
    assert not (REPO / "scripts/export_wind_forecasts.py").exists()


def test_wind_configs_do_not_write_into_home_directory():
    for path in (REPO / "conf").glob("*wind*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert "~" not in text
        assert "/data62T/" not in text
