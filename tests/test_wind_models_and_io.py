from __future__ import annotations

import importlib.util
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from physicsnemo import Module
from physicsnemo.metrics.diffusion import RegressionLoss
from physicsnemo.utils.corrdiff import diffusion_step, regression_step
from physicsnemo.utils.diffusion import stochastic_sampler

from swaither.models.wind_forecast import (
    WindCorrDiffUNet,
    WindEDMPrecondSuperResolution,
    WindForecastUNet,
)
from swaither.wind_helpers import (
    crps_ensemble,
    ensemble_spread_skill,
    write_ensemble_netcdf,
    write_forecast_netcdf,
)

from .test_wind_data import (
    _grid,
    _stats,
    _write_condition,
    _write_high,
    _write_low,
    _write_static,
)


def _tiny_forecast_model(forecast_hours: int = 1) -> WindForecastUNet:
    return WindForecastUNet(
        18,
        forecast_hours=forecast_hours,
        wind_components=2,
        unet_kwargs={
            "model_channels": 8,
            "channel_mult": [1, 2, 2],
            "attn_resolutions": [],
            "embedding_type": "zero",
            "img_resolution": [32, 32],
            "use_fp16": False,
            "checkpoint_level": 0,
            "use_apex_gn": False,
        },
    )


def test_forecast_forward_backward_and_checkpoint_roundtrip(tmp_path):
    model = _tiny_forecast_model()
    x = torch.randn(2, 18, 29, 33)
    target = torch.randn(2, 2, 29, 33)
    prediction = model(torch.zeros_like(target), x)
    assert prediction.shape == target.shape
    assert model.img_out_channels == 2
    assert model.__model_checkpoint_version__ == 2
    torch.mean((prediction - target) ** 2).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    checkpoint = tmp_path / "wind_unet.mdlus"
    model.save(str(checkpoint))
    restored = Module.from_checkpoint(str(checkpoint), {}).eval()
    with torch.no_grad():
        np.testing.assert_allclose(
            restored(torch.zeros_like(target), x).numpy(),
            model.eval()(torch.zeros_like(target), x).detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


def test_72_hour_forecast_forward_and_checkpoint_roundtrip(tmp_path):
    model = _tiny_forecast_model(forecast_hours=72)
    x = torch.randn(2, 18, 29, 33)
    target = torch.randn(2, 144, 29, 33)
    prediction = model(torch.zeros_like(target), x)
    assert prediction.shape == target.shape
    assert model.img_out_channels == 144
    torch.mean((prediction - target) ** 2).backward()

    checkpoint = tmp_path / "wind_unet_72h.mdlus"
    model.save(str(checkpoint))
    restored = Module.from_checkpoint(str(checkpoint), {}).eval()
    assert restored.forecast_hours == 72
    with torch.no_grad():
        np.testing.assert_allclose(
            restored(torch.zeros_like(target), x).numpy(),
            model.eval()(torch.zeros_like(target), x).detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


def test_forecast_model_overfits_a_fixed_small_batch():
    torch.manual_seed(7)
    model = _tiny_forecast_model()
    x = torch.randn(2, 18, 29, 33)
    y = torch.randn(2, 2, 29, 33)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial_loss = None
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(torch.zeros_like(y), x)
        loss = torch.mean((prediction - y) ** 2)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.6 * initial_loss


def test_corrdiff_regression_and_diffusion_are_two_channel():
    common = {
        "img_resolution": [16, 16],
        "model_channels": 8,
        "channel_mult": [1, 2],
        "attn_resolutions": [],
    }
    regression = WindCorrDiffUNet(
        img_in_channels=8,
        img_out_channels=2,
        embedding_type="zero",
        **common,
    )
    x = torch.randn(1, 8, 16, 16)
    y = torch.randn(1, 2, 16, 16)
    assert regression(torch.zeros_like(y), x).shape == y.shape
    loss = RegressionLoss()(net=regression, img_clean=y, img_lr=x)
    loss.mean().backward()

    diffusion = WindEDMPrecondSuperResolution(
        img_in_channels=10,
        img_out_channels=2,
        embedding_type="positional",
        **common,
    )
    condition = torch.randn(1, 10, 16, 16)
    sigma = torch.ones(1, 1, 1, 1)
    generated = diffusion(torch.randn_like(y), condition, sigma)
    assert generated.shape == y.shape


def _load_export_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_wind_forecasts.py"
    spec = importlib.util.spec_from_file_location("wind_export", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_training_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_wind_step1.py"
    spec = importlib.util.spec_from_file_location("wind_train_step1", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validation_reports_absolute_metrics_for_every_lead(tmp_path):
    from torch.utils.data import DataLoader

    from swaither.datasets.wind import WindForecastDataset

    lat = _grid(3, 1.0)
    lon = _grid(4, 1.0)
    times = pd.date_range("2020-01-01", periods=78, freq="h")
    low = tmp_path / "validation_low.nc"
    static = tmp_path / "validation_static.nc"
    stats = tmp_path / "validation_stats.json"
    _write_low(low, times, lat, lon)
    _write_static(static, lat, lon)
    probe = WindForecastDataset(
        low,
        static,
        forecast_horizon_hours=72,
        years=[2020],
        normalize=False,
    )
    _stats(stats, probe.physical_input_names, probe.output_variables)
    dataset = WindForecastDataset(
        low,
        static,
        forecast_horizon_hours=72,
        years=[2020],
        stats_path=stats,
    )
    loader = DataLoader(dataset, batch_size=1)
    training = _load_training_module()
    loss, overall, per_lead = training._validation(
        _tiny_forecast_model(forecast_hours=72).eval(),
        loader,
        torch.device("cpu"),
        dataset,
        amp=False,
    )

    assert loss >= 0
    assert "vector_rmse" in overall
    assert list(per_lead) == [str(lead) for lead in range(1, 73)]
    assert all("vector_rmse" in metrics for metrics in per_lead.values())
    assert all("normalized_residual_mse" in metrics for metrics in per_lead.values())


def test_one_shot_72_hour_conditioning_and_end_to_end_shapes(tmp_path):
    from swaither.datasets.wind import WindForecastDataset, WindSuperResolutionDataset

    low_lat = _grid(3, 1.0)
    low_lon = _grid(4, 1.0)
    high_lat = _grid(17, 1.0)
    high_lon = _grid(17, 1.0)
    low_times = pd.date_range("2020-01-01", periods=78, freq="h")
    high_times = pd.date_range("2020-01-01", periods=100, freq="h")
    low = tmp_path / "low.nc"
    static_low = tmp_path / "static_low.nc"
    high = tmp_path / "high.nc"
    static_high = tmp_path / "static_high.nc"
    condition = tmp_path / "conditions.nc"
    _write_low(low, low_times, low_lat, low_lon)
    _write_static(static_low, low_lat, low_lon)
    _write_high(high, high_times, high_lat, high_lon)
    _write_static(static_high, high_lat, high_lon)

    forecast_probe = WindForecastDataset(
        low,
        static_low,
        forecast_horizon_hours=72,
        years=[2020],
        normalize=False,
    )
    forecast_stats = tmp_path / "forecast_stats.json"
    _stats(
        forecast_stats,
        forecast_probe.physical_input_names,
        forecast_probe.output_variables,
    )
    forecast_data = WindForecastDataset(
        low,
        static_low,
        forecast_horizon_hours=72,
        years=[2020],
        stats_path=forecast_stats,
    )
    export = _load_export_module()
    model = _tiny_forecast_model(forecast_hours=72).eval()
    forward_calls = 0

    def count_forward(_module, _inputs, _output):
        nonlocal forward_calls
        forward_calls += 1

    model.register_forward_hook(count_forward)
    u, v, initial_times = export.forecast(
        model,
        forecast_data,
        [5, 6, 7],
        batch_size=2,
        device=torch.device("cpu"),
        amp=False,
    )
    assert forward_calls == 2
    assert u.shape == v.shape == (3, 72, 3, 4)
    write_forecast_netcdf(
        condition,
        u,
        v,
        initial_times,
        list(range(1, 73)),
        forecast_data.latitude(),
        forecast_data.longitude(),
    )
    with xr.open_dataset(condition, engine="netcdf4") as ds:
        assert ds.u10.dims == ("initial_time", "lead_hour", "lat", "lon")
        assert ds.u10.units == "m s-1"
        assert ds.valid_time.dims == ("initial_time", "lead_hour")
        assert ds.sizes["lead_hour"] == 72
        np.testing.assert_array_equal(ds.lead_hour.values, np.arange(1, 73))
        assert str(ds.valid_time.values[0, 0]) == "2020-01-01T06:00:00Z"
        assert str(ds.valid_time.values[0, -1]) == "2020-01-04T05:00:00Z"

    sr_data = WindSuperResolutionDataset(
        condition,
        high,
        static_high,
        leads=(1, 2, 3),
        condition_sampling="all",
        normalize=False,
    )
    y, x = sr_data[0]
    assert y.shape == (2, 16, 16)
    assert x.shape == (8, 16, 16)
    regression = WindCorrDiffUNet(
        img_resolution=[16, 16],
        img_in_channels=8,
        img_out_channels=2,
        model_channels=8,
        channel_mult=[1, 2],
        attn_resolutions=[],
        embedding_type="zero",
    ).eval()
    mean = regression_step(regression, x.unsqueeze(0), latents_shape=(1, 2, 16, 16))
    assert mean.shape == (1, 2, 16, 16)
    diffusion = WindEDMPrecondSuperResolution(
        img_resolution=[16, 16],
        img_in_channels=10,
        img_out_channels=2,
        model_channels=8,
        channel_mult=[1, 2],
        attn_resolutions=[],
        embedding_type="positional",
    ).eval()
    ensemble = diffusion_step(
        net=diffusion,
        sampler_fn=partial(stochastic_sampler, num_steps=2),
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[123, 456]],
        img_lr=x.unsqueeze(0).expand(2, -1, -1, -1).contiguous(),
        rank=0,
        device=torch.device("cpu"),
        mean_hr=mean,
    )
    assert ensemble.shape == (2, 2, 16, 16)
    repeated = diffusion_step(
        net=diffusion,
        sampler_fn=partial(stochastic_sampler, num_steps=2),
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[123, 456]],
        img_lr=x.unsqueeze(0).expand(2, -1, -1, -1).contiguous(),
        rank=0,
        device=torch.device("cpu"),
        mean_hr=mean,
    )
    np.testing.assert_allclose(repeated.numpy(), ensemble.numpy(), rtol=0, atol=0)


def test_netcdf_output_metrics_and_crps(tmp_path):
    initial = pd.date_range("2024-01-01", periods=2, freq="h")
    lat, lon, leads = _grid(3, 1.0), _grid(4, 1.0), [1, 2]
    deterministic = tmp_path / "deterministic.nc"
    ensemble_file = tmp_path / "ensemble.nc"
    write_forecast_netcdf(deterministic, np.ones((2, 2, 3, 4)), -np.ones((2, 2, 3, 4)), initial, leads, lat, lon)
    write_ensemble_netcdf(ensemble_file, np.ones((2, 2, 3, 3, 4)), -np.ones((2, 2, 3, 3, 4)), initial, leads, lat, lon)
    with xr.open_dataset(ensemble_file, engine="netcdf4") as ds:
        assert ds.u10.dims == ("initial_time", "lead_hour", "ensemble", "lat", "lon")
        assert ds.u10.units == "m s-1"

    members = np.arange(12, dtype=np.float64).reshape(12, 1)
    truth = np.array([5.5])
    assert 0 < crps_ensemble(members, truth) < 10
    spread, skill = ensemble_spread_skill(members, truth)
    assert spread == pytest.approx(np.std(members))
    assert skill == pytest.approx(abs(np.mean(members) - truth[0]))
