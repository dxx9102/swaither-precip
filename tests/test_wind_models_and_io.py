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
    _write_condition,
    _write_high,
    _write_low,
    _write_static,
)


def _tiny_forecast_model() -> WindForecastUNet:
    return WindForecastUNet(
        18,
        2,
        {
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


def test_autoregressive_rollout_conditioning_and_end_to_end_shapes(tmp_path):
    from swaither.datasets.wind import WindForecastDataset, WindSuperResolutionDataset

    low_lat = _grid(3, 1.0)
    low_lon = _grid(4, 1.0)
    high_lat = _grid(17, 1.0)
    high_lon = _grid(17, 1.0)
    low_times = pd.date_range("2020-01-01", periods=13, freq="h")
    high_times = pd.date_range("2020-01-01", periods=18, freq="h")
    low = tmp_path / "low.nc"
    static_low = tmp_path / "static_low.nc"
    high = tmp_path / "high.nc"
    static_high = tmp_path / "static_high.nc"
    condition = tmp_path / "conditions.nc"
    _write_low(low, low_times, low_lat, low_lon)
    _write_static(static_low, low_lat, low_lon)
    _write_high(high, high_times, high_lat, high_lon)
    _write_static(static_high, high_lat, high_lon)

    forecast_data = WindForecastDataset(low, static_low, years=[2020], normalize=False)
    export = _load_export_module()
    u, v, initial_times = export.rollout(
        _tiny_forecast_model().eval(),
        forecast_data,
        [5, 6],
        max_lead=6,
        batch_size=2,
        device=torch.device("cpu"),
        amp=False,
    )
    assert u.shape == v.shape == (2, 6, 3, 4)
    write_forecast_netcdf(
        condition,
        u,
        v,
        initial_times,
        [1, 2, 3, 4, 5, 6],
        forecast_data.latitude(),
        forecast_data.longitude(),
    )
    with xr.open_dataset(condition, engine="netcdf4") as ds:
        assert ds.u10.dims == ("initial_time", "lead_hour", "lat", "lon")
        assert ds.u10.units == "m s-1"
        assert ds.valid_time.dims == ("initial_time", "lead_hour")

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
