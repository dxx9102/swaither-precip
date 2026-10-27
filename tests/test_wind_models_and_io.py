from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import torch
import xarray as xr
from physicsnemo import Module
from physicsnemo.metrics.diffusion import ResidualLoss
from physicsnemo.utils.corrdiff import diffusion_step
from physicsnemo.utils.diffusion import stochastic_sampler

from swaither.models import (
    WindEDMPrecondSuperResolution,
    WindForecastUNet,
    WindRandomPatching2D,
    WindRegressionLeadAdapter,
)
from swaither.wind_helpers import (
    WindForecastNetCDFWriter,
    write_ensemble_netcdf,
    write_forecast_netcdf,
)


def _tiny_forecast_model(forecast_hours: int = 2) -> WindForecastUNet:
    return WindForecastUNet(
        img_in_channels=6,
        forecast_hours=forecast_hours,
        wind_components=2,
        unet_kwargs={
            "img_resolution": [3, 5],
            "model_channels": 8,
            "channel_mult": [1, 2],
            "attn_resolutions": [],
            "dropout": 0.0,
            "embedding_type": "zero",
            "use_apex_gn": False,
            "model_type": "SongUNet",
        },
    )


def test_forecast_unet_pads_crops_and_checkpoint_roundtrip(tmp_path):
    model = _tiny_forecast_model(forecast_hours=3).eval()
    x = torch.randn(2, 6, 3, 5)
    with torch.no_grad():
        prediction = model(torch.zeros(2, 6, 3, 5), x)
    assert prediction.shape == (2, 6, 3, 5)
    checkpoint = tmp_path / "regression.mdlus"
    model.save(str(checkpoint))
    restored = Module.from_checkpoint(str(checkpoint), {}).eval()
    with torch.no_grad():
        again = restored(torch.zeros(2, 6, 3, 5), x)
    torch.testing.assert_close(prediction, again)


def test_lead_adapter_selects_two_channels_from_frozen_72h_output():
    class StubRegression(torch.nn.Module):
        def forward(self, zero, condition):
            batch = condition.shape[0]
            values = torch.arange(
                batch * 144,
                dtype=condition.dtype,
                device=condition.device,
            ).reshape(batch, 144, 1, 1)
            return values.expand(-1, -1, *condition.shape[-2:])

    regression = StubRegression()
    adapter = WindRegressionLeadAdapter(
        regression,
        forecast_hours=72,
        wind_components=2,
        conditioning_input_channels=4,
        lead_encoding_channels=2,
    )
    x0 = torch.zeros(2, 2, 3, 5)
    condition = torch.zeros(2, 6, 3, 5)
    selected = adapter(x0, condition, lead_time_label=torch.tensor([1, 3]))
    assert selected.shape == (2, 2, 3, 5)
    assert float(selected[0, 0, 0, 0]) == 0.0
    # The second batch element starts at channel 144; lead 3 starts at 4.
    assert float(selected[1, 0, 0, 0]) == 148.0
    assert selected.grad_fn is None
    assert all(not parameter.requires_grad for parameter in adapter.regression_net.parameters())


def test_random_patching_preserves_22_channel_residual_contract():
    class StubRegression(torch.nn.Module):
        def forward(self, zero, condition):
            return zero[:, :6] + 0.1

    shape = (16, 16)
    adapter = WindRegressionLeadAdapter(
        StubRegression(),
        forecast_hours=3,
        wind_components=2,
        conditioning_input_channels=4,
        lead_encoding_channels=2,
    )
    diffusion = WindEDMPrecondSuperResolution(
        img_resolution=list(shape),
        img_in_channels=8,
        img_out_channels=2,
        model_channels=8,
        channel_mult=[1, 2],
        attn_resolutions=[],
        dropout=0.0,
        embedding_type="zero",
        use_apex_gn=False,
    )
    patching = WindRandomPatching2D(
        img_shape=shape,
        patch_shape=(8, 8),
        patch_num=1,
    )
    loss = ResidualLoss(
        regression_net=adapter,
        hr_mean_conditioning=True,
    )(
        net=diffusion,
        img_clean=torch.randn(1, 2, *shape),
        img_lr=torch.randn(1, 6, *shape),
        lead_time_label=torch.tensor([2]),
        patching=patching,
    )
    assert loss.shape == (1, 2, 8, 8)


def test_edm_diffusion_two_channel_forward_and_sampler():
    model = WindEDMPrecondSuperResolution(
        img_resolution=[16, 16],
        img_in_channels=22,
        img_out_channels=2,
        model_channels=8,
        channel_mult=[1, 2],
        attn_resolutions=[],
        dropout=0.0,
        embedding_type="zero",
        use_apex_gn=False,
    )
    noisy = torch.randn(1, 2, 16, 16)
    base_condition = torch.randn(1, 20, 16, 16)
    model_condition = torch.randn(1, 22, 16, 16)
    sigma = torch.ones(1)
    output = model(noisy, model_condition, sigma)
    assert output.shape == (1, 2, 16, 16)

    sampler_fn = partial(stochastic_sampler, num_steps=2)
    ensemble = diffusion_step(
        net=model,
        sampler_fn=sampler_fn,
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[11, 12]],
        img_lr=base_condition.expand(2, -1, -1, -1).contiguous(),
        rank=0,
        device=torch.device("cpu"),
        mean_hr=torch.zeros(1, 2, 16, 16),
    )
    assert ensemble.shape == (2, 2, 16, 16)
    assert torch.isfinite(ensemble).all()
    first = ensemble[0]
    repeated = diffusion_step(
        net=model,
        sampler_fn=sampler_fn,
        img_shape=(16, 16),
        img_out_channels=2,
        rank_batches=[[11, 12]],
        img_lr=base_condition.expand(2, -1, -1, -1).contiguous(),
        rank=0,
        device=torch.device("cpu"),
        mean_hr=torch.zeros(1, 2, 16, 16),
    )[0]
    torch.testing.assert_close(first, repeated)


def test_edm_diffusion_pads_and_crops_odd_full_domain():
    model = WindEDMPrecondSuperResolution(
        img_resolution=[17, 19],
        img_in_channels=22,
        img_out_channels=2,
        model_channels=8,
        channel_mult=[1, 2, 2],
        attn_resolutions=[],
        dropout=0.0,
        embedding_type="zero",
        use_apex_gn=False,
    )
    output = model(
        torch.randn(1, 2, 17, 19),
        torch.randn(1, 22, 17, 19),
        torch.ones(1),
    )
    assert output.shape == (1, 2, 17, 19)


def test_edm_diffusion_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(7)
    model = WindEDMPrecondSuperResolution(
        img_resolution=[16, 16],
        img_in_channels=22,
        img_out_channels=2,
        model_channels=8,
        channel_mult=[1, 2],
        attn_resolutions=[],
        dropout=0.0,
        embedding_type="zero",
        use_apex_gn=False,
    ).eval()
    noisy = torch.randn(1, 2, 16, 16)
    condition = torch.randn(1, 22, 16, 16)
    sigma = torch.ones(1)
    with torch.no_grad():
        expected = model(noisy, condition, sigma)
    checkpoint = tmp_path / "diffusion.mdlus"
    model.save(str(checkpoint))
    restored = Module.from_checkpoint(str(checkpoint), {}).eval()
    with torch.no_grad():
        actual = restored(noisy, condition, sigma)
    torch.testing.assert_close(expected, actual)


def test_wind_netcdf_writers_use_requested_dimensions(tmp_path):
    lat = np.linspace(2.0, 3.0, 3)
    lon = np.linspace(4.0, 5.0, 4)
    times = [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-02 00:00")]
    leads = [1, 2, 3]
    deterministic = tmp_path / "regression.nc"
    write_forecast_netcdf(
        deterministic,
        np.zeros((2, 3, 3, 4)),
        np.ones((2, 3, 3, 4)),
        times,
        leads,
        lat,
        lon,
        source="wind-regression",
    )
    with xr.open_dataset(deterministic, engine="netcdf4") as ds:
        assert ds.u10.dims == ("initial_time", "lead_hour", "lat", "lon")
        assert ds.attrs["source"] == "wind-regression"

    ensemble = tmp_path / "diffusion.nc"
    write_ensemble_netcdf(
        ensemble,
        np.zeros((2, 3, 4, 3, 4)),
        np.ones((2, 3, 4, 3, 4)),
        times,
        leads,
        lat,
        lon,
        source="wind-residual-diffusion",
    )
    with xr.open_dataset(ensemble, engine="netcdf4") as ds:
        assert ds.u10.dims == (
            "initial_time",
            "lead_hour",
            "ensemble",
            "lat",
            "lon",
        )
        assert ds.attrs["source"] == "wind-residual-diffusion"


def test_wind_netcdf_streaming_writer_accepts_lead_slices(tmp_path):
    lat = np.linspace(2.0, 3.0, 3)
    lon = np.linspace(4.0, 5.0, 4)
    times = [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-02 00:00")]
    leads = [1, 2]
    deterministic = tmp_path / "stream_regression.nc"
    with WindForecastNetCDFWriter(
        deterministic, times, leads, lat, lon, source="regression"
    ) as writer:
        writer.write_initial(0, np.zeros((2, 3, 4)), np.ones((2, 3, 4)))
        writer.write_lead(1, 0, np.full((3, 4), 2.0), np.full((3, 4), 3.0))
        writer.write_lead(1, 1, np.full((3, 4), 4.0), np.full((3, 4), 5.0))
    with xr.open_dataset(deterministic, engine="netcdf4") as ds:
        assert ds.u10.dims == ("initial_time", "lead_hour", "lat", "lon")
        np.testing.assert_allclose(
            ds.u10[1, 0].values, [[2.0] * 4, [2.0] * 4, [2.0] * 4]
        )

    ensemble = tmp_path / "stream_diffusion.nc"
    with WindForecastNetCDFWriter(
        ensemble, times, leads, lat, lon, source="diffusion", ensemble_size=3
    ) as writer:
        writer.write_initial(
            0, np.zeros((2, 3, 3, 4)), np.ones((2, 3, 3, 4))
        )
        writer.write_lead(1, 0, np.full((3, 3, 4), 2.0), np.full((3, 3, 4), 3.0))
        writer.write_lead(1, 1, np.full((3, 3, 4), 4.0), np.full((3, 3, 4), 5.0))
    with xr.open_dataset(ensemble, engine="netcdf4") as ds:
        assert ds.u10.dims == (
            "initial_time",
            "lead_hour",
            "ensemble",
            "lat",
            "lon",
        )
        np.testing.assert_allclose(ds.v10[1, 1].values, np.full((3, 3, 4), 5.0))
