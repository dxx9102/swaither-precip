#!/usr/bin/env bash
# Submit with:
#   yhbatch -p a800x_595 -G 1 -o ./outputs/slurm/wind_joint_test.out scripts/yhbatch_test_wind_joint.sh
set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

pytest -q \
  tests/test_wind_data.py \
  tests/test_wind_models_and_io.py \
  tests/test_wind_configs.py \
  tests/test_wind_stats.py \
  tests/test_wind_evaluate.py

python -u - <<'PY'
import tempfile
from pathlib import Path

import torch
from physicsnemo.metrics.diffusion import ResidualLoss
from physicsnemo import Module
from swaither.models import (
    WindEDMPrecondSuperResolution,
    WindForecastUNet,
    WindRandomPatching2D,
    WindRegressionLeadAdapter,
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this Slurm job")

device = torch.device("cuda")
shape = (281, 321)
regression = WindForecastUNet(
    img_in_channels=18,
    forecast_hours=72,
    wind_components=2,
    unet_kwargs={
        "img_resolution": list(shape),
        "model_channels": 64,
        "channel_mult": [1, 2, 2],
        "attn_resolutions": [],
        "dropout": 0.05,
        "embedding_type": "zero",
        "use_apex_gn": False,
        "amp_mode": True,
        "model_type": "SongUNet",
    },
).to(device)
X = torch.randn(1, 18, *shape, device=device)
Z = torch.zeros(1, 144, *shape, device=device)
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
    mean = regression(Z, X)
    regression_loss = mean.float().square().mean()
regression_loss.backward()
regression.zero_grad(set_to_none=True)
print(
    "regression shape", tuple(mean.shape),
    "peak_GiB", round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
)

with tempfile.TemporaryDirectory() as temporary:
    checkpoint = Path(temporary) / "regression.mdlus"
    regression.eval().requires_grad_(False)
    regression.save(str(checkpoint))
    del mean, regression_loss, X, Z, regression
    torch.cuda.empty_cache()

    regression = Module.from_checkpoint(str(checkpoint), {})
    regression = regression.eval().requires_grad_(False).to(device)

    adapter = WindRegressionLeadAdapter(
        regression,
        forecast_hours=72,
        wind_components=2,
        conditioning_input_channels=18,
        lead_encoding_channels=2,
    ).to(device)
    diffusion = WindEDMPrecondSuperResolution(
        img_resolution=list(shape),
        img_in_channels=22,
        img_out_channels=2,
        model_channels=64,
        channel_mult=[1, 2, 2],
        attn_resolutions=[],
        dropout=0.05,
        embedding_type="positional",
        use_apex_gn=False,
        amp_mode=True,
        checkpoint_level=1,
    ).to(device)
    loss_fn = ResidualLoss(
        regression_net=adapter,
        hr_mean_conditioning=True,
    )
    patching = WindRandomPatching2D(
        img_shape=shape,
        patch_shape=(272, 320),
        patch_num=1,
    )
    y = torch.randn(1, 2, *shape, device=device)
    x = torch.randn(1, 20, *shape, device=device)
    lead = torch.tensor([6], device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        diffusion_loss = loss_fn(
            net=diffusion,
            img_clean=y,
            img_lr=x,
            lead_time_label=lead,
            patching=patching,
        ).sum()
    diffusion_loss.backward()
    assert all(parameter.grad is None for parameter in regression.parameters())
    assert any(parameter.grad is not None for parameter in diffusion.parameters())
    print(
        "diffusion loss", round(float(diffusion_loss.detach().cpu()), 6),
        "peak_GiB", round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
    )

print("GPU: ", torch.cuda.get_device_name(device))
print("Wind joint GPU forward/backward test passed")
PY
