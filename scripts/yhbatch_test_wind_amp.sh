#!/usr/bin/env bash
# Submit with:
# yhbatch -p a800x_595 -G 1 -c 8 -t 20 scripts/yhbatch_test_wind_amp.sh

set -euo pipefail

repo_root="/XYFS02/HDD_POOL/cmafj_klli/cmafj_klli_1/dx/swaither-precip"
cd "${repo_root}"

module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
source scripts/wind_env.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[$(date '+%F %T')] Wind AMP test started"
echo "Repository: ${repo_root}"
echo "SLURM job: ${SLURM_JOB_ID:-interactive}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python -u - <<'PY'
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "scripts"))

from train_wind_step1 import _model


class ShapeOnlyDataset:
    grid_channels = 4

    @staticmethod
    def image_shape():
        return (32, 40)

    @staticmethod
    def input_channels():
        return ("u10", "v10")


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this Slurm job")

cfg = OmegaConf.create(
    {
        "unet": {
            "model_channels": 8,
            "channel_mult": [1, 2, 2],
            "attn_resolutions": [],
            "embedding_type": "zero",
            "use_apex_gn": False,
        }
    }
)

device = torch.device("cuda")
model = _model(cfg, ShapeOnlyDataset(), use_fp16=False, amp=True).to(device)
if model.unet.amp_mode is not True:
    raise AssertionError("PhysicsNeMo U-Net amp_mode was not enabled")

target = torch.randn(2, 2, 32, 40, device=device)
condition = torch.randn(2, 6, 32, 40, device=device)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
    prediction = model(torch.zeros_like(target), condition)
    loss = torch.mean((prediction.float() - target.float()) ** 2)
loss.backward()

if prediction.shape != target.shape:
    raise AssertionError(f"Unexpected output shape: {tuple(prediction.shape)}")
if not any(parameter.grad is not None for parameter in model.parameters()):
    raise AssertionError("No gradients were produced")

print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"AMP dtype: {torch.bfloat16}")
print(f"amp_mode: {model.unet.amp_mode}")
print(f"prediction shape: {tuple(prediction.shape)}")
print("Wind AMP forward/backward test passed")
PY

smoke_root="/tmp/wind_amp_smoke_${SLURM_JOB_ID:-$$}"
echo "[$(date '+%F %T')] Running real Step 1 memory-probe smoke test"
python -u scripts/train_wind_step1.py \
  --config-name=config_training_wind_step1_unet.yaml \
  training.epochs=0 \
  training.max_samples=1 \
  training.num_workers=0 \
  paths.outputs="${smoke_root}"

echo "[$(date '+%F %T')] Wind AMP test completed"
