import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo import Module
from physicsnemo.models.diffusion import UNet

def _pad_to_multiple(x: torch.Tensor | None, mult: int):
    if x is None:
        return None, None
    B, C, H, W = x.shape
    ph = (-H) % mult
    pw = (-W) % mult
    if ph == 0 and pw == 0:
        return x, (H, W)
    return F.pad(x, (0, pw, 0, ph)), (H, W)

def _crop_to(x: torch.Tensor, hw: tuple[int, int]):
    H, W = hw
    return x[..., :H, :W]

def _to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return type(o)(_to_py(v) for v in o)
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    return o

class UNetResidualCorrectorClamped3(Module):
    """
    y_hat = clamp( p0 + gamma * Δ , [clamp_min, clamp_max] )
    where p0 is the first input channel of img_lr; UNet predicts Δ (1 channel).
    """
    # (optional) class-level meta — nice, but instance will overwrite unless we reset below
    meta = types.SimpleNamespace(name="UNetResidualCorrectorClampedAdd")
    __model_checkpoint_version__ = 1

    def __init__(
        self,
        img_in_channels: int,
        img_out_channels: int = 1,
        unet_kwargs: dict | None = None,
        clamp_min: float = 0.0,
        clamp_max: float = 20.0,
        **_ignored,
    ):
        super().__init__()

        # >>>>>>>>>>>>>>>>>>>>>>>> CRITICAL LINE <<<<<<<<<<<<<<<<<<<<<<<<
        # Module.__init__() sets self.meta = None. Reset it now.
        if getattr(self, "meta", None) is None:
            self.meta = types.SimpleNamespace(name="UNetResidualCorrectorClampedAdd")
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        assert int(img_out_channels) == 1, "This wrapper predicts a single precip channel."
        self.img_out_channels = 1
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        uk = dict(unet_kwargs or {})
        uk.setdefault("img_in_channels", int(img_in_channels))
        uk.setdefault("img_out_channels", 1)
        ch_mult = uk.get("channel_mult", [1, 2, 2])
        self._down_factor = 2 ** len([int(v) for v in ch_mult])

        self.unet = UNet(**uk)
        self.gamma = nn.Parameter(torch.zeros(1))

        self._args = {
            "__name__":   self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "__args__": {
                "img_in_channels": int(img_in_channels),
                "img_out_channels": int(img_out_channels),
                "unet_kwargs": _to_py(uk),
                "clamp_min": float(self.clamp_min),
                "clamp_max": float(self.clamp_max),
            },
        }

    def forward(self, x0, img_lr, augment_pipe=None, **unused):
        # anchor on input precip
        p0 = img_lr[:, :1]
                
        # call base UNet
        delta = self.unet(x0, img_lr)               # (B,1,H,W) or (B,2,H,W) 
        
        y_hat = p0 + self.gamma * delta
        
        return torch.clamp(y_hat, self.clamp_min, self.clamp_max)