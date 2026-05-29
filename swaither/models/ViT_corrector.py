import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTBiasCorrectorFiLM(nn.Module):
    """
    Simple ViT-based bias corrector with FiLM lead-time conditioning.

    Expects (for compatibility with your existing pipeline):
      - x0: dummy tensor (ignored), kept for interface compatibility
      - img_lr: (B, C_in, H, W) normalized inputs
      - lead_time_label OR lead_time: (B,) or (B, 1) ints in [0, n_leads-1]
      - Extra kwargs like force_fp32, time_id_label, patching, augment_pipe
        are accepted and ignored.

    Returns:
      - pred_norm: (B, C_out, H, W) in the same normalized space as img_clean
    """

    def __init__(
        self,
        img_in_channels: int,
        img_out_channels: int,
        img_resolution,
        patch_size: int = 4,
        d_model: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        mlp_hidden: int = 256,
        n_leads: int = 24,
        emb_dim: int = 32,
    ):
        super().__init__()

        self.img_out_channels = img_out_channels
        self.in_channels = img_in_channels
        self.H, self.W = int(img_resolution[0]), int(img_resolution[1])
        self.patch_size = int(patch_size)
        self.n_leads = int(n_leads)

        assert self.H % self.patch_size == 0, "H must be divisible by patch_size"
        assert self.W % self.patch_size == 0, "W must be divisible by patch_size"

        # ---------------- FiLM lead-time conditioning ----------------
        self.lead_emb = nn.Embedding(self.n_leads, emb_dim)
        self.film_mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 2 * img_in_channels),  # gamma & beta
        )

        # ---------------- Patch embedding ----------------
        self.proj = nn.Conv2d(
            img_in_channels,
            d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.Hp = self.H // self.patch_size
        self.Wp = self.W // self.patch_size
        self.num_patches = self.Hp * self.Wp

        # Learnable 2D positional embeddings for the tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        # ---------------- Transformer encoder ----------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=mlp_hidden,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # ---------------- Patch -> image decoder ----------------
        self.decoder = nn.ConvTranspose2d(
            d_model,
            img_out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)

    def _apply_film(self, x, lead_time_label):
        """
        x: (B, C_in, H, W)
        lead_time_label: (B,) or (B, 1) of ints in [0, n_leads-1]
        """
        if lead_time_label.dim() > 1:
            lead_idx = lead_time_label.squeeze(-1)
        else:
            lead_idx = lead_time_label

        lead_idx = lead_idx.long().clamp(0, self.n_leads - 1)

        # (B, emb_dim)
        emb = self.lead_emb(lead_idx)
        # (B, 2*C_in)
        film = self.film_mlp(emb)
        gamma, beta = film.chunk(2, dim=-1)  # (B, C_in) each

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C_in, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C_in, 1, 1)

        # classic FiLM: y = x * (1 + gamma) + beta
        return x * (1.0 + gamma) + beta

    def forward(self, x0, img_lr, lead_time_label=None, lead_time=None, **kwargs):
        """
        x0 is ignored (kept for compatibility).

        img_lr: (B, C_in, H, W) normalized
        lead_time_label or lead_time: (B,) or (B, 1)

        Extra kwargs (force_fp32, time_id_label, patching, augment_pipe, ...)
        are accepted and ignored.
        """
        # pick whichever name was used
        if lead_time_label is None and lead_time is not None:
            lead_time_label = lead_time
        if lead_time_label is None:
            raise ValueError("ViTBiasCorrectorFiLM: lead_time_label (or lead_time) must be provided")

        B, C, H, W = img_lr.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert H == self.H and W == self.W, f"Expected {(self.H, self.W)}, got {(H, W)}"

        # 1) FiLM condition the physical channels
        x = self._apply_film(img_lr, lead_time_label)  # (B, C_in, H, W)

        # 2) Patch embedding: (B, C_in, H, W) -> (B, d_model, Hp, Wp)
        x = self.proj(x)
        B, D, Hp, Wp = x.shape

        # 3) Flatten patches to tokens: (B, Hp*Wp, D)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D), N = Hp*Wp

        # 4) Add positional embeddings
        x = x + self.pos_emb[:, : x.size(1), :]

        # 5) Transformer encoder
        x = self.transformer(x)  # (B, N, D)

        # 6) Back to feature map
        x = x.transpose(1, 2).view(B, D, Hp, Wp)  # (B, D, Hp, Wp)

        # 7) Decode to full-res precip map
        out = self.decoder(x)  # (B, C_out, H, W)

        return out
