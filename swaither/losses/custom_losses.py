"""
Custom Losses used within the project.

Contains:
  - RegressionLossTemporalSmooth
  - RegressionLossTimeConsistency
  - RegressionHuberLossTimeConsistency

Check if the losses for details regarding their characteristics
"""

import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from physicsnemo.metrics.diffusion import RegressionLoss

class RegressionLossTemporalSmooth(RegressionLoss):
    """
    Standard RegressionLoss + temporal smoothness in lead space.

    L = L_data(img_clean, f(img_lr, lead)) 
        + lambda_smooth * E[(f(img_lr, lead) - f(img_lr, lead_pert))^2]

    where lead_pert is a random neighbour in lead index space.
    """

    def __init__(
        self,
        lambda_smooth: float = 0.1,
        max_delta_lead: int = 1,
        n_leads: int = 24,
        **base_kwargs,
    ):
        """
        Args
        ----
        lambda_smooth : weight of the smoothness term.
        max_delta_lead : use random integer deltas in [-max_delta_lead, +max_delta_lead] \ {0}.
        n_leads : total number of lead indices (0..n_leads-1).
        base_kwargs : passed through to `RegressionLoss` (e.g. prob_channels).
        """
        super().__init__(**base_kwargs)
        self.lambda_smooth = float(lambda_smooth)
        self.max_delta_lead = int(max_delta_lead)
        self.n_leads = int(n_leads)

    def __call__(self, *args, **kwargs):
        """
        Expected kwargs (as in RegressionLoss):
          - net: the model
          - img_clean: (B, 1, H, W)
          - img_lr: (B, C_in, H, W)
          - lead_time_label: (B,) or (B,1)  [only for lead-aware models]
          - augment_pipe, patching, use_patch_grad_acc, ...
        """
        # 1) Standard data loss from parent
        base_loss = super().__call__(*args, **kwargs)

        # If no smoothing or no lead info, just return base loss
        if self.lambda_smooth <= 0.0:
            return base_loss

        lead = kwargs.get("lead_time_label", None)
        if lead is None:
            return base_loss

        net = kwargs["net"]
        img_lr = kwargs["img_lr"]
        device = img_lr.device
        dtype = img_lr.dtype

        # lead may be (B,1) or (B,)
        lead_orig = lead
        if lead.dim() == 2 and lead.size(1) == 1:
            lead_flat = lead[:, 0]  # (B,)
        else:
            lead_flat = lead

        B, _, H, W = img_lr.shape

        # Handle DDP wrapper: get underlying module if needed
        net_for_attr = getattr(net, "module", net)
        c_out = getattr(net_for_attr, "img_out_channels", 1)

        # zero latent, same as CorrDiff regression wrapper
        x0 = torch.zeros((B, c_out, H, W), device=device, dtype=dtype)

        # 2) Sample neighbour leads in [-max_delta_lead, +max_delta_lead]\{0}
        if self.max_delta_lead > 0:
            delta = torch.randint(
                low=-self.max_delta_lead,
                high=self.max_delta_lead + 1,
                size=lead_flat.shape,
                device=device,
            )
            # avoid 0 deltas
            delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        else:
            # degenerate: no neighbour -> no smoothness
            return base_loss

        lead_pert_flat = (lead_flat + delta).clamp(0, self.n_leads - 1)

        # Restore same shape convention as incoming lead_time_label
        if lead_orig.dim() == 2 and lead_orig.size(1) == 1:
            lead_main = lead_flat.unsqueeze(1)      # (B,1)
            lead_pert = lead_pert_flat.unsqueeze(1) # (B,1)
        else:
            lead_main = lead_flat                   # (B,)
            lead_pert = lead_pert_flat              # (B,)

        # 3) First prediction: f(x, lead_main)
        net_kwargs_main = {}
        # Only pass lead_time_label if caller had it
        if "lead_time_label" in kwargs:
            net_kwargs_main["lead_time_label"] = lead_main

        y_main = net(x0, img_lr, **net_kwargs_main)

        # 4) Second prediction: f(x, lead_pert)
        net_kwargs_pert = {}
        if "lead_time_label" in kwargs:
            net_kwargs_pert["lead_time_label"] = lead_pert

        y_pert = net(x0, img_lr, **net_kwargs_pert)

        smooth_loss = (y_pert - y_main).pow(2).mean()

        return base_loss + self.lambda_smooth * smooth_loss


class RegressionLossTimeConsistency(RegressionLoss):
    """
    Same as PhysicsNeMo's RegressionLoss + time-consistency penalty over samples
    that share the same valid time (time_id_label).

    Total loss (per-pixel tensor):
        L_total = L_base + lambda_time * L_time

    Exposes (for logging):
        self.last_base_loss       : scalar
        self.last_time_penalty    : scalar
        self.last_mse_per_lead    : dict {lead_index -> mse_scalar}
    """

    def __init__(self, lambda_time: float = 1.0):
        super().__init__()
        self.lambda_time = float(lambda_time)
        self.last_base_loss = None
        self.last_time_penalty = None
        self.last_mse_per_lead = None

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe=None,
        lead_time_label: torch.Tensor | None = None,
        time_id_label: torch.Tensor | None = None,
    ) -> torch.Tensor:

        weight = 1.0  # same as original RegressionLoss

        # --- Normalize label shapes (B,1) -> (B,) if needed ---
        if lead_time_label is not None and lead_time_label.dim() == 2 and lead_time_label.size(1) == 1:
            lead_time_label = lead_time_label.squeeze(1)
        if time_id_label is not None and time_id_label.dim() == 2 and time_id_label.size(1) == 1:
            time_id_label = time_id_label.squeeze(1)

        # === ORIGINAL RegressionLoss body =====================================
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )

        y = y_tot[:, : img_clean.shape[1], :, :]        # (B, C_hr, H, W)
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]     # (B, C_lr, H, W)

        zero_input = torch.zeros_like(y, device=img_clean.device)

        if lead_time_label is not None:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                augment_labels=augment_labels,
            )

        base_loss_tensor = weight * ((D_yn - y) ** 2)   # (B, C_hr, H, W)
        # ======================================================================

        # --- Time-consistency penalty: group by valid time -------------------
        time_penalty = img_clean.new_tensor(0.0)

        if self.lambda_time > 0.0 and time_id_label is not None:
            unique_times = torch.unique(time_id_label)
            acc_penalty = img_clean.new_tensor(0.0)
            group_count = 0

            for t in unique_times:
                idx = (time_id_label == t).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() <= 1:
                    continue

                group_pred = D_yn[idx]                      # (N_t, C_hr, H, W)
                group_mean = group_pred.mean(dim=0, keepdim=True)
                group_penalty = F.mse_loss(
                    group_pred, group_mean.expand_as(group_pred)
                )
                acc_penalty = acc_penalty + group_penalty
                group_count += 1

            if group_count > 0:
                time_penalty = acc_penalty / group_count    # scalar

        # --- Combine total tensor loss ---------------------------------------
        total_loss_tensor = base_loss_tensor + self.lambda_time * time_penalty

        # --- Scalars for logging ---------------------------------------------
        self.last_base_loss = base_loss_tensor.mean().detach()
        self.last_time_penalty = time_penalty.detach()

        # --- Per-lead MSE for logging (current batch only) -------------------
        self.last_mse_per_lead = None
        if lead_time_label is not None:
            with torch.no_grad():
                # per-sample MSE across channels & pixels
                sq_err = (D_yn - y) ** 2                  # (B, C, H, W)
                per_sample_mse = sq_err.view(sq_err.size(0), -1).mean(dim=1)  # (B,)

                leads_cpu = lead_time_label.detach().to("cpu")
                mse_cpu = per_sample_mse.detach().to("cpu")

                mse_per_lead = {}
                for ell in torch.unique(leads_cpu):
                    mask = (leads_cpu == ell)
                    if mask.any():
                        mse = mse_cpu[mask].mean().item()
                        mse_per_lead[int(ell.item())] = mse

                self.last_mse_per_lead = mse_per_lead

        return total_loss_tensor


class RegressionHuberLossTimeConsistency(RegressionLoss):
    """
    Huber regression loss + time-consistency penalty over samples
    that share the same valid time (time_id_label).

    Total loss (per-pixel tensor):
        L_total = L_huber + lambda_time * L_time

    Exposes (for logging):
        self.last_base_loss       : scalar (huber part)
        self.last_time_penalty    : scalar
        self.last_mse_per_lead    : dict {lead_index -> mse_scalar}
    """

    def __init__(self, lambda_time: float = 1.0, huber_delta: float = 1.0):
        super().__init__()
        self.lambda_time = float(lambda_time)
        self.huber_delta = float(huber_delta)
        self.last_base_loss = None
        self.last_time_penalty = None
        self.last_mse_per_lead = None

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe=None,
        lead_time_label: torch.Tensor | None = None,
        time_id_label: torch.Tensor | None = None,
    ) -> torch.Tensor:

        weight = 1.0  # same as original RegressionLoss

        # --- Normalize label shapes (B,1) -> (B,) if needed ---
        if lead_time_label is not None and lead_time_label.dim() == 2 and lead_time_label.size(1) == 1:
            lead_time_label = lead_time_label.squeeze(1)
        if time_id_label is not None and time_id_label.dim() == 2 and time_id_label.size(1) == 1:
            time_id_label = time_id_label.squeeze(1)

        # === ORIGINAL RegressionLoss forward (up to prediction) ==============
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )

        y   = y_tot[:, : img_clean.shape[1], :, :]      # (B, C_hr, H, W)
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]     # (B, C_lr, H, W)

        zero_input = torch.zeros_like(y, device=img_clean.device)

        if lead_time_label is not None:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                augment_labels=augment_labels,
            )

        # --- Huber base loss (per-pixel tensor) ------------------------------
        # We want a tensor of shape (B, C, H, W), so use reduction="none".
        try:
            base_loss_tensor = weight * F.smooth_l1_loss(
                D_yn, y, reduction="none", beta=self.huber_delta
            )
        except TypeError:
            # For older PyTorch that doesn’t support `beta`
            base_loss_tensor = weight * F.smooth_l1_loss(
                D_yn, y, reduction="none"
            )
        # ======================================================================

        # --- Time-consistency penalty: group by valid time -------------------
        time_penalty = img_clean.new_tensor(0.0)

        if self.lambda_time > 0.0 and time_id_label is not None:
            unique_times = torch.unique(time_id_label)
            acc_penalty = img_clean.new_tensor(0.0)
            group_count = 0

            for t in unique_times:
                idx = (time_id_label == t).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() <= 1:
                    continue

                group_pred = D_yn[idx]                      # (N_t, C_hr, H, W)
                group_mean = group_pred.mean(dim=0, keepdim=True)

                # Huber on deviations from group mean
                try:
                    group_penalty = F.smooth_l1_loss(
                        group_pred,
                        group_mean.expand_as(group_pred),
                        reduction="mean",
                        beta=self.huber_delta,
                    )
                except TypeError:
                    group_penalty = F.smooth_l1_loss(
                        group_pred,
                        group_mean.expand_as(group_pred),
                        reduction="mean",
                    )

                acc_penalty = acc_penalty + group_penalty
                group_count += 1

            if group_count > 0:
                time_penalty = acc_penalty / group_count    # scalar

        # --- Combine total tensor loss ---------------------------------------
        # time_penalty is scalar, we broadcast-add it to every pixel
        total_loss_tensor = base_loss_tensor + self.lambda_time * time_penalty

        # --- Scalars for logging ---------------------------------------------
        self.last_base_loss = base_loss_tensor.mean().detach()
        self.last_time_penalty = time_penalty.detach()

        # --- Per-lead MSE for logging (still MSE, for comparability) ---------
        self.last_mse_per_lead = None
        if lead_time_label is not None:
            with torch.no_grad():
                sq_err = (D_yn - y) ** 2                        # (B, C, H, W)
                per_sample_mse = sq_err.view(sq_err.size(0), -1).mean(dim=1)  # (B,)

                leads_cpu = lead_time_label.detach().to("cpu")
                mse_cpu   = per_sample_mse.detach().to("cpu")

                mse_per_lead = {}
                for ell in torch.unique(leads_cpu):
                    mask = (leads_cpu == ell)
                    if mask.any():
                        mse = mse_cpu[mask].mean().item()
                        mse_per_lead[int(ell.item())] = mse

                self.last_mse_per_lead = mse_per_lead

        return total_loss_tensor


