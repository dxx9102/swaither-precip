"""
Script with Visualizes precipitation super-resolution metrics across lead times for multiple models.

Usage:
    1. Update MODELS list with your model names, file paths, and lead_position (for single-lead models).
    2. Adjust METRIC_GROUPS to select which metrics appear in which subplot.
    3. Run the script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import xarray as xr

# ─────────────────────────────────────────────────────────────
# 0.  CONFIGURATION  — edit these to match your setup
# ─────────────────────────────────────────────────────────────

# Full set of lead positions and their labels (the "master" x-axis)
ALL_LEAD_POSITIONS = [0, 1, 2, 3, 7, 11, 15, 19, 23]  # lead coord values for 9-lead models
HORIZON_LABELS     = ["6h", "12h", "18h", "24h", "2d", "3d", "4d", "5d", "6d"]
LEAD_TO_LABEL      = dict(zip(ALL_LEAD_POSITIONS, HORIZON_LABELS))

# Model definitions.
#   - "path":          path to the .nc metrics file
#   - "lead_position": (optional) for single-lead models, the lead coordinate value
#                      at which to plot the single point.  If omitted → multi-lead model.
MODELS = [
    {
        "name": "UNet, No Curri, λ=2",
        "path": "../output/generations/UNet_lam2_ValSampleStride4/metrics/metrics_mean_extra_clean.nc",
    },
    {
        "name": "UNet Multi FineTuned",
        "path": "../output/generations/UNet_MultiStepFineTuning2_ValSampleStride4/metrics/metrics_mean_extra_clean.nc",
    },
    {
        "name": "B0: AIFS",
        "path": "../output/generations/AIFSOrigPreds_ValSampleStride4/metrics/metrics_mean_extra_clean.nc",
    },
    {
        "name": "B4: Pixel LR, MultiVar",
        "path": "../output/generations/BaselineB4_ValSampleStride4/metrics/metrics_mean_extra_clean.nc",
    },
    {
        "name": "UNet Single 6h",
        "path": "../output/generations/SingleLead_6h_ValSampleStride4/metrics/metrics_mean_extra_clean.nc",
        "lead_position": 0,   # this model's single lead maps to position 0 → "6h"
    },
    {
        "name": "UNet Single 3day",
        "path": "../output/generations/SingleLead_3days_ValSampleStride4/metrics/metrics_mean_extra_clean.nc",
        "lead_position": 11,  # maps to position 11 → "3d"
    },
    {
        "name": "UNet Single 6day",
        "path": "../output/generations/SingleLead_6days_ValSampleStride4/metrics/metrics_mean_extra_clean.nc",
        "lead_position": 23,  # maps to position 23 → "6d"
    },
]

# Group metrics into subplots.
METRIC_GROUPS = [
    ("CRPS",                    ["crps"]),
    ("MSE on wet pixels",       ["mse_wet"]),
    ("Detection (CSI)",         ["csi"]),
    # ("Heavy-event errors",    ["mse_heavy"]),
    # ("FAR / Freq. Bias",      ["far", "freq_bias"]),
    # ("Brier / Reliability",   ["brier_score", "reliability_index"]),
    # ("SAL",                   ["eS", "eA", "eL"]),
    # ("Spectral",              ["spectral_slope_error", "spectral_correlation"]),
    # ("FSS (1 / 5 / 17 px)",  ["avfss_1", "avfss_5", "avfss_17"]),
]

# Visual settings
SHOW_SPREAD = False
FIGSIZE_PER_SUBPLOT = (5.5, 3.5)

# ─────────────────────────────────────────────────────────────
# 1.  COLOUR PALETTE & STYLE
# ─────────────────────────────────────────────────────────────

MODEL_COLORS = [
    "#2D6A4F",  # deep green
    "#E76F51",  # burnt orange
    "#264653",  # dark teal
    "#E9C46A",  # golden yellow
    "#7209B7",  # vivid purple
    "#F4A261",  # sandy peach
    "#118AB2",  # cerulean blue
]

METRIC_LINESTYLES = ["-", "--", ":", "-."]

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "legend.fontsize":   8,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#FAFAFA",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_models(models: list) -> list:
    """
    Load each model's dataset and attach metadata.
    Returns a list of dicts with keys: name, ds, is_single, lead_position.
    """
    loaded = []
    for m in models:
        ds = xr.open_dataset(m["path"])
        is_single = "lead_position" in m
        loaded.append({
            "name":          m["name"],
            "ds":            ds,
            "is_single":     is_single,
            "lead_position": m.get("lead_position", None),
        })
    return loaded


# ─────────────────────────────────────────────────────────────
# 3.  PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_metrics(models_loaded: list, metric_groups: list, show_spread: bool = True):
    """
    Create a multi-panel figure.
    Multi-lead models → lines with optional ±1σ shading.
    Single-lead models → single diamond marker at their lead_position.
    """
    n_groups = len(metric_groups)
    ncols = min(3, n_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig_w = FIGSIZE_PER_SUBPLOT[0] * ncols
    fig_h = FIGSIZE_PER_SUBPLOT[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (group_title, metric_list) in enumerate(metric_groups):
        ax = axes_flat[idx]
        ax.set_title(group_title, pad=8)

        for mi, metric in enumerate(metric_list):
            ls = METRIC_LINESTYLES[mi % len(METRIC_LINESTYLES)]

            for ci, minfo in enumerate(models_loaded):
                ds   = minfo["ds"]
                name = minfo["name"]
                if metric not in ds:
                    continue

                color = MODEL_COLORS[ci % len(MODEL_COLORS)]
                label = name if mi == 0 else None

                vals = ds[metric]
                mean = vals.mean(dim="time").values
                std  = vals.std(dim="time").values

                if minfo["is_single"]:
                    # ── Single-lead model: one diamond at the correct x position ──
                    x_pos = minfo["lead_position"]
                    ax.plot(
                        x_pos, mean[0],
                        color=color, marker="D", markersize=8,
                        markeredgecolor="white", markeredgewidth=0.8,
                        linestyle="none", label=label, zorder=5,
                    )
                    if show_spread:
                        ax.errorbar(
                            x_pos, mean[0], yerr=std[0],
                            color=color, fmt="none", capsize=4,
                            capthick=1.2, elinewidth=1.2, zorder=4,
                        )
                else:
                    # ── Multi-lead model: line plot ──
                    leads = ds.lead.values
                    ax.plot(
                        leads, mean,
                        color=color, ls=ls, lw=1.8,
                        marker="o", markersize=3.5,
                        label=label, zorder=3,
                    )
                    if show_spread:
                        ax.fill_between(
                            leads, mean - std, mean + std,
                            color=color, alpha=0.08, zorder=1,
                        )

        # ── Shared x-axis: always show the full lead range with labels ──
        ax.set_xticks(ALL_LEAD_POSITIONS)
        ax.set_xticklabels(HORIZON_LABELS, fontsize=8)
        ax.set_xlim(ALL_LEAD_POSITIONS[0] - 0.5, ALL_LEAD_POSITIONS[-1] + 0.5)
        ax.set_xlabel("Lead time")
        ax.set_ylabel("Value")

        # Metric-linestyle sub-legend (only if multiple metrics in one panel)
        if len(metric_list) > 1:
            metric_handles = [
                Line2D([0], [0], color="grey",
                       ls=METRIC_LINESTYLES[j % len(METRIC_LINESTYLES)],
                       lw=1.4, label=metric_list[j])
                for j in range(len(metric_list))
            ]
            metric_leg = ax.legend(
                handles=metric_handles, loc="upper left",
                fontsize=7, framealpha=0.7, title="Metric", title_fontsize=7,
                handlelength=2.2,
            )
            ax.add_artist(metric_leg)

    # Hide unused subplot panels
    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # ── Global model legend ──
    # Diamond for single-lead, circle+line for multi-lead
    model_handles = []
    for i, minfo in enumerate(models_loaded):
        marker = "D" if minfo["is_single"] else "o"
        ms     = 6   if minfo["is_single"] else 4
        ls     = "none" if minfo["is_single"] else "-"
        mec    = "white" if minfo["is_single"] else MODEL_COLORS[i % len(MODEL_COLORS)]
        model_handles.append(
            Line2D([0], [0],
                   color=MODEL_COLORS[i % len(MODEL_COLORS)],
                   lw=2.2, ls=ls, marker=marker, markersize=ms,
                   markeredgecolor=mec, markeredgewidth=0.6,
                   label=minfo["name"])
        )

    fig.legend(
        handles=model_handles,
        loc="lower center",
        ncol=min(len(models_loaded), 4),
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        borderpad=0.6,
        columnspacing=1.5,
    )

    fig.suptitle(
        "Precipitation Super-Resolution — Metrics vs Lead Time",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1.0])
    return fig


# ─────────────────────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    models_loaded = load_models(MODELS)
    fig = plot_metrics(models_loaded, METRIC_GROUPS, show_spread=SHOW_SPREAD)

    fig.savefig("metrics_per_lead.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig("metrics_per_lead.pdf", bbox_inches="tight", facecolor="white")
    print("Saved: metrics_per_lead.png / .pdf")
    plt.show()