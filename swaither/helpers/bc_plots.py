"""
Metric-comparison plots for bias-correction evaluation.

Contains:
  - plot_metric_comparison
  - plot_metric_comparison_grouped_legend
  - plot_fss_by_lead
  - plot_sal_by_lead
  - plot_fss_sal_combined
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_comparison(
    baselines: dict,
    multi_lead_models: dict,
    single_lead_models: dict,
    metric: str = "MSE_all",
    split: str = "val",
    n_leads: int = 24,
    lead_time_hours: int = 6,  # Hours per lead time step
    figsize: tuple = (14, 8),
    ylim: tuple = (0, 0.06),
    save_path: str | None = None,
    title: str | None = None,
):
    """
    Plot metric comparison across lead times for baselines, multi-lead models, and single-lead models.

    Args:
        baselines: dict of {name: per_lead_dict} for baseline methods
        multi_lead_models: dict of {name: results_dict} for multi-lead models
        single_lead_models: dict of {name: (lead_idx, results_dict)} for single-lead models
        metric: Metric to plot (e.g., "MSE_all", "MAE_all", "CSI", etc.)
        split: "train" or "val"
        n_leads: Number of lead times
        lead_time_hours: Hours per lead time step (default 6h)
        figsize: Figure size
        ylim: Y-axis limits
        save_path: If provided, save figure to this path
        title: Optional title for the plot
    """

    # Create figure with light grey background
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("#F5F5F5")  # Light grey background

    lead_range = np.arange(n_leads)

    # Convert lead indices to hours
    lead_hours = (lead_range + 1) * lead_time_hours  # 6h, 12h, 18h, ...

    # =========================================================================
    # Colorblind-friendly palettes (based on Wong, 2011 & IBM Design)
    # =========================================================================

    # Baseline colors: shades of grey
    n_baselines = len(baselines)
    baseline_colors = plt.cm.Greys(np.linspace(0.35, 0.65, n_baselines))

    # Multi-lead model colors: colorblind-friendly palette (IBM Design Library)
    colorblind_palette = [
        "#648FFF",  # Blue
        "#785EF0",  # Purple
        "#DC267F",  # Magenta
        "#FE6100",  # Orange
        "#FFB000",  # Gold/Yellow
        "#22A884",  # Teal
        "#414487",  # Dark blue
        "#2A788E",  # Steel blue
    ]

    # Single-lead model colors: distinct, saturated colors
    single_lead_palette = [
        "#E31A1C",  # Red
        "#1F78B4",  # Blue
        "#33A02C",  # Green
        "#FF7F00",  # Orange
        "#6A3D9A",  # Purple
    ]

    # =========================================================================
    # Plot baselines (dashed lines, diamond markers)
    # =========================================================================
    for idx, (name, per_lead_data) in enumerate(baselines.items()):
        values = [per_lead_data[i][metric] for i in range(n_leads)]
        ax.plot(
            lead_hours,
            values,
            marker="d",
            markersize=5,
            alpha=0.6,
            linestyle="--",
            linewidth=1.5,
            color=baseline_colors[idx],
            label=name,
        )

    # =========================================================================
    # Plot multi-lead models (solid lines, circle markers)
    # =========================================================================
    for idx, (name, results) in enumerate(multi_lead_models.items()):
        per_lead = results[split]["per_lead"]
        values = [per_lead[i][metric] for i in range(n_leads)]
        color = colorblind_palette[idx % len(colorblind_palette)]
        ax.plot(
            lead_hours,
            values,
            marker="o",
            markersize=7,
            alpha=0.9,
            linestyle="-",
            linewidth=2.2,
            color=color,
            label=name,
        )

    # =========================================================================
    # Plot single-lead models (X markers, no lines)
    # =========================================================================
    for idx, (name, (lead_idx, results)) in enumerate(single_lead_models.items()):
        value = results[split]["global"][metric]
        lead_hour = (lead_idx + 1) * lead_time_hours
        color = single_lead_palette[idx % len(single_lead_palette)]
        ax.plot(
            lead_hour,
            value,
            marker="X",
            markersize=16,
            markeredgecolor="white",
            markeredgewidth=2,
            color=color,
            linestyle="none",
            label=name,
            zorder=10,
        )

    # =========================================================================
    # Axis formatting
    # =========================================================================

    # X-axis: lead time in hours with nice labels
    ax.set_xlabel("Lead Time", fontsize=13, fontweight="medium", labelpad=10)
    ax.set_xlim(lead_hours[0] - 3, lead_hours[-1] + 3)

    # Custom x-tick labels showing both hours and days where relevant
    tick_positions = []
    tick_labels = []
    for i, h in enumerate(lead_hours):
        if h % 24 == 0:  # Every day
            tick_positions.append(h)
            days = h // 24
            tick_labels.append(f"{h}h\n({days}d)")
        elif h == 6:  # First point
            tick_positions.append(h)
            tick_labels.append("6h")
        elif h % 12 == 0:  # Every 12 hours
            tick_positions.append(h)
            tick_labels.append(f"{h}h")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)

    # Y-axis
    ax.set_ylabel(
        metric.replace("_", " "), fontsize=13, fontweight="medium", labelpad=10
    )
    ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=10)

    # =========================================================================
    # Title
    # =========================================================================
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    else:
        metric_name = metric.replace("_", " ").title()
        ax.set_title(
            f"{metric_name} by Lead Time ({split.capitalize()} Set)",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )

    # =========================================================================
    # Grid
    # =========================================================================
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.set_axisbelow(True)

    # Add subtle spines
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(1)

    # =========================================================================
    # Legend - split into sections
    # =========================================================================
    handles, labels = ax.get_legend_handles_labels()

    # Create a nicely formatted legend
    legend = ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        ncol=1,
        handlelength=2.5,
        handletextpad=0.8,
        labelspacing=0.6,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, ax


def plot_metric_comparison_grouped_legend(
    baselines: dict,
    multi_lead_models: dict,
    single_lead_models: dict,
    metric: str = "MSE_all",
    split: str = "val",
    n_leads: int = 24,
    lead_time_hours: int = 6,
    figsize: tuple = (15, 8),
    ylim: tuple = (0, 0.06),
    save_path: str | None = None,
    title: str | None = None,
):
    """
    Plot metric comparison across lead times for baselines, multi-lead models, and single-lead models.
    Uses colorblind-friendly palettes throughout.
    """

    # Create figure with light grey background
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("#F5F5F5")

    lead_range = np.arange(n_leads)
    lead_hours = (lead_range + 1) * lead_time_hours

    # =========================================================================
    # Colorblind-friendly palettes
    # =========================================================================

    # Baselines: muted/earthy tones (Paul Tol's muted palette)
    baseline_palette = [
        "#CC6677",  # Rose
        "#DDCC77",  # Sand/Gold
        "#117733",  # Green
        "#88CCEE",  # Cyan
        "#882255",  # Wine
        "#44AA99",  # Teal
        "#999933",  # Olive
        "#AA4499",  # Purple
    ]

    # Multi-lead models: vibrant palette (IBM Design)
    model_palette = [
        "#648FFF",  # Blue
        "#785EF0",  # Purple
        "#DC267F",  # Magenta
        "#FE6100",  # Orange
        "#FFB000",  # Gold
        "#22A884",  # Teal
        "#414487",  # Dark blue
        "#2A788E",  # Steel blue
    ]

    # Single-lead models: high-contrast markers
    single_palette = [
        "#D55E00",  # Vermillion
        "#0072B2",  # Blue
        "#009E73",  # Bluish green
        "#E69F00",  # Orange
        "#CC79A7",  # Reddish purple
    ]

    # Store handles for grouped legend
    baseline_handles = []
    baseline_labels = []
    model_handles = []
    model_labels = []
    single_handles = []
    single_labels = []

    # =========================================================================
    # Plot baselines (dashed lines, diamond markers)
    # =========================================================================
    for idx, (name, per_lead_data) in enumerate(baselines.items()):
        values = [per_lead_data[i][metric] for i in range(n_leads)]
        color = baseline_palette[idx % len(baseline_palette)]
        (line,) = ax.plot(
            lead_hours,
            values,
            marker="d",
            markersize=6,
            alpha=0.7,
            linestyle="--",
            linewidth=1.8,
            color=color,
        )
        baseline_handles.append(line)
        baseline_labels.append(name)

    # =========================================================================
    # Plot multi-lead models (solid lines, circle markers)
    # =========================================================================
    for idx, (name, results) in enumerate(multi_lead_models.items()):
        per_lead = results[split]["per_lead"]
        values = [per_lead[i][metric] for i in range(n_leads)]
        color = model_palette[idx % len(model_palette)]
        (line,) = ax.plot(
            lead_hours,
            values,
            marker="o",
            markersize=7,
            alpha=0.9,
            linestyle="-",
            linewidth=2.2,
            color=color,
        )
        model_handles.append(line)
        model_labels.append(name)

    # =========================================================================
    # Plot single-lead models (X markers, no lines)
    # =========================================================================
    for idx, (name, (lead_idx, results)) in enumerate(single_lead_models.items()):
        value = results[split]["global"][metric]
        lead_hour = (lead_idx + 1) * lead_time_hours
        color = single_palette[idx % len(single_palette)]
        (point,) = ax.plot(
            lead_hour,
            value,
            marker="X",
            markersize=12,
            markeredgecolor="white",
            markeredgewidth=2,
            color=color,
            linestyle="none",
            zorder=10,
        )
        single_handles.append(point)
        single_labels.append(name)

    # =========================================================================
    # Axis formatting
    # =========================================================================
    ax.set_xlabel("Lead Time", fontsize=13, fontweight="medium", labelpad=10)
    ax.set_xlim(lead_hours[0] - 3, lead_hours[-1] + 3)

    # Custom x-tick labels
    tick_positions = []
    tick_labels_list = []
    for i, h in enumerate(lead_hours):
        if h % 24 == 0:
            tick_positions.append(h)
            days = h // 24
            tick_labels_list.append(f"{h}h\n({days}d)")
        elif h == 6:
            tick_positions.append(h)
            tick_labels_list.append("6h")
        elif h % 12 == 0:
            tick_positions.append(h)
            tick_labels_list.append(f"{h}h")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list, fontsize=10)

    ax.set_ylabel(
        metric.replace("_", " "), fontsize=13, fontweight="medium", labelpad=10
    )
    ax.set_ylim(ylim)
    ax.tick_params(axis="y", labelsize=10)

    # Title
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    else:
        metric_name = metric.replace("_", " ").title()
        ax.set_title(
            f"{metric_name} by Lead Time ({split.capitalize()} Set)",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )

    # Grid
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
        spine.set_linewidth(1)

    # =========================================================================
    # Grouped Legend
    # =========================================================================
    from matplotlib.lines import Line2D

    section_header = Line2D([0], [0], color="none", label="")

    all_handles = []
    all_labels = []

    # Baselines section
    all_handles.append(section_header)
    all_labels.append(r"$\bf{Baselines}$")
    all_handles.extend(baseline_handles)
    all_labels.extend(baseline_labels)

    # Multi-lead models section
    all_handles.append(section_header)
    all_labels.append(r"$\bf{Multi-Lead\ Models}$")
    all_handles.extend(model_handles)
    all_labels.extend(model_labels)

    # Single-lead models section
    all_handles.append(section_header)
    all_labels.append(r"$\bf{Single-Lead\ Models}$")
    all_handles.extend(single_handles)
    all_labels.extend(single_labels)

    legend = ax.legend(
        all_handles,
        all_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#CCCCCC",
        facecolor="white",
        ncol=1,
        handlelength=2.5,
        handletextpad=0.8,
        labelspacing=0.5,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig, ax


def plot_fss_by_lead(
    baselines: dict,
    multi_lead_models: dict,
    single_lead_models: dict,
    split: str = "val",
    n_leads: int = 24,
    lead_time_hours: int = 6,
    neighborhoods: list[int] = [1, 5, 17],
    figsize: tuple = (16, 5),
    save_path: str | None = None,
    suptitle: str | None = None,
):
    """
    Plot FSS at different spatial scales across lead times.
    One subplot per neighborhood size.
    """
    n_neighborhoods = len(neighborhoods)
    fig, axes = plt.subplots(1, n_neighborhoods, figsize=figsize, facecolor="white")

    if n_neighborhoods == 1:
        axes = [axes]

    lead_range = np.arange(n_leads)
    lead_hours = (lead_range + 1) * lead_time_hours

    # Color palettes
    baseline_palette = [
        "#CC6677",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
        "#AA4499",
    ]
    model_palette = [
        "#648FFF",
        "#785EF0",
        "#DC267F",
        "#FE6100",
        "#FFB000",
        "#22A884",
        "#414487",
        "#2A788E",
    ]
    single_palette = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7"]

    for ax_idx, (ax, n) in enumerate(zip(axes, neighborhoods)):
        ax.set_facecolor("#F5F5F5")
        metric = f"FSS_{n}"

        # Plot baselines
        for idx, (name, per_lead_data) in enumerate(baselines.items()):
            values = [per_lead_data[i].get(metric, np.nan) for i in range(n_leads)]
            color = baseline_palette[idx % len(baseline_palette)]
            ax.plot(
                lead_hours,
                values,
                marker="d",
                markersize=5,
                alpha=0.6,
                linestyle="--",
                linewidth=1.5,
                color=color,
                label=name if ax_idx == 0 else None,
            )

        # Plot multi-lead models
        for idx, (name, results) in enumerate(multi_lead_models.items()):
            per_lead = results[split]["per_lead"]
            values = [per_lead[i].get(metric, np.nan) for i in range(n_leads)]
            color = model_palette[idx % len(model_palette)]
            ax.plot(
                lead_hours,
                values,
                marker="o",
                markersize=6,
                alpha=0.9,
                linestyle="-",
                linewidth=2,
                color=color,
                label=name if ax_idx == 0 else None,
            )

        # Plot single-lead models
        for idx, (name, (lead_idx, results)) in enumerate(single_lead_models.items()):
            value = results[split]["global"].get(metric, np.nan)
            lead_hour = (lead_idx + 1) * lead_time_hours
            color = single_palette[idx % len(single_palette)]
            ax.plot(
                lead_hour,
                value,
                marker="X",
                markersize=14,
                markeredgecolor="white",
                markeredgewidth=1.5,
                color=color,
                linestyle="none",
                zorder=10,
                label=name if ax_idx == 0 else None,
            )

        # Reference line at FSS = 0.5 (useful skill threshold)
        ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.6, linewidth=1.5)
        ax.text(
            lead_hours[-1] + 2,
            0.5,
            "Useful",
            fontsize=8,
            color="red",
            va="center",
            alpha=0.7,
        )

        # Formatting
        ax.set_ylim(0, 1)
        ax.set_xlim(lead_hours[0] - 3, lead_hours[-1] + 3)
        ax.set_xlabel("Lead Time", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("FSS", fontsize=11)

        # Scale description
        if n == 1:
            scale_desc = "Pixel-level"
        else:
            scale_desc = f"{n}×{n} pixels"
        ax.set_title(f"FSS (n={n})\n{scale_desc}", fontsize=12, fontweight="bold")

        # X-ticks
        tick_positions = [h for h in lead_hours if h % 24 == 0 or h == 6]
        tick_labels = [f"{h}h" if h < 24 else f"{h // 24}d" for h in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)

        # Grid
        ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.show()
    return fig, axes


def plot_sal_by_lead(
    baselines: dict,
    multi_lead_models: dict,
    single_lead_models: dict,
    split: str = "val",
    n_leads: int = 24,
    lead_time_hours: int = 6,
    figsize: tuple = (16, 5),
    save_path: str | None = None,
    suptitle: str | None = None,
):
    """
    Plot SAL components (Structure, Amplitude, Location) across lead times.
    One subplot per component.
    """
    sal_components = [
        ("SAL_S", "Structure (S)", (-2, 2), "Flat/Large ←  Perfect  → Peaked/Small"),
        ("SAL_A", "Amplitude (A)", (-2, 2), "Under ←  Perfect  → Over"),
        ("SAL_L", "Location (L)", (0, 0.3), "Perfect → Displaced"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")

    lead_range = np.arange(n_leads)
    lead_hours = (lead_range + 1) * lead_time_hours

    # Color palettes
    baseline_palette = [
        "#CC6677",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
        "#AA4499",
    ]
    model_palette = [
        "#648FFF",
        "#785EF0",
        "#DC267F",
        "#FE6100",
        "#FFB000",
        "#22A884",
        "#414487",
        "#2A788E",
    ]
    single_palette = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7"]

    for ax_idx, (ax, (metric, title, ylim, ylabel_desc)) in enumerate(
        zip(axes, sal_components)
    ):
        ax.set_facecolor("#F5F5F5")

        # Plot baselines
        for idx, (name, per_lead_data) in enumerate(baselines.items()):
            values = [per_lead_data[i].get(metric, np.nan) for i in range(n_leads)]
            color = baseline_palette[idx % len(baseline_palette)]
            ax.plot(
                lead_hours,
                values,
                marker="d",
                markersize=5,
                alpha=0.6,
                linestyle="--",
                linewidth=1.5,
                color=color,
                label=name if ax_idx == 0 else None,
            )

        # Plot multi-lead models
        for idx, (name, results) in enumerate(multi_lead_models.items()):
            per_lead = results[split]["per_lead"]
            values = [per_lead[i].get(metric, np.nan) for i in range(n_leads)]
            color = model_palette[idx % len(model_palette)]
            ax.plot(
                lead_hours,
                values,
                marker="o",
                markersize=6,
                alpha=0.9,
                linestyle="-",
                linewidth=2,
                color=color,
                label=name if ax_idx == 0 else None,
            )

        # Plot single-lead models
        for idx, (name, (lead_idx, results)) in enumerate(single_lead_models.items()):
            value = results[split]["global"].get(metric, np.nan)
            lead_hour = (lead_idx + 1) * lead_time_hours
            color = single_palette[idx % len(single_palette)]
            ax.plot(
                lead_hour,
                value,
                marker="X",
                markersize=14,
                markeredgecolor="white",
                markeredgewidth=1.5,
                color=color,
                linestyle="none",
                zorder=10,
                label=name if ax_idx == 0 else None,
            )

        # Reference line at 0 (perfect score for S and A)
        if metric in ["SAL_S", "SAL_A"]:
            ax.axhline(y=0, color="green", linestyle=":", alpha=0.6, linewidth=1.5)
            ax.text(
                lead_hours[-1] + 2,
                0,
                "Perfect",
                fontsize=8,
                color="green",
                va="center",
                alpha=0.7,
            )
        elif metric == "SAL_L":
            ax.axhline(y=0, color="green", linestyle=":", alpha=0.6, linewidth=1.5)
            ax.text(
                lead_hours[-1] + 2,
                0,
                "Perfect",
                fontsize=8,
                color="green",
                va="center",
                alpha=0.7,
            )

        # Formatting
        ax.set_ylim(ylim)
        ax.set_xlim(lead_hours[0] - 3, lead_hours[-1] + 3)
        ax.set_xlabel("Lead Time", fontsize=11)
        ax.set_ylabel(ylabel_desc, fontsize=9, style="italic")
        ax.set_title(title, fontsize=12, fontweight="bold")

        # X-ticks
        tick_positions = [h for h in lead_hours if h % 24 == 0 or h == 6]
        tick_labels = [f"{h}h" if h < 24 else f"{h // 24}d" for h in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)

        # Grid
        ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.show()
    return fig, axes


def plot_fss_sal_combined(
    baselines: dict,
    multi_lead_models: dict,
    single_lead_models: dict,
    split: str = "val",
    n_leads: int = 24,
    lead_time_hours: int = 6,
    fss_neighborhood: int = 5,
    figsize: tuple = (16, 8),
    save_path: str | None = None,
    suptitle: str | None = None,
):
    """
    Combined plot: FSS (one scale) + all 3 SAL components in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor="white")
    axes = axes.flatten()

    metrics_config = [
        (
            f"FSS_{fss_neighborhood}",
            f"FSS (n={fss_neighborhood})",
            (0, 1),
            "FSS",
            0.5,
            "Useful skill",
        ),
        ("SAL_S", "Structure (S)", (-1.5, 1.5), "S", 0, "Perfect"),
        ("SAL_A", "Amplitude (A)", (-1.5, 1.5), "A", 0, "Perfect"),
        ("SAL_L", "Location (L)", (0, 0.8), "L", 0, "Perfect"),
    ]

    lead_range = np.arange(n_leads)
    lead_hours = (lead_range + 1) * lead_time_hours

    # Color palettes
    baseline_palette = [
        "#CC6677",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
        "#AA4499",
    ]
    model_palette = [
        "#648FFF",
        "#785EF0",
        "#DC267F",
        "#FE6100",
        "#FFB000",
        "#22A884",
        "#414487",
        "#2A788E",
    ]
    single_palette = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7"]

    for ax_idx, (ax, (metric, title, ylim, ylabel, ref_val, ref_label)) in enumerate(
        zip(axes, metrics_config)
    ):
        ax.set_facecolor("#F5F5F5")

        # Plot baselines
        for idx, (name, per_lead_data) in enumerate(baselines.items()):
            values = [per_lead_data[i].get(metric, np.nan) for i in range(n_leads)]
            color = baseline_palette[idx % len(baseline_palette)]
            ax.plot(
                lead_hours,
                values,
                marker="d",
                markersize=4,
                alpha=0.6,
                linestyle="--",
                linewidth=1.3,
                color=color,
                label=name if ax_idx == 0 else None,
            )

        # Plot multi-lead models
        for idx, (name, results) in enumerate(multi_lead_models.items()):
            per_lead = results[split]["per_lead"]
            values = [per_lead[i].get(metric, np.nan) for i in range(n_leads)]
            color = model_palette[idx % len(model_palette)]
            ax.plot(
                lead_hours,
                values,
                marker="o",
                markersize=5,
                alpha=0.9,
                linestyle="-",
                linewidth=1.8,
                color=color,
                label=name if ax_idx == 0 else None,
            )

        # Plot single-lead models
        for idx, (name, (lead_idx, results)) in enumerate(single_lead_models.items()):
            value = results[split]["global"].get(metric, np.nan)
            lead_hour = (lead_idx + 1) * lead_time_hours
            color = single_palette[idx % len(single_palette)]
            ax.plot(
                lead_hour,
                value,
                marker="X",
                markersize=12,
                markeredgecolor="white",
                markeredgewidth=1.5,
                color=color,
                linestyle="none",
                zorder=10,
                label=name if ax_idx == 0 else None,
            )

        # Reference line
        ref_color = "green" if metric != f"FSS_{fss_neighborhood}" else "red"
        ax.axhline(y=ref_val, color=ref_color, linestyle=":", alpha=0.6, linewidth=1.5)

        # Formatting
        ax.set_ylim(ylim)
        ax.set_xlim(lead_hours[0] - 3, lead_hours[-1] + 3)
        ax.set_xlabel("Lead Time", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # X-ticks
        tick_positions = [h for h in lead_hours if h % 24 == 0 or h == 6]
        tick_labels = [f"{h}h" if h < 24 else f"{h // 24}d" for h in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9)

        # Grid
        ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.show()
    return fig, axes