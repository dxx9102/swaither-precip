"""
Spider plot with lead-time quadrants — supports single-horizon models.

Contains helpers that create Spider plots and are used within the Step2 evaluation notebook.

Models that only cover specific lead times appear only in their
respective quadrant(s), while multi-horizon models span all quadrants.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from math import pi


def _format_lead(lead_idx):
    h = (int(lead_idx) + 1) * 6
    if h >= 48 and h % 24 == 0:
        return f"{h // 24}d"
    return f"{h}h"


"""
Spider plot with lead-time quadrants — supports single-horizon models.

Models that only cover specific lead times appear only in their
respective quadrant(s), while multi-horizon models span all quadrants.
"""

def spider_plot_unified3(
    datasets,
    metrics,
    leads=(0, 3, 11, 23),
    higher_is_better=None,
    metric_labels=None,
    colors=None,
    fill_alpha=0.07,
    line_alpha=0.85,
    figsize=(5, 5),
    title=None,
    show_values=True,
    quadrant_colors=None,
    model_lead_map=None,
):
    """
    Single radar chart with 4 lead-time quadrants.

    The circle is divided into 4 "pizza slices", one per lead.
    Within each slice, the spokes correspond to the chosen metrics.
    All models are overlaid so their areas can be compared directly.

    Multi-horizon models span all quadrants; single-horizon models
    appear only in the quadrant(s) matching their lead time.

    Parameters
    ----------
    datasets : dict[str, xr.Dataset]
        {model_name: xarray_dataset}.
    metrics : list[str]
        Metric names (repeated in each quadrant).
    leads : list of 4 int
        Lead-time indices for the 4 quadrants.
    higher_is_better : list[str] or None
        Metrics where higher = better.
    metric_labels : dict[str, str] or None
        Pretty names for metrics.
    colors : list[str] or None
        One per model.
    fill_alpha, line_alpha : float
        Visual styling.
    figsize : tuple
    title : str or None
    show_values : bool
        Annotate raw values at vertices in each model's color.
    quadrant_colors : list of 4 str or None
        Background tint for each quadrant.
    model_lead_map : dict[str, list[int]] or None
        Maps model_name -> list of lead indices it covers.
        Models not listed are assumed to cover ALL leads.
        Example: {"UNet Single 6h": [0], "UNet Single 3d": [11]}

    Returns
    -------
    fig : matplotlib.Figure
    """
    assert len(leads) in (3, 4), "Provide 3 or 4 leads."
    # Always 4 visual quadrants; pad with None if only 3 leads
    n_quadrants = 4
    if len(leads) == 3:
        leads = list(leads) + [None]  # 4th quadrant = empty

    model_names = list(datasets.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    if higher_is_better is None:
        higher_is_better = []
    if metric_labels is None:
        metric_labels = {}
    if colors is None:
        _cb_palette = [
            "#0072B2",  # blue
            "#D55E00",  # vermillion
            "#009E73",  # bluish green
            "#CC79A7",  # reddish purple
            "#E69F00",  # orange
            "#56B4E9",  # sky blue
            "#F0E442",  # yellow
            "#000000",  # black
        ]
        colors = [_cb_palette[i % len(_cb_palette)] for i in range(n_models)]
    if quadrant_colors is None:
        base = ["#FFFFFF", "#F0F0F0"]
        quadrant_colors = [base[i % 2] for i in range(n_quadrants)]
    if model_lead_map is None:
        model_lead_map = {}

    display = [metric_labels.get(m, m) for m in metrics]

    # ── Helper: check if model covers a given lead ──
    def _model_has_lead(model_name, lead_idx):
        if lead_idx is None:
            return False  # empty quadrant
        if model_name not in model_lead_map:
            return True  # multi-horizon → all leads
        return lead_idx in model_lead_map[model_name]

    # ── 1) Collect raw values: (n_models, n_quadrants, n_metrics), NaN where missing ──
    raw = np.full((n_models, n_quadrants, n_metrics), np.nan)
    for mi, name in enumerate(model_names):
        ds = datasets[name]
        for li, lead in enumerate(leads):
            if not _model_has_lead(name, lead):
                continue
            for ki, metric in enumerate(metrics):
                vals = ds[metric].sel(lead=lead).values
                raw[mi, li, ki] = float(np.nanmean(vals))

    # ── 2) Normalize: outward = better, across ALL models & leads ──
    normed = np.full_like(raw, np.nan)
    norm_bounds = {}  # {metric_idx: (vmin, vmax, higher_better)}
    for ki, metric in enumerate(metrics):
        all_vals = raw[:, :, ki].ravel()
        valid = all_vals[~np.isnan(all_vals)]
        if len(valid) == 0:
            continue
        vmin, vmax = np.nanmin(valid), np.nanmax(valid)
        hib = metric in higher_is_better
        norm_bounds[ki] = (vmin, vmax, hib)
        if vmax - vmin < 1e-12:
            normed[:, :, ki] = 0.5
        elif hib:
            normed[:, :, ki] = (raw[:, :, ki] - vmin) / (vmax - vmin)
        else:
            normed[:, :, ki] = 1.0 - (raw[:, :, ki] - vmin) / (vmax - vmin)

    normed = np.clip(normed, 0.02, 1.0)

    # ── 3) Build the angle vector ──
    angles = []
    for qi in range(n_quadrants):
        start = qi * (2 * pi / n_quadrants)
        for ki in range(n_metrics):
            a = start + ki * (2 * pi / n_quadrants) / n_metrics
            angles.append(a)
    angles = np.array(angles)

    # ── 4) Build spoke labels (blank for empty quadrants) ──
    spoke_labels = []
    for qi in range(n_quadrants):
        for ki in range(n_metrics):
            if leads[qi] is None:
                spoke_labels.append("")
            else:
                spoke_labels.append(display[ki])

    # ── 5) Plot ──
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Quadrant backgrounds
    slice_deg = 360 / n_quadrants
    for qi in range(n_quadrants):
        start_deg = qi * slice_deg
        wedge = Wedge(
            (0.5, 0.5), 0.5, 90 - start_deg - slice_deg, 90 - start_deg,
            transform=ax.transAxes,
            facecolor=quadrant_colors[qi], alpha=0.2,
            edgecolor="none", zorder=0,
        )
        ax.add_patch(wedge)

    # Quadrant separator lines
    for qi in range(n_quadrants):
        boundary_angle = qi * (2 * pi / n_quadrants)
        ax.plot([boundary_angle, boundary_angle], [0, 1.15],
                color="#BBB", lw=2.0, ls="-", zorder=0, clip_on=False)

    # Quadrant labels (lead times)
    for qi, lead in enumerate(leads):
        if lead is None:
            continue
        mid_angle = qi * (2 * pi / n_quadrants) + (2 * pi / n_quadrants) / 2
        ax.text(
            mid_angle, 1.50, _format_lead(lead),
            ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333",
            fontfamily="sans-serif",
        )

    # Axis setup
    ax.set_xticks(angles)
    ax.set_xticklabels(spoke_labels, fontsize=9.5, color="darkslategrey") #"#555"
    ax.tick_params(axis='x', pad=13)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], fontsize=6)
    ax.set_rlabel_position(0)
    ax.spines["polar"].set_color("#DDD")
    ax.grid(color="gainsboro", linewidth=0.4)
    ax.set_facecolor("white")

    # ── 6) Plot each model ──
    for mi, name in enumerate(model_names):
        is_partial = name in model_lead_map

        if not is_partial:
            # ── Multi-horizon model: polygon through active quadrants only ──
            active_angles = []
            active_values = []
            for qi in range(n_quadrants):
                if leads[qi] is None:
                    continue
                for ki in range(n_metrics):
                    active_angles.append(angles[qi * n_metrics + ki])
                    active_values.append(normed[mi, qi, ki])
            # Close the polygon
            active_angles_closed = np.append(active_angles, active_angles[0])
            active_values_closed = active_values + [active_values[0]]

            ax.plot(
                active_angles_closed, active_values_closed,
                linewidth=1.8, color=colors[mi],
                alpha=line_alpha, label=name,
            )
            ax.fill(
                active_angles_closed, active_values_closed,
                color=colors[mi], alpha=fill_alpha,
            )
        else:
            # ── Single-horizon model: draw arcs only in active quadrants ──
            # Plot each active quadrant as a separate segment that closes
            # through the origin, creating a filled "wedge"
            for li, lead in enumerate(leads):
                if not _model_has_lead(name, lead):
                    continue

                # Indices for this quadrant's spokes
                q_start = li * n_metrics
                q_indices = list(range(q_start, q_start + n_metrics))

                q_angles = angles[q_indices]
                q_values = [normed[mi, li, ki] for ki in range(n_metrics)]

                # Close the arc through the center for a wedge shape
                arc_angles = np.concatenate([
                    [q_angles[0]], q_angles, [q_angles[-1]]
                ])
                arc_values = np.concatenate([
                    [0.0], q_values, [0.0]
                ])

                ax.plot(
                    q_angles, q_values,
                    linewidth=2.2, color=colors[mi],
                    alpha=line_alpha, zorder=10,
                    label=name if lead == model_lead_map[name][0] else None,
                )
                ax.fill(
                    arc_angles, arc_values,
                    color=colors[mi], alpha=fill_alpha * 1.5, zorder=9,
                )

                # Dots at vertices for visibility
                ax.scatter(
                    q_angles, q_values,
                    color=colors[mi], s=18, zorder=12, alpha=line_alpha,
                )

        # (per-model annotations removed — replaced by tick-scale labels below)

    # ── 7) Tick-scale labels: show raw metric values at radial ticks ──
    # Printed only in the first active quadrant, small and grey.
    tick_levels = [0.25, 0.50, 0.75, 1.0]
    # Find first active quadrant
    ref_qi = next((qi for qi in range(n_quadrants) if leads[qi] is not None), 0)

    for ki in range(n_metrics):
        if ki not in norm_bounds:
            continue
        vmin, vmax, hib = norm_bounds[ki]
        spoke_angle = angles[ref_qi * n_metrics + ki]

        for tick in tick_levels:
            # Back-compute raw value at this tick
            if hib:
                raw_val = vmin + tick * (vmax - vmin)
            else:
                raw_val = vmin + (1.0 - tick) * (vmax - vmin)

            # Format: use fewer decimals for large values
            if abs(raw_val) >= 10:
                label = f"{raw_val:.1f}"
            else:
                label = f"{raw_val:.3f}"

            ax.text(
                spoke_angle, tick,
                label,
                fontsize=5.5, color="black", #"#999"
                ha="center", va="center",
                alpha=0.7, zorder=14,
                bbox=dict(
                    facecolor="white", edgecolor="none",
                    alpha=0.6, pad=0.8,
                ),
            )

    # ── Legend at bottom ──
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=min(n_models, 4),
        fontsize=10, frameon=True, fancybox=True,
        columnspacing=1.5,
    )

    # Note
    note = "↑ outward = better on all axes"
    if higher_is_better:
        inverted = [m for m in metrics if m not in higher_is_better]
        if inverted:
            inv_labels = [metric_labels.get(m, m) for m in inverted]
            note += f"  (inverted: {', '.join(inv_labels)})"
    fig.text(0.5, -0.02, note, ha="center", fontsize=9, color="#999", style="italic")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    return fig