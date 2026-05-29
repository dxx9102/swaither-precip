"""
Table helpers.

Contains helpers that are used within the Step2 evaluation notebook, to create LateX tables from the extracted metrics on the super resolution samples.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Optional

# =============================================================================
# Lead Time Mapping
# =============================================================================

# Your custom lead time mapping: position in array -> hours
# Position 0 = 6h, Position 1 = 12h, etc.
LEAD_POSITION_TO_HOURS = {
    0: 6,
    1: 12,
    2: 18,
    3: 24,
    4: 48,   # 2 days
    5: 72,   # 3 days
    6: 96,   # 4 days
    7: 120,  # 5 days
    8: 144,  # 6 days
}

# List version for easy indexing
LEAD_HOURS_LIST = [6, 12, 18, 24, 48, 72, 96, 120, 144]


def hours_to_label(hours: int) -> str:
    """Convert hours to human-readable label."""
    if hours < 24:
        return f"{hours}h"
    elif hours % 24 == 0:
        days = hours // 24
        return f"{days}d"
    else:
        return f"{hours}h"


def position_to_label(position: int) -> str:
    """Convert array position to human-readable label."""
    hours = LEAD_POSITION_TO_HOURS.get(position, position * 6)
    return hours_to_label(hours)


def inspect_dataset_leads(ds: xr.Dataset):
    """
    Debug function to inspect the lead coordinate in a dataset.
    """
    if "lead" not in ds.coords:
        print("No 'lead' coordinate found!")
        print(f"Available coords: {list(ds.coords.keys())}")
        return
    
    leads = ds["lead"].values
    print(f"Lead coordinate dtype: {leads.dtype}")
    print(f"Lead values: {leads}")
    print(f"Number of leads: {len(leads)}")
    
    print("\nExpected mapping (by position):")
    for i, l in enumerate(leads):
        hours = LEAD_POSITION_TO_HOURS.get(i, "unknown")
        print(f"  Position {i}: lead coord value = {l}, should map to {hours} hours ({hours_to_label(hours) if isinstance(hours, int) else hours})")

# =============================================================================
# Helper Functions
# =============================================================================

def extract_metrics_per_lead(
    ds: xr.Dataset,
    metrics: list[str] | None = None,
) -> dict:
    """
    Extract metrics per lead time from xr.Dataset.
    
    Returns dictionary keyed by POSITION (0, 1, 2, ...) not by lead coordinate value.
    
    Args:
        ds: Dataset from compute_metrics_on_latlon_window_extended
        metrics: List of metrics to extract (default: all)
    
    Returns:
        Dictionary {position: {metric: value, "_hours": hours}}
    """
    if metrics is None:
        metrics = [v for v in ds.data_vars.keys()]
    
    per_lead = {}
    lead_values = ds["lead"].values
    
    for position, lead_val in enumerate(lead_values):
        # Select by the actual coordinate value
        lead_ds = ds.sel(lead=lead_val)
        
        per_lead[position] = {
            m: float(lead_ds[m].mean(dim="time", skipna=True).values)
            for m in metrics if m in ds.data_vars
        }
        # Store the hours for this position
        per_lead[position]["_hours"] = LEAD_POSITION_TO_HOURS.get(position, position * 6)
        per_lead[position]["_lead_value"] = lead_val
    
    return per_lead


def extract_global_metrics(
    ds: xr.Dataset,
    metrics: list[str] | None = None,
) -> dict:
    """
    Extract global metrics (averaged over time and lead).
    """
    if metrics is None:
        metrics = list(ds.data_vars.keys())
    
    return {
        m: float(ds[m].mean(skipna=True).values)
        for m in metrics if m in ds.data_vars
    }


# =============================================================================
# LaTeX Table Generation Functions
# =============================================================================

def generate_ensemble_latex_table_compact(
    model_results: dict,
    metrics: list[str] | None = None,
    precision: int = 3,
    highlight_best: bool = True,
    save_path: str | None = None,
    caption: str = "Ensemble super-resolution metrics (averaged across all leads)",
    label: str = "tab:sr_metrics_compact",
) -> tuple[str, pd.DataFrame]:
    """
    Generate a compact LaTeX table with average metrics for multiple models.
    """
    if metrics is None:
        metrics = ["crps", "mse", "mae", "csi", "brier_score", 
                   "avfss_1", "avfss_5", "eS", "eA", "eL", 
                   "spread_skill_ratio"]
    
    rows = []
    for name, results in model_results.items():
        if isinstance(results, xr.Dataset):
            global_metrics = extract_global_metrics(results, metrics)
        else:
            global_metrics = results
        
        row = {"Model": name}
        for m in metrics:
            row[m] = global_metrics.get(m, np.nan)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    higher_is_better = {
        "csi", "pod", "avfss_1", "avfss_5", "avfss_17",
        "spectral_correlation",
    }
    
    best = {}
    if highlight_best:
        for m in metrics:
            if m not in df.columns or m == "Model":
                continue
            valid = df[m].dropna()
            if valid.empty:
                continue
            
            if m == "spread_skill_ratio":
                best[m] = valid.iloc[(valid - 1.0).abs().argmin()]
            elif m in higher_is_better:
                best[m] = valid.max()
            else:
                best[m] = valid.min()
    
    def fmt(val, is_best=False):
        if pd.isna(val) or np.isnan(val):
            return "—"
        s = f"{val:.{precision}f}"
        return f"\\textbf{{{s}}}" if is_best else s
    
    def clean_name(m):
        return m.replace("_", "\\_")
    
    cols = list(df.columns)
    n_metrics = len(cols) - 1
    col_format = "l" + "c" * n_metrics
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule",
    ]
    
    header = ["Model"] + [clean_name(m) for m in metrics]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    
    for _, row in df.iterrows():
        row_strs = [str(row["Model"]).replace("_", "\\_")]
        for m in metrics:
            val = row.get(m, np.nan)
            is_best = highlight_best and best.get(m) is not None and val == best[m]
            row_strs.append(fmt(val, is_best))
        lines.append(" & ".join(row_strs) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    
    latex_str = "\n".join(lines)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"Saved: {save_path}")
    
    print(latex_str)
    
    return latex_str, df


def generate_ensemble_latex_table_per_lead(
    model_results: dict,
    metric: str,
    positions_to_show: list[int] | None = None,
    precision: int = 3,
    highlight_best: bool = True,
    save_path: str | None = None,
    caption: str | None = None,
    label: str | None = None,
    debug: bool = False,
) -> tuple[str, pd.DataFrame]:
    """
    Generate LaTeX table for a single metric across lead times.
    
    Args:
        model_results: Dict of {model_name: xr.Dataset}
        metric: Metric to display
        positions_to_show: Array POSITIONS to include (0=6h, 1=12h, 2=18h, 3=24h, 4=48h, 5=72h, 6=96h, 7=120h, 8=144h)
                          Default: [0, 3, 5, 8] for 6h, 24h, 3d, 6d
        precision: Decimal places
        highlight_best: Bold the best value per column
        save_path: Path to save LaTeX file
        caption: Table caption
        label: Table label
        debug: Print debug information
    
    Returns:
        LaTeX string and DataFrame
    """
    # Default: show 6h, 24h (1d), 72h (3d), 144h (6d)
    if positions_to_show is None:
        positions_to_show = [0, 3, 5, 8]
    
    if caption is None:
        caption = f"{metric.replace('_', ' ').upper()} by lead time"
    if label is None:
        label = f"tab:sr_{metric.lower()}"
    
    higher_is_better_metrics = {
        "csi", "pod", "avfss_1", "avfss_5", "avfss_17",
        "spectral_correlation",
    }
    hib = metric in higher_is_better_metrics
    
    # Get number of leads from first model
    first_result = next(iter(model_results.values()))
    if isinstance(first_result, xr.Dataset):
        n_leads = len(first_result["lead"])
    else:
        n_leads = len(first_result)
    
    # Filter positions to those that exist
    positions_to_show = [p for p in positions_to_show if p < n_leads]
    
    if debug:
        print(f"Number of leads in dataset: {n_leads}")
        print(f"Positions to show: {positions_to_show}")
        print(f"Corresponding hours: {[LEAD_POSITION_TO_HOURS.get(p, '?') for p in positions_to_show]}")
    
    if len(positions_to_show) == 0:
        raise ValueError(f"No valid positions to show. Dataset has {n_leads} leads.")
    
    # Build rows
    rows = []
    for name, results in model_results.items():
        if isinstance(results, xr.Dataset):
            per_lead = extract_metrics_per_lead(results, [metric])
        else:
            per_lead = results
        
        if debug:
            print(f"\nModel: {name}")
            print(f"  Per-lead keys: {list(per_lead.keys())}")
        
        row = {"Model": name}
        
        for pos in positions_to_show:
            hours = LEAD_POSITION_TO_HOURS.get(pos, pos * 6)
            label_col = hours_to_label(hours)
            
            val = per_lead.get(pos, {}).get(metric, np.nan)
            row[label_col] = val
            
            if debug:
                print(f"  Position {pos} ({label_col}): {val}")
        
        # Average across all leads
        all_vals = [per_lead.get(i, {}).get(metric, np.nan) for i in range(n_leads)]
        valid_vals = [v for v in all_vals if not np.isnan(v)]
        row["Avg"] = np.mean(valid_vals) if valid_vals else np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Find best per column
    best = {}
    if highlight_best:
        for col in df.columns:
            if col == "Model":
                continue
            valid = df[col].dropna()
            if valid.empty:
                continue
            
            if metric == "spread_skill_ratio":
                best[col] = valid.iloc[(valid - 1.0).abs().argmin()]
            elif hib:
                best[col] = valid.max()
            else:
                best[col] = valid.min()
    
    def fmt(val, is_best=False):
        if pd.isna(val) or np.isnan(val):
            return "—"
        s = f"{val:.{precision}f}"
        return f"\\textbf{{{s}}}" if is_best else s
    
    # Build LaTeX
    cols = list(df.columns)
    col_format = "l" + "c" * (len(cols) - 1)
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    
    for _, row in df.iterrows():
        row_strs = [str(row["Model"]).replace("_", "\\_")]
        for col in cols[1:]:
            val = row[col]
            is_best = highlight_best and best.get(col) is not None and val == best[col]
            row_strs.append(fmt(val, is_best))
        lines.append(" & ".join(row_strs) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_str = "\n".join(lines)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"Saved: {save_path}")
    
    print(latex_str)
    
    return latex_str, df


def generate_ensemble_latex_table_grouped(
    model_results: dict,
    precision: int = 3,
    highlight_best: bool = True,
    save_path: str | None = None,
    caption: str = "Ensemble super-resolution evaluation",
    label: str = "tab:sr_grouped",
) -> tuple[str, pd.DataFrame]:
    """
    Generate a grouped LaTeX table with multiple metric categories.
    """
    metric_groups = {
        "Probabilistic": ["crps", "brier_score", "spread_skill_ratio"],
        "Deterministic": ["mse", "mae", "csi"],
        "Spatial": ["avfss_1", "avfss_5", "eS", "eA", "eL"],
    }
    
    all_metrics = []
    for group_metrics in metric_groups.values():
        all_metrics.extend(group_metrics)
    
    higher_is_better = {"csi", "avfss_1", "avfss_5", "avfss_17"}
    
    rows = []
    for name, results in model_results.items():
        if isinstance(results, xr.Dataset):
            global_metrics = extract_global_metrics(results, all_metrics)
        else:
            global_metrics = results
        
        row = {"Model": name}
        for m in all_metrics:
            row[m] = global_metrics.get(m, np.nan)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    best = {}
    if highlight_best:
        for m in all_metrics:
            if m not in df.columns:
                continue
            valid = df[m].dropna()
            if valid.empty:
                continue
            
            if m == "spread_skill_ratio":
                best[m] = valid.iloc[(valid - 1.0).abs().argmin()]
            elif m in higher_is_better:
                best[m] = valid.max()
            else:
                best[m] = valid.min()
    
    def fmt(val, is_best=False):
        if pd.isna(val) or np.isnan(val):
            return "—"
        s = f"{val:.{precision}f}"
        return f"\\textbf{{{s}}}" if is_best else s
    
    def clean_name(m):
        replacements = {
            "crps": "CRPS",
            "brier_score": "Brier",
            "spread_skill_ratio": "SSR",
            "mse": "MSE",
            "mae": "MAE",
            "csi": "CSI",
            "avfss_1": "avFSS$_1$",
            "avfss_5": "avFSS$_5$",
            "eS": "$eS$",
            "eA": "$eA$",
            "eL": "$eL$",
        }
        return replacements.get(m, m.replace("_", "\\_"))
    
    n_metrics = len(all_metrics)
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * n_metrics + "}",
        "\\toprule",
    ]
    
    group_header = [""]
    for group_name, group_metrics in metric_groups.items():
        n = len(group_metrics)
        group_header.append(f"\\multicolumn{{{n}}}{{c}}{{{group_name}}}")
    lines.append(" & ".join(group_header) + " \\\\")
    
    col_idx = 2
    cmidrules = []
    for group_metrics in metric_groups.values():
        n = len(group_metrics)
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n - 1}}}")
        col_idx += n
    lines.append(" ".join(cmidrules))
    
    metric_header = ["Model"] + [clean_name(m) for m in all_metrics]
    lines.append(" & ".join(metric_header) + " \\\\")
    lines.append("\\midrule")
    
    for _, row in df.iterrows():
        row_strs = [str(row["Model"]).replace("_", "\\_")]
        for m in all_metrics:
            val = row.get(m, np.nan)
            is_best = highlight_best and best.get(m) is not None and val == best[m]
            row_strs.append(fmt(val, is_best))
        lines.append(" & ".join(row_strs) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    
    latex_str = "\n".join(lines)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"Saved: {save_path}")
    
    print(latex_str)
    
    return latex_str, df


def generate_ensemble_latex_full(
    model_results: dict,
    metrics: list[str] | None = None,
    positions_to_show: list[int] | None = None,
    precision: int = 3,
    highlight_best: bool = True,
    save_path: str | None = None,
    caption: str = "Ensemble super-resolution metrics",
    label: str = "tab:sr_full",
) -> str:
    """
    Generate comprehensive LaTeX output.
    """
    if metrics is None:
        metrics = ["crps", "mse", "csi", "avfss_5", "eL", "spread_skill_ratio"]
    
    if positions_to_show is None:
        positions_to_show = [0, 3, 5, 8]  # 6h, 24h, 3d, 6d
    
    latex_parts = []
    
    latex_compact, _ = generate_ensemble_latex_table_compact(
        model_results,
        metrics=metrics,
        precision=precision,
        highlight_best=highlight_best,
        caption=f"{caption} (averaged)",
        label=f"{label}_avg",
    )
    latex_parts.append("% === Average Metrics ===")
    latex_parts.append(latex_compact)
    
    key_metrics_per_lead = ["crps", "csi", "avfss_5"]
    
    for metric in key_metrics_per_lead:
        latex_metric, _ = generate_ensemble_latex_table_per_lead(
            model_results,
            metric=metric,
            positions_to_show=positions_to_show,
            precision=precision,
            highlight_best=highlight_best,
            caption=f"{metric.upper().replace('_', ' ')} by lead time",
            label=f"{label}_{metric}",
        )
        latex_parts.append(f"\n% === {metric.upper()} per Lead ===")
        latex_parts.append(latex_metric)
    
    latex_str = "\n\n".join(latex_parts)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"Saved: {save_path}")
    
    return latex_str


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_metrics_by_lead_sr(
    model_results: dict,
    metrics: list[str] | None = None,
    figsize: tuple = (14, 10),
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Plot selected metrics across lead times for super-resolution models.
    
    Args:
        model_results: Dict of {model_name: xr.Dataset}
        metrics: Metrics to plot
        figsize: Figure size
        title: Overall title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    if metrics is None:
        metrics = ["crps", "mse", "csi", "avfss_5", "eL", "spread_skill_ratio"]
    
    # Get number of leads from first model
    first_result = next(iter(model_results.values()))
    if isinstance(first_result, xr.Dataset):
        n_leads = len(first_result["lead"])
    else:
        n_leads = len([k for k in first_result.keys() if isinstance(k, int)])
    
    # Build x-axis: hours
    lead_hours = [LEAD_POSITION_TO_HOURS.get(i, i * 6) for i in range(n_leads)]
    
    # Filter to available metrics
    if isinstance(first_result, xr.Dataset):
        metrics = [m for m in metrics if m in first_result.data_vars]
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("No valid metrics to plot")
    
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    axes = np.atleast_2d(axes).flatten()
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.set_facecolor('#F5F5F5')
        
        for color, (name, results) in zip(colors, model_results.items()):
            if isinstance(results, xr.Dataset):
                per_lead = extract_metrics_per_lead(results, [metric])
            else:
                per_lead = results
            
            values = [per_lead.get(i, {}).get(metric, np.nan) for i in range(n_leads)]
            
            ax.plot(lead_hours, values, marker='o', linewidth=2, 
                   color=color, label=name if idx == 0 else None)
        
        # Reference lines for specific metrics
        if metric == "spread_skill_ratio":
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        elif metric in ["eS", "eA"]:
            ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        elif metric == "eL":
            ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Lead Time', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(metric.upper().replace('_', ' '), fontsize=12, fontweight='bold')
        ax.grid(True, color='white', linewidth=1)
        
        # Custom x-ticks with labels
        ax.set_xticks(lead_hours)
        ax.set_xticklabels([hours_to_label(h) for h in lead_hours], fontsize=9, rotation=45)
    
    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # Legend
    if len(model_results) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
                  fontsize=10, frameon=True)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return fig, axes


# =============================================================================
# Usage Examples
# =============================================================================

"""
# First, inspect your dataset to verify the lead structure:
inspect_dataset_leads(ds_corrdiff)

# This will show:
# Lead coordinate dtype: int64
# Lead values: [0 1 2 3 4 5 6 7 8]
# Number of leads: 9
#
# Expected mapping (by position):
#   Position 0: lead coord value = 0, should map to 6 hours (6h)
#   Position 1: lead coord value = 1, should map to 12 hours (12h)
#   ...
#   Position 8: lead coord value = 8, should map to 144 hours (6d)

# Example 1: Compact table (averaged metrics)
model_results = {
    "CorrDiff": ds_corrdiff,
}

latex, df = generate_ensemble_latex_table_compact(
    model_results,
    metrics=["crps", "mse", "csi", "avfss_5", "eL"],
    caption="Super-resolution model comparison",
)


# Example 2: Per-lead table
# positions_to_show uses ARRAY POSITIONS:
#   0 = 6h
#   1 = 12h
#   2 = 18h
#   3 = 24h (1d)
#   4 = 48h (2d)
#   5 = 72h (3d)
#   6 = 96h (4d)
#   7 = 120h (5d)
#   8 = 144h (6d)

# Show 6h, 1d, 3d, 6d:
latex, df = generate_ensemble_latex_table_per_lead(
    model_results,
    metric="crps",
    positions_to_show=[0, 3, 5, 8],  # 6h, 24h, 72h, 144h
    debug=True,  # Print debug info
)

# Show all leads:
latex, df = generate_ensemble_latex_table_per_lead(
    model_results,
    metric="crps",
    positions_to_show=[0, 1, 2, 3, 4, 5, 6, 7, 8],  # All 9 leads
)


# Example 3: Plot metrics
fig, axes = plot_metrics_by_lead_sr(
    model_results,
    metrics=["crps", "csi", "avfss_5", "eL"],
    title="Super-Resolution Metrics by Lead Time",
)
"""