"""
LaTeX table generators for bias-correction evaluation.

Contains:
  - generate_latex_tables
  - generate_compact_latex_table
  - generate_per_lead_latex_table
"""

def generate_latex_tables(
    baselines: dict,
    multi_lead_models: dict,
    single_lead_models: dict,
    split: str = "val",
    n_leads: int = 24,
    lead_time_hours: int = 6,
    metrics: list[str] | None = None,
    leads_to_show: list[int] | None = None,
    show_average: bool = True,
    precision: int = 3,
    highlight_best: bool = True,
    save_path: str | None = None,
    caption: str = "Model comparison metrics",
    label: str = "tab:metrics",
):
    """
    Generate LaTeX tables comparing baselines and models.

    Args:
        baselines: dict of {name: per_lead_dict}
        multi_lead_models: dict of {name: results_dict}
        single_lead_models: dict of {name: (lead_idx, results_dict)}
        split: "train" or "val"
        n_leads: number of lead times
        lead_time_hours: hours per lead time step
        metrics: list of metrics to include (default: MSE, MAE, CSI, FSS_5, SAL_L)
        leads_to_show: specific lead indices to show (default: [0, 5, 11, 17, 23])
        show_average: whether to include average across all leads
        precision: decimal places for rounding
        highlight_best: whether to bold the best value in each column
        save_path: if provided, save LaTeX to this file
        caption: table caption
        label: table label for referencing

    Returns:
        latex_str: LaTeX table string
        df: pandas DataFrame with the data
    """

    if metrics is None:
        metrics = ["MSE_all", "MAE_all", "CSI", "FSS_5", "SAL_L"]

    if leads_to_show is None:
        # Default: 6h, 36h, 3d, 4.5d, 6d
        leads_to_show = [0, 5, 11, 17, 23]

    # Filter to valid leads
    leads_to_show = [l for l in leads_to_show if l < n_leads]

    # Build list of all models
    all_models = {}

    # Add baselines
    for name, per_lead in baselines.items():
        all_models[name] = {"type": "baseline", "data": per_lead}

    # Add multi-lead models
    for name, results in multi_lead_models.items():
        all_models[name] = {"type": "multi", "data": results[split]["per_lead"]}

    # Add single-lead models (with their specific lead)
    for name, (lead_idx, results) in single_lead_models.items():
        all_models[name] = {
            "type": "single",
            "lead_idx": lead_idx,
            "data": results[split]["global"],
        }

    # ==========================================================================
    # Table 1: Metrics at specific lead times
    # ==========================================================================

    def get_metric_value(model_info, metric, lead_idx):
        """Extract metric value for a given model and lead."""
        if model_info["type"] == "single":
            # Single-lead model: only has value at its specific lead
            if lead_idx == model_info["lead_idx"]:
                return model_info["data"].get(metric, np.nan)
            else:
                return np.nan
        else:
            # Baseline or multi-lead: has all leads
            return model_info["data"].get(lead_idx, {}).get(metric, np.nan)

    def get_average_metric(model_info, metric, n_leads):
        """Compute average metric across all leads."""
        if model_info["type"] == "single":
            # Single-lead: just return the single value
            return model_info["data"].get(metric, np.nan)
        else:
            values = [
                model_info["data"].get(i, {}).get(metric, np.nan)
                for i in range(n_leads)
            ]
            valid = [v for v in values if not np.isnan(v)]
            return np.mean(valid) if valid else np.nan

    # Build data for per-lead table
    rows = []
    for model_name, model_info in all_models.items():
        row = {"Model": model_name}
        for lead_idx in leads_to_show:
            lead_hours = (lead_idx + 1) * lead_time_hours
            for metric in metrics:
                col_name = f"{metric}@{lead_hours}h"
                row[col_name] = get_metric_value(model_info, metric, lead_idx)
        rows.append(row)

    df_per_lead = pd.DataFrame(rows)

    # Build data for average table
    rows_avg = []
    for model_name, model_info in all_models.items():
        row = {"Model": model_name}
        for metric in metrics:
            row[metric] = get_average_metric(model_info, metric, n_leads)
        rows_avg.append(row)

    df_avg = pd.DataFrame(rows_avg)

    # ==========================================================================
    # Generate LaTeX
    # ==========================================================================

    def format_value(val, precision, is_best=False):
        """Format a value for LaTeX."""
        if np.isnan(val):
            return "—"
        formatted = f"{val:.{precision}f}"
        if is_best:
            return f"\\textbf{{{formatted}}}"
        return formatted

    def find_best(df, col, higher_is_better):
        """Find the best value in a column."""
        valid = df[col].dropna()
        if valid.empty:
            return None
        if higher_is_better:
            return valid.max()
        return valid.min()

    # Define which metrics are "higher is better"
    higher_is_better = {
        "CSI": True,
        "F1": True,
        "precision": True,
        "recall_POD": True,
        "FSS_1": True,
        "FSS_5": True,
        "FSS_17": True,
    }
    # All others (MSE, MAE, SAL_L, FAR, etc.) are "lower is better"

    def df_to_latex(df, caption, label, precision, highlight_best, metric_cols):
        """Convert DataFrame to LaTeX table."""
        lines = []

        # Preamble
        n_cols = len(df.columns)
        col_format = "l" + "c" * (n_cols - 1)

        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{" + caption + "}")
        lines.append("\\label{" + label + "}")
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.append("\\begin{tabular}{" + col_format + "}")
        lines.append("\\toprule")

        # Header
        headers = list(df.columns)
        # Clean up header names for LaTeX
        clean_headers = []
        for h in headers:
            h_clean = h.replace("_", "\\_").replace("@", " @ ")
            clean_headers.append(h_clean)
        lines.append(" & ".join(clean_headers) + " \\\\")
        lines.append("\\midrule")

        # Find best values for each column
        best_values = {}
        if highlight_best:
            for col in df.columns:
                if col == "Model":
                    continue
                # Determine if higher or lower is better
                base_metric = col.split("@")[0] if "@" in col else col
                hib = higher_is_better.get(base_metric, False)
                best_values[col] = find_best(df, col, hib)

        # Data rows
        for _, row in df.iterrows():
            row_strs = []
            for col in df.columns:
                if col == "Model":
                    # Escape underscores in model names
                    model_name = str(row[col]).replace("_", "\\_")
                    row_strs.append(model_name)
                else:
                    val = row[col]
                    is_best = (
                        highlight_best
                        and best_values.get(col) is not None
                        and val == best_values[col]
                    )
                    row_strs.append(format_value(val, precision, is_best))
            lines.append(" & ".join(row_strs) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}%")
        lines.append("}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    # Generate LaTeX for both tables
    latex_parts = []

    # Per-lead table
    metric_cols_per_lead = [c for c in df_per_lead.columns if c != "Model"]
    latex_per_lead = df_to_latex(
        df_per_lead,
        caption=f"{caption} at specific lead times",
        label=f"{label}_per_lead",
        precision=precision,
        highlight_best=highlight_best,
        metric_cols=metric_cols_per_lead,
    )
    latex_parts.append("% === Per-Lead Metrics ===")
    latex_parts.append(latex_per_lead)

    # Average table
    if show_average:
        latex_avg = df_to_latex(
            df_avg,
            caption=f"{caption} (averaged across all leads)",
            label=f"{label}_avg",
            precision=precision,
            highlight_best=highlight_best,
            metric_cols=metrics,
        )
        latex_parts.append("\n% === Average Metrics ===")
        latex_parts.append(latex_avg)

    latex_str = "\n\n".join(latex_parts)

    # Save if requested
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"LaTeX saved to: {save_path}")

    # Print
    print(latex_str)

    return latex_str, df_per_lead, df_avg


def generate_compact_latex_table(
    baselines: dict,
    multi_lead_models: dict,
    single_lead_models: dict,
    split: str = "val",
    n_leads: int = 24,
    metrics: list[str] | None = None,
    precision: int = 3,
    highlight_best: bool = True,
    save_path: str | None = None,
    caption: str = "Model comparison (average metrics)",
    label: str = "tab:metrics_compact",
    include_model_type: bool = True,
):
    """
    Generate a compact LaTeX table with just average metrics.

    This is a simpler version showing one row per model with averaged metrics.
    """

    if metrics is None:
        metrics = [
            "MSE_all",
            "MAE_all",
            "CSI",
            "FAR",
            "FSS_1",
            "FSS_5",
            "SAL_S",
            "SAL_A",
            "SAL_L",
        ]

    # Build rows
    rows = []

    # Baselines
    for name, per_lead in baselines.items():
        row = {"Model": name}
        if include_model_type:
            row["Type"] = "Baseline"
        for metric in metrics:
            values = [per_lead.get(i, {}).get(metric, np.nan) for i in range(n_leads)]
            valid = [v for v in values if not np.isnan(v)]
            row[metric] = np.mean(valid) if valid else np.nan
        rows.append(row)

    # Multi-lead models
    for name, results in multi_lead_models.items():
        per_lead = results[split]["per_lead"]
        row = {"Model": name}
        if include_model_type:
            row["Type"] = "Multi-Lead"
        for metric in metrics:
            values = [per_lead.get(i, {}).get(metric, np.nan) for i in range(n_leads)]
            valid = [v for v in values if not np.isnan(v)]
            row[metric] = np.mean(valid) if valid else np.nan
        rows.append(row)

    # Single-lead models
    for name, (lead_idx, results) in single_lead_models.items():
        row = {"Model": name}
        if include_model_type:
            row["Type"] = f"Single ({(lead_idx + 1) * 6}h)"
        for metric in metrics:
            row[metric] = results[split]["global"].get(metric, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Higher is better for these metrics
    higher_is_better = {
        "CSI": True,
        "F1": True,
        "precision": True,
        "recall_POD": True,
        "FSS_1": True,
        "FSS_5": True,
        "FSS_17": True,
    }

    def format_val(val, prec, is_best=False):
        if np.isnan(val):
            return "—"
        s = f"{val:.{prec}f}"
        return f"\\textbf{{{s}}}" if is_best else s

    # Find best for each metric
    best = {}
    if highlight_best:
        for m in metrics:
            valid = df[m].dropna()
            if not valid.empty:
                best[m] = valid.max() if higher_is_better.get(m, False) else valid.min()

    # Build LaTeX
    cols = list(df.columns)
    col_format = "l" * (2 if include_model_type else 1) + "c" * len(metrics)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule",
    ]

    # Header
    header_clean = [c.replace("_", "\\_") for c in cols]
    lines.append(" & ".join(header_clean) + " \\\\")
    lines.append("\\midrule")

    # Group by type if included
    if include_model_type:
        current_type = None
        for _, row in df.iterrows():
            if row["Type"] != current_type:
                if current_type is not None:
                    lines.append("\\midrule")
                current_type = row["Type"]

            row_strs = []
            for col in cols:
                if col in ["Model", "Type"]:
                    row_strs.append(str(row[col]).replace("_", "\\_"))
                else:
                    is_best = (
                        highlight_best
                        and best.get(col) is not None
                        and row[col] == best[col]
                    )
                    row_strs.append(format_val(row[col], precision, is_best))
            lines.append(" & ".join(row_strs) + " \\\\")
    else:
        for _, row in df.iterrows():
            row_strs = []
            for col in cols:
                if col == "Model":
                    row_strs.append(str(row[col]).replace("_", "\\_"))
                else:
                    is_best = (
                        highlight_best
                        and best.get(col) is not None
                        and row[col] == best[col]
                    )
                    row_strs.append(format_val(row[col], precision, is_best))
            lines.append(" & ".join(row_strs) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )

    latex_str = "\n".join(lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"Saved: {save_path}")

    print(latex_str)

    return latex_str, df


def generate_per_lead_latex_table(
    baselines: dict,
    multi_lead_models: dict,
    single_lead_models: dict,
    metric: str = "MSE_all",
    split: str = "val",
    n_leads: int = 24,
    lead_time_hours: int = 6,
    leads_to_show: list[int] | None = None,
    precision: int = 3,
    highlight_best: bool = True,
    save_path: str | None = None,
    caption: str | None = None,
    label: str | None = None,
):
    """
    Generate a LaTeX table for a SINGLE metric across lead times.

    Columns: Model | Lead1 | Lead2 | ... | Avg
    """

    if leads_to_show is None:
        leads_to_show = [0, 5, 11, 17, 23]  # 6h, 36h, 3d, 4.5d, 6d

    leads_to_show = [l for l in leads_to_show if l < n_leads]

    if caption is None:
        caption = f"{metric.replace('_', ' ')} by lead time"
    if label is None:
        label = f"tab:{metric.lower()}"

    # Higher is better
    higher_is_better_metrics = {
        "CSI",
        "F1",
        "precision",
        "recall_POD",
        "FSS_1",
        "FSS_5",
        "FSS_17",
    }
    hib = metric in higher_is_better_metrics

    rows = []

    # Baselines
    for name, per_lead in baselines.items():
        row = {"Model": name}
        all_vals = []
        for li in leads_to_show:
            h = (li + 1) * lead_time_hours
            val = per_lead.get(li, {}).get(metric, np.nan)
            row[f"{h}h"] = val
            if not np.isnan(val):
                all_vals.append(val)
        # Average across ALL leads (not just shown)
        full_vals = [per_lead.get(i, {}).get(metric, np.nan) for i in range(n_leads)]
        valid_full = [v for v in full_vals if not np.isnan(v)]
        row["Avg"] = np.mean(valid_full) if valid_full else np.nan
        rows.append(row)

    # Multi-lead
    for name, results in multi_lead_models.items():
        per_lead = results[split]["per_lead"]
        row = {"Model": name}
        for li in leads_to_show:
            h = (li + 1) * lead_time_hours
            row[f"{h}h"] = per_lead.get(li, {}).get(metric, np.nan)
        full_vals = [per_lead.get(i, {}).get(metric, np.nan) for i in range(n_leads)]
        valid_full = [v for v in full_vals if not np.isnan(v)]
        row["Avg"] = np.mean(valid_full) if valid_full else np.nan
        rows.append(row)

    # Single-lead
    for name, (lead_idx, results) in single_lead_models.items():
        row = {"Model": name}
        for li in leads_to_show:
            h = (li + 1) * lead_time_hours
            if li == lead_idx:
                row[f"{h}h"] = results[split]["global"].get(metric, np.nan)
            else:
                row[f"{h}h"] = np.nan
        row["Avg"] = results[split]["global"].get(metric, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Find best per column
    best = {}
    if highlight_best:
        for col in df.columns:
            if col == "Model":
                continue
            valid = df[col].dropna()
            if not valid.empty:
                best[col] = valid.max() if hib else valid.min()

    def fmt(val, is_best=False):
        if np.isnan(val):
            return "—"
        s = f"{val:.{precision}f}"
        return f"\\textbf{{{s}}}" if is_best else s

    # LaTeX
    cols = list(df.columns)
    col_fmt = "l" + "c" * (len(cols) - 1)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        " & ".join(cols) + " \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        row_strs = []
        for col in cols:
            if col == "Model":
                row_strs.append(str(row[col]).replace("_", "\\_"))
            else:
                is_best = (
                    highlight_best
                    and best.get(col) is not None
                    and row[col] == best[col]
                )
                row_strs.append(fmt(row[col], is_best))
        lines.append(" & ".join(row_strs) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    latex_str = "\n".join(lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"Saved: {save_path}")

    print(latex_str)

    return latex_str, df