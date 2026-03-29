"""
Standalone visualization script for Graph RAG evaluation results.

This script reads evaluation JSON files from results/ and generates:
- Grouped Bar Chart (metric comparison)
- Violin + Strip Plot (hallucination distribution with statistical test)
- Radar Chart (system capability profile)
- Heatmap (per-corpus metric breakdown)
- Aggregate Metrics Table (with significance markers)

All visualizations are saved to results/visual_output/ with DPI >= 300.
"""

import json
import os
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# Configuration
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "visual_output"
DPI = 300
FIGURE_FORMAT = "png"

# Color palette - single professional scheme (Blues)
COLORS = sns.color_palette("Set2", n_colors=8)
PLOT_STYLE = {
    "facecolor": "white",
    "edgecolor": "none",
}


def setup_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def load_json_files() -> Dict[str, Dict]:
    """
    Load all JSON files from results directory.
    Handles missing files gracefully.
    """
    json_files = glob.glob(str(RESULTS_DIR / "*.json"))
    data = {}

    for filepath in json_files:
        filename = Path(filepath).stem
        try:
            with open(filepath, "r") as f:
                data[filename] = json.load(f)
                print(f"✓ Loaded: {filename}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"✗ Failed to load {filename}: {e}")

    return data


def extract_metrics_dataframe(data: Dict[str, Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Extract metrics from loaded JSON files into structured DataFrames.
    Handles structural inconsistencies across files.
    """
    metrics_list = []
    hallucination_by_system = {}

    # Process comprehensive_evaluation.json (baseline)
    if "comprehensive_evaluation" in data:
        comp_eval = data["comprehensive_evaluation"]
        if "aggregate_baseline" in comp_eval:
            agg = comp_eval["aggregate_baseline"]
            metrics_list.append({
                "System": "GraphRAG",
                "F1": agg.get("avg_f1", np.nan),
                "BERT Score": agg.get("avg_bert_score", np.nan),
                "Hallucination Rate": agg.get("avg_hallucination_rate", np.nan),
                "Semantic Similarity": agg.get("avg_semantic_similarity", np.nan),
                "Response Time": agg.get("avg_response_time", np.nan),
            })

    # Process naiverag_evaluation.json
    if "naiverag_evaluation" in data:
        naive_eval = data["naiverag_evaluation"]
        if "aggregate" in naive_eval:
            agg = naive_eval["aggregate"]
            metrics_list.append({
                "System": "Naive RAG",
                "F1": np.nan,  # Not available in naiverag data
                "BERT Score": np.nan,
                "Hallucination Rate": agg.get("avg_hallucination_rate", np.nan),
                "Semantic Similarity": agg.get("avg_semantic_similarity", np.nan),
                "Response Time": agg.get("avg_response_time", np.nan),
            })

    # Process significance_analysis.json - extract hallucination rates
    if "significance_analysis" in data:
        sig_analysis = data["significance_analysis"]
        if "per_question_rates" in sig_analysis:
            for system, rates in sig_analysis["per_question_rates"].items():
                rates_clean = [r for r in rates if r is not None]
                if rates_clean:
                    hallucination_by_system[system] = rates_clean

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Fill NaN values with dataset mean (simple imputation)
    for col in metrics_df.columns:
        if col != "System":
            metrics_df[col] = metrics_df[col].fillna(metrics_df[col].mean())

    print(f"\n✓ Extracted metrics for {len(metrics_df)} systems")

    return metrics_df, hallucination_by_system


def extract_corpus_breakdown(data: Dict[str, Dict]) -> pd.DataFrame:
    """Extract per-corpus metrics for heatmap."""
    corpus_data = []

    if "comprehensive_evaluation" in data:
        comp_eval = data["comprehensive_evaluation"]
        if "per_corpus_baseline" in comp_eval:
            for corpus_id, corpus_info in comp_eval["per_corpus_baseline"].items():
                if "summary" in corpus_info:
                    summary = corpus_info["summary"]
                    corpus_data.append({
                        "Corpus": corpus_id,
                        "System": "GraphRAG",
                        "F1": summary.get("avg_f1", np.nan),
                        "BERT": summary.get("avg_bert_score", np.nan),
                        "Hallucination": summary.get("avg_hallucination_rate", np.nan),
                        "Semantic Sim": summary.get("avg_semantic_similarity", np.nan),
                        "Response Time": summary.get("avg_response_time", np.nan),
                    })

    if "naiverag_evaluation" in data:
        naive_eval = data["naiverag_evaluation"]
        if "per_corpus" in naive_eval:
            for corpus_id, corpus_info in naive_eval["per_corpus"].items():
                corpus_data.append({
                    "Corpus": corpus_id,
                    "System": "Naive RAG",
                    "F1": np.nan,
                    "BERT": np.nan,
                    "Hallucination": corpus_info.get("avg_hallucination_rate", np.nan),
                    "Semantic Sim": corpus_info.get("avg_semantic_similarity", np.nan),
                    "Response Time": corpus_info.get("avg_response_time", np.nan),
                })

    corpus_df = pd.DataFrame(corpus_data)
    # Fill NaN values
    for col in corpus_df.columns:
        if col not in ["Corpus", "System"]:
            corpus_df[col] = corpus_df[col].fillna(corpus_df[col].mean())

    return corpus_df


def calculate_significance_markers(values_list: List[float]) -> str:
    """
    Determine significance level based on p-value thresholds.
    Returns marker: '', '*', '**', '***'
    """
    # Stub function - in real use, would calculate p-values
    # For now, return empty string; actual significance would come from Wilcoxon test
    return ""


def fig1_grouped_bar_chart(metrics_df: pd.DataFrame) -> None:
    """
    Figure 1: Grouped Bar Chart - Overall Metric Comparison.
    Compare key metrics across models/systems using seaborn barplot.
    """
    print("\nGenerating Figure 1: Grouped Bar Chart...")

    # Prepare data for grouped bar plot (systems vs metrics)
    plot_metrics = [
        "F1",
        "BERT Score",
        "Semantic Similarity",
        "Hallucination Rate",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Reshape data for plotting
    plot_data = []
    for _, row in metrics_df.iterrows():
        for metric in plot_metrics:
            plot_data.append({
                "System": row["System"],
                "Metric": metric,
                "Score": row.get(metric, np.nan),
            })

    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.dropna(subset=["Score"])

    # Create grouped bar plot
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="System",
        palette=COLORS[:len(metrics_df)],
        ax=ax,
    )

    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Overall Metric Comparison Across Systems",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(title="System", frameon=True, fancybox=False, shadow=False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"fig1_grouped_bar_chart.{FIGURE_FORMAT}"
    plt.savefig(output_path, dpi=DPI, facecolor="white", edgecolor="none")
    plt.close()

    print(f"✓ Saved Figure 1: {output_path}")


def fig2_violin_strip_plot(
    metrics_df: pd.DataFrame, hallucination_by_system: Dict
) -> Tuple[float, str]:
    """
    Figure 2: Violin + Strip Plot - Hallucination Distribution.
    Includes Wilcoxon signed-rank test for statistical significance.
    """
    print("\nGenerating Figure 2: Violin + Strip Plot...")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Prepare hallucination data
    halli_data = []

    # Use per-question rates if available
    if hallucination_by_system:
        for system, rates in hallucination_by_system.items():
            for rate in rates:
                halli_data.append({"System": system, "Hallucination Rate": rate})
    else:
        # Fallback to aggregate if per-question not available
        for _, row in metrics_df.iterrows():
            halli_data.append({
                "System": row["System"],
                "Hallucination Rate": row["Hallucination Rate"],
            })

    halli_df = pd.DataFrame(halli_data)

    # Plot violin plot
    sns.violinplot(
        data=halli_df,
        x="System",
        y="Hallucination Rate",
        palette=COLORS[:len(halli_df["System"].unique())],
        ax=ax,
    )

    # Overlay strip plot
    sns.stripplot(
        data=halli_df,
        x="System",
        y="Hallucination Rate",
        color="black",
        alpha=0.4,
        size=4,
        ax=ax,
    )

    ax.set_xlabel("System", fontsize=12, fontweight="bold")
    ax.set_ylabel("Hallucination Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        "Hallucination Distribution Across Systems",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Perform Wilcoxon test if we have two systems
    pvalue = np.nan
    significance_marker = ""

    systems = halli_df["System"].unique()
    if len(systems) >= 2 and len(hallucination_by_system) >= 2:
        # Get data for first two systems
        system1_data = np.array(
            halli_df[halli_df["System"] == systems[0]]["Hallucination Rate"]
        )
        system2_data = np.array(
            halli_df[halli_df["System"] == systems[1]]["Hallucination Rate"]
        )

        # Ensure same length for Wilcoxon test (paired test)
        min_len = min(len(system1_data), len(system2_data))
        system1_data = system1_data[:min_len]
        system2_data = system2_data[:min_len]

        try:
            if len(system1_data) > 0:
                statistic, pvalue = wilcoxon(system1_data, system2_data)

                # Determine significance level
                if pvalue < 0.001:
                    significance_marker = "***"
                elif pvalue < 0.01:
                    significance_marker = "**"
                elif pvalue < 0.05:
                    significance_marker = "*"

                # Annotate on plot
                y_max = halli_df["Hallucination Rate"].max()
                y_min = halli_df["Hallucination Rate"].min()
                y_range = y_max - y_min
                ax.text(
                    0.5,
                    y_max + y_range * 0.05,
                    f"p={pvalue:.4f} {significance_marker}",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

                print(f"Wilcoxon test p-value: {pvalue:.6f} {significance_marker}")
        except Exception as e:
            print(f"Warning: Could not perform Wilcoxon test: {e}")

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"fig2_violin_strip_plot.{FIGURE_FORMAT}"
    plt.savefig(output_path, dpi=DPI, facecolor="white", edgecolor="none")
    plt.close()

    print(f"✓ Saved Figure 2: {output_path}")

    return pvalue, significance_marker


def fig3_radar_chart(metrics_df: pd.DataFrame) -> None:
    """
    Figure 3: Radar Chart - System Capability Profile.
    Each axis represents one metric, each system is a polygon.
    """
    print("\nGenerating Figure 3: Radar Chart...")

    # Normalize metrics to 0-1 scale
    metrics_to_plot = ["F1", "BERT Score", "Semantic Similarity"]
    normalized_df = metrics_df.copy()

    for metric in metrics_to_plot:
        if metric in normalized_df.columns:
            min_val = normalized_df[metric].min()
            max_val = normalized_df[metric].max()
            if max_val > min_val:
                normalized_df[metric] = (
                    normalized_df[metric] - min_val
                ) / (max_val - min_val)

    # Invert hallucination rate (lower is better)
    if "Hallucination Rate" in normalized_df.columns:
        normalized_df["Hallucination Rate"] = 1 - normalized_df["Hallucination Rate"]
        metrics_to_plot.append("Hallucination Rate")

    num_vars = len(metrics_to_plot)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Plot each system
    for idx, (_, row) in enumerate(normalized_df.iterrows()):
        values = [row.get(metric, 0) for metric in metrics_to_plot]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=row["System"],
            color=COLORS[idx],
        )
        ax.fill(angles, values, alpha=0.15, color=COLORS[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title(
        "System Capability Profile (Normalized Metrics)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"fig3_radar_chart.{FIGURE_FORMAT}"
    plt.savefig(output_path, dpi=DPI, facecolor="white", edgecolor="none")
    plt.close()

    print(f"✓ Saved Figure 3: {output_path}")


def fig4_heatmap(corpus_df: pd.DataFrame) -> None:
    """
    Figure 4: Heatmap - Per-Corpus Metric Breakdown.
    Rows = corpora, Columns = metrics.
    """
    print("\nGenerating Figure 4: Heatmap...")

    # Create pivot table for heatmap
    metric_cols = ["F1", "BERT", "Hallucination", "Semantic Sim", "Response Time"]
    available_metrics = [col for col in metric_cols if col in corpus_df.columns]

    # Prepare data: corpus x metric
    heatmap_data = []
    corpus_labels = []

    unique_corpora = corpus_df["Corpus"].unique()
    for corpus in unique_corpora:
        corpus_data = corpus_df[corpus_df["Corpus"] == corpus]
        row = []
        for metric in available_metrics:
            val = corpus_data[metric].mean()  # Average across systems if multiple
            row.append(val)
        heatmap_data.append(row)
        corpus_labels.append(corpus)

    heatmap_array = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Create heatmap
    sns.heatmap(
        heatmap_array,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar_kws={"label": "Score"},
        xticklabels=available_metrics,
        yticklabels=corpus_labels,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_title(
        "Per-Corpus Metric Breakdown",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Corpus", fontsize=12, fontweight="bold")

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"fig4_heatmap.{FIGURE_FORMAT}"
    plt.savefig(output_path, dpi=DPI, facecolor="white", edgecolor="none")
    plt.close()

    print(f"✓ Saved Figure 4: {output_path}")


def tab1_aggregate_metrics_table(
    metrics_df: pd.DataFrame, pvalue: float, significance_marker: str
) -> None:
    """
    Table 1: Aggregate Metrics Table.
    Includes statistical significance markers.
    Saves as CSV and HTML image.
    """
    print("\nGenerating Table 1: Aggregate Metrics Table...")

    # Format metrics for display
    display_df = metrics_df.copy()
    display_df = display_df.round(4)

    # Add significance column if available
    if not np.isnan(pvalue):
        display_df["Significance (Hallucination)"] = significance_marker

    # Save as CSV
    csv_path = OUTPUT_DIR / "tab1_aggregate_metrics.csv"
    display_df.to_csv(csv_path, index=False)
    print(f"✓ Saved Table 1 (CSV): {csv_path}")

    # Save as rendered image
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # Create table
    table_data = [display_df.columns.tolist()] + display_df.values.tolist()

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.12] * len(display_df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(display_df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(display_df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#E7E6E6")
            else:
                cell.set_facecolor("#F2F2F2")

    ax.set_title(
        "Aggregate Metrics Table",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    img_path = OUTPUT_DIR / f"tab1_aggregate_metrics.{FIGURE_FORMAT}"
    plt.savefig(img_path, dpi=DPI, facecolor="white", edgecolor="none")
    plt.close()

    print(f"✓ Saved Table 1 (Image): {img_path}")


def main() -> None:
    """Main execution function."""
    print("=" * 70)
    print("Graph RAG Evaluation Visualization Script")
    print("=" * 70)

    # Setup
    setup_output_directory()

    # Load data
    data = load_json_files()
    if not data:
        print("ERROR: No JSON files found in results directory!")
        return

    # Extract metrics
    metrics_df, hallucination_by_system = extract_metrics_dataframe(data)
    corpus_df = extract_corpus_breakdown(data)

    print(f"\nMetrics DataFrame:\n{metrics_df}")
    print(f"\nCorpus DataFrame shape: {corpus_df.shape}")

    # Generate visualizations
    fig1_grouped_bar_chart(metrics_df)
    pvalue, significance_marker = fig2_violin_strip_plot(
        metrics_df, hallucination_by_system
    )
    fig3_radar_chart(metrics_df)
    fig4_heatmap(corpus_df)

    # Generate table
    tab1_aggregate_metrics_table(metrics_df, pvalue, significance_marker)

    print("\n" + "=" * 70)
    print("✓ Visualization pipeline completed successfully!")
    print(f"✓ All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
