"""
Statistical Significance Analysis: GraphRAG vs NaiveRAG

Compares per-question hallucination_rate values between GraphRAG (baseline)
and NaiveRAG using:
  1. Wilcoxon signed-rank test  (scipy.stats.wilcoxon)
  2. 95 % bootstrap confidence interval on the mean difference
  3. Standardised effect size  (mean diff / pooled std)

Results are printed as a formatted table and saved to
results/significance_analysis.json.

Usage:
    python experiments/significance_analysis.py

Author: GraphRAG Research Team
Date: 2026
"""

import json
import math
import os
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
GRAPHRAG_RESULTS = os.path.join(RESULTS_DIR, "comprehensive_evaluation.json")
NAIVERAG_RESULTS = os.path.join(RESULTS_DIR, "naiverag_evaluation.json")
OUTPUT_PATH = os.path.join(RESULTS_DIR, "significance_analysis.json")


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def significance_analysis(
    graphrag_hallucination_rates: List[float],
    naiverag_hallucination_rates: List[float],
    *,
    n_bootstrap: int = 1_000,
    ci_level: float = 95.0,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a full statistical significance analysis on two paired lists of
    hallucination rates (one per question, one per system).

    Parameters
    ----------
    graphrag_hallucination_rates : list[float]
        Per-question hallucination rates for the GraphRAG system.
    naiverag_hallucination_rates : list[float]
        Per-question hallucination rates for the NaiveRAG system.
    n_bootstrap : int
        Number of bootstrap resamples (default 1 000).
    ci_level : float
        Confidence level for the bootstrap CI (default 95 %).
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        - n_questions           int
        - graphrag_mean         float
        - naiverag_mean         float
        - mean_difference       float   (NaiveRAG − GraphRAG; positive ⇒ GraphRAG is better)
        - wilcoxon_stat         float
        - p_value               float
        - ci_lower              float   (bootstrap 95 % CI of the difference)
        - ci_upper              float
        - effect_size           float   (mean diff / pooled std)
        - interpretation        str
    """
    gr = np.asarray(graphrag_hallucination_rates, dtype=np.float64)
    nr = np.asarray(naiverag_hallucination_rates, dtype=np.float64)

    if len(gr) != len(nr):
        raise ValueError(
            f"Lists must have equal length, got {len(gr)} vs {len(nr)}"
        )

    n = len(gr)
    diff = nr - gr  # positive ⇒ NaiveRAG has higher hallucination (GraphRAG is better)

    # ---- Wilcoxon signed-rank test ----
    # If all differences are zero the test is undefined; handle gracefully.
    if np.all(diff == 0):
        w_stat, p_val = 0.0, 1.0
    else:
        try:
            w_stat, p_val = wilcoxon(diff, alternative="two-sided")
            w_stat = float(w_stat)
            p_val = float(p_val)
        except ValueError:
            # e.g. fewer than 6 non-zero differences → exact test impossible
            w_stat, p_val = float("nan"), float("nan")

    # ---- Bootstrap 95 % CI on the mean difference ----
    rng = np.random.default_rng(random_seed)
    boot_means: np.ndarray = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diff[idx].mean()

    alpha = (100.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 - alpha))

    # ---- Effect size: mean difference / pooled std ----
    mean_diff = float(diff.mean())
    pooled_std = float(np.sqrt((gr.var(ddof=1) + nr.var(ddof=1)) / 2.0))
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # ---- Human-readable interpretation ----
    if p_val <= 0.01:
        sig_label = "highly significant (p ≤ 0.01)"
    elif p_val <= 0.05:
        sig_label = "significant (p ≤ 0.05)"
    elif p_val <= 0.10:
        sig_label = "marginally significant (p ≤ 0.10)"
    else:
        sig_label = "not significant (p > 0.10)"

    abs_es = abs(effect_size)
    if abs_es >= 0.8:
        es_label = "large"
    elif abs_es >= 0.5:
        es_label = "medium"
    elif abs_es >= 0.2:
        es_label = "small"
    else:
        es_label = "negligible"

    direction = (
        "GraphRAG has LOWER hallucination"
        if mean_diff > 0
        else "NaiveRAG has LOWER hallucination"
        if mean_diff < 0
        else "no difference"
    )

    interpretation = (
        f"{direction}; difference is {sig_label} "
        f"with a {es_label} effect size (d = {effect_size:+.4f})."
    )

    return {
        "n_questions": int(n),
        "graphrag_mean": float(gr.mean()),
        "naiverag_mean": float(nr.mean()),
        "mean_difference": mean_diff,
        "wilcoxon_stat": w_stat,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size": effect_size,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_results_table(results: Dict[str, Any]) -> None:
    """Print the significance analysis results as a formatted table."""
    w = 60
    print()
    print("=" * w)
    print("  STATISTICAL SIGNIFICANCE ANALYSIS")
    print("  GraphRAG vs NaiveRAG — Hallucination Rate")
    print("=" * w)
    print()

    rows = [
        ("Number of questions", f"{results['n_questions']}"),
        ("GraphRAG mean hallucination", f"{results['graphrag_mean']:.4f}"),
        ("NaiveRAG mean hallucination", f"{results['naiverag_mean']:.4f}"),
        ("Mean difference (Naive − Graph)", f"{results['mean_difference']:+.4f}"),
        ("", ""),
        ("Wilcoxon signed-rank statistic", f"{results['wilcoxon_stat']:.4f}"),
        ("p-value (two-sided)", f"{results['p_value']:.6f}"),
        ("", ""),
        ("95% Bootstrap CI (lower)", f"{results['ci_lower']:+.4f}"),
        ("95% Bootstrap CI (upper)", f"{results['ci_upper']:+.4f}"),
        ("", ""),
        ("Effect size (d)", f"{results['effect_size']:+.4f}"),
    ]

    for label, value in rows:
        if label == "":
            continue
        print(f"  {label:<36s}  {value:>16s}")

    print()
    print("-" * w)
    print(f"  Interpretation:")
    print(f"    {results['interpretation']}")
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_hallucination_rates(filepath: str, details_key: str) -> List[float]:
    """Extract per-question hallucination_rate values from a result JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get(details_key, [])
    rates = [
        item["metrics"]["hallucination_rate"]
        for item in details
        if "metrics" in item and "hallucination_rate" in item["metrics"]
    ]
    return rates


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading evaluation results…")

    if not os.path.exists(GRAPHRAG_RESULTS):
        print(f"❌ GraphRAG results not found: {GRAPHRAG_RESULTS}")
        sys.exit(1)
    if not os.path.exists(NAIVERAG_RESULTS):
        print(f"❌ NaiveRAG results not found: {NAIVERAG_RESULTS}")
        sys.exit(1)

    graphrag_rates = _load_hallucination_rates(
        GRAPHRAG_RESULTS, "baseline_details"
    )
    naiverag_rates = _load_hallucination_rates(
        NAIVERAG_RESULTS, "naiverag_details"
    )

    print(f"  GraphRAG: {len(graphrag_rates)} per-question hallucination rates")
    print(f"  NaiveRAG: {len(naiverag_rates)} per-question hallucination rates")

    if len(graphrag_rates) != len(naiverag_rates):
        print(
            "⚠️  Unequal question counts — truncating to the shorter list "
            f"(min = {min(len(graphrag_rates), len(naiverag_rates))})"
        )
        n = min(len(graphrag_rates), len(naiverag_rates))
        graphrag_rates = graphrag_rates[:n]
        naiverag_rates = naiverag_rates[:n]

    if not graphrag_rates:
        print("❌ No hallucination rates found. Aborting.")
        sys.exit(1)

    # ---- Run analysis ----
    results = significance_analysis(graphrag_rates, naiverag_rates)

    # ---- Print ----
    print_results_table(results)

    # ---- Save ----
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        # Add the raw per-question values alongside the analysis
        output = {
            "per_question_rates": {
                "graphrag": graphrag_rates,
                "naiverag": naiverag_rates,
            },
            **results,
        }
        json.dump(output, f, indent=4)

    print(f"✅ Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
