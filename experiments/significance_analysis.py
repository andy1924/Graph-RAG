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


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Infinity values with None for strict JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    return obj


def _compute_wilcoxon_safe(
    graphrag_rates: List[float],
    naiverag_rates: List[float],
    *,
    scope_label: str,
) -> Dict[str, Any]:
    """
    Compute Wilcoxon robustly and return JSON-safe values.

    Returns keys:
      - wilcoxon_stat: float | None
      - p_value: float | None
      - note: optional explanation when undefined/degenerate
    """
    differences = [g - n for g, n in zip(graphrag_rates, naiverag_rates)]

    # Mathematically undefined when all paired differences are exactly zero.
    if differences and all(d == 0 for d in differences):
        return {
            "wilcoxon_stat": None,
            "p_value": None,
            "note": "All differences are zero; test is undefined.",
        }

    if graphrag_rates and all(v == 0 for v in graphrag_rates):
        print(
            f"⚠️  WARNING [{scope_label}]: GraphRAG hallucination rates are all 0.0 "
            "(degenerate distribution). This can indicate upstream hallucination "
            "detector saturation/bug and reduces statistical sensitivity."
        )
    if naiverag_rates and all(v == 0 for v in naiverag_rates):
        print(
            f"⚠️  WARNING [{scope_label}]: NaiveRAG hallucination rates are all 0.0 "
            "(degenerate distribution). This can indicate upstream hallucination "
            "detector saturation/bug and reduces statistical sensitivity."
        )

    try:
        stat, p_val = wilcoxon(differences)
        return {"wilcoxon_stat": float(stat), "p_value": float(p_val)}
    except Exception as e:
        return {
            "wilcoxon_stat": None,
            "p_value": None,
            "note": f"Wilcoxon could not be computed: {e}",
        }


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

    wilcoxon_result = _compute_wilcoxon_safe(
        graphrag_hallucination_rates,
        naiverag_hallucination_rates,
        scope_label="overall",
    )
    w_stat = wilcoxon_result["wilcoxon_stat"]
    p_val = wilcoxon_result["p_value"]

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
    if p_val is None:
        sig_label = "undefined (Wilcoxon not computable)"
    elif p_val <= 0.01:
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

    if "note" in wilcoxon_result:
        interpretation = f"{interpretation} Note: {wilcoxon_result['note']}"

    result = {
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

    if "note" in wilcoxon_result:
        result["note"] = wilcoxon_result["note"]

    return result


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_results_table(results: Dict[str, Any]) -> None:
    """Print the significance analysis results as a formatted table."""
    def _fmt_optional(value: Any, fmt: str) -> str:
        return "null" if value is None else format(value, fmt)

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
        ("Wilcoxon signed-rank statistic", _fmt_optional(results.get("wilcoxon_stat"), ".4f")),
        ("p-value (two-sided)", _fmt_optional(results.get("p_value"), ".6f")),
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
    """Extract per-question hallucination_rate values from a result JSON.
    
    Supports two structures:
      1. Flat: data["details"] = [{...}, ...]
      2. Nested: data["per_corpus_baseline"][corpus_id]["details"] = [{...}, ...]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    rates = []

    # 1) Flat "details" list (new naiverag format)
    if "details" in data and isinstance(data["details"], list):
        for item in data["details"]:
            if "metrics" in item and "hallucination_rate" in item["metrics"]:
                rates.append(item["metrics"]["hallucination_rate"])
        return rates

    # 2) Nested per_corpus structure (graphrag format)
    for key in ("per_corpus_baseline", "per_corpus_naiverag"):
        if key in data:
            for corpus_id, corpus_data in data[key].items():
                details = corpus_data.get("details", [])
                for item in details:
                    if "metrics" in item and "hallucination_rate" in item["metrics"]:
                        rates.append(item["metrics"]["hallucination_rate"])
            if rates:
                return rates

    # 3) Fallback: try the explicit details_key
    details = data.get(details_key, [])
    for item in details:
        if "metrics" in item and "hallucination_rate" in item["metrics"]:
            rates.append(item["metrics"]["hallucination_rate"])

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

    corpus_names = ["attention_paper", "tesla", "google", "spacex"]
    n_per_corpus = 15

    per_corpus_significance = {}
    for i, corpus_name in enumerate(corpus_names):
        start = i * n_per_corpus
        end = start + n_per_corpus
        g_rates = graphrag_rates[start:end]
        n_rates = naiverag_rates[start:end]

        if not g_rates or not n_rates:
            per_corpus_significance[corpus_name] = {
                "graphrag_mean_hallucination": None,
                "naiverag_mean_hallucination": None,
                "difference": None,
                "wilcoxon_stat": None,
                "p_value": None,
                "note": "Insufficient per-corpus data for significance test.",
                "graphrag_wins": False,
            }
            continue

        g_mean = sum(g_rates) / len(g_rates)
        n_mean = sum(n_rates) / len(n_rates)

        wilcoxon_result = _compute_wilcoxon_safe(
            g_rates,
            n_rates,
            scope_label=f"per-corpus:{corpus_name}",
        )

        per_corpus_significance[corpus_name] = {
            "graphrag_mean_hallucination": round(g_mean, 4),
            "naiverag_mean_hallucination": round(n_mean, 4),
            "difference": round(g_mean - n_mean, 4),
            "wilcoxon_stat": wilcoxon_result["wilcoxon_stat"],
            "p_value": wilcoxon_result["p_value"],
            **({"note": wilcoxon_result["note"]} if "note" in wilcoxon_result else {}),
            "graphrag_wins": g_mean < n_mean,
        }

    results["per_corpus_significance"] = per_corpus_significance

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
        output = sanitize_for_json(output)
        json.dump(output, f, indent=4, allow_nan=False)

    print(f"✅ Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
