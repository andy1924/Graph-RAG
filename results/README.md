# Results Directory

This directory contains all experimental results and evaluation outputs.

## Directory Structure

```
results/
├── evaluation_<timestamp>.json     # Individual evaluation results
├── comprehensive_evaluation.json   # Aggregate evaluation results
├── benchmark_results_<name>.json   # Benchmark-specific results
└── README.md                       # This file
```

## Result File Format

Each evaluation produces a JSON file with the following structure:

```json
{
  "experiment_id": "baseline_20250309_103000",
  "timestamp": "2025-03-09T10:30:00",
  
  "retrieval_metrics": {
    "precision": 0.92,
    "recall": 0.88,
    "f1": 0.90
  },
  
  "answer_quality_metrics": {
    "rouge": 0.45,
    "bert_score": 0.68,
    "bert_score_status": "computed",
    "semantic_similarity": 0.78
  },
  
  "hallucination_metrics": {
    "hallucination_rate": 0.12,
    "grounded_ratio": 0.88,
    "fact_consistency": 0.95
  },
  
  "multimodal_metrics": {
    "text_coverage": 0.70,
    "table_coverage": 0.20,
    "image_coverage": 0.10,
    "overall_coverage": 0.65
  },
  
  "performance_metrics": {
    "avg_response_time": 0.34,
    "throughput": 2.94,  # questions/second
    "memory_usage": 512   # MB
  },
  
  "per_question_results": [
    {
      "question_id": "q1",
      "question": "Sample question",
      "answer": "Generated answer",
      "metrics": {...}
    }
  ]
}
```

For comprehensive runs (`results/comprehensive_evaluation.json`), per-corpus
summaries also include deterministic split metadata:

```json
{
  "summary": {
    "num_questions": 50,
    "train_questions": 40,
    "heldout_questions": 10,
    "heldout_fraction": 0.2,
    "heldout_seed": 42
  }
}
```

## Key Metrics Interpretation

## Cross-System F1 Comparability Note

When comparing GraphRAG and NaiveRAG in aggregate outputs (including
`results/visual_output/tab1_aggregate_metrics.csv` and the rendered table image),
the `F1` values are **not the same underlying metric**:

- GraphRAG `F1`: graph-node retrieval match (retrieved graph node IDs/labels vs expected relevant graph items).
- NaiveRAG `F1`: answer-entity match in retrieved text chunks (`aggregate.avg_retrieval_f1` in `results/naiverag_evaluation.json`).

Use these values for within-system tracking and directional diagnostics, but do not
claim strict apples-to-apples equivalence without harmonizing retrieval metric definitions.

## Multi-Metric Significance Analysis

`results/significance_analysis.json` now includes a `per_metric_significance` section
with inferential statistics for multiple per-question metrics:

- `hallucination_rate`
- `semantic_similarity`
- `rouge_score` (ROUGE-1 proxy in current pipeline)
- `bert_score`
- `retrieval_f1` (explicitly labeled non-comparable unless metric definitions are harmonized)

For each comparable metric, the report includes:

- `mean_difference` computed as **GraphRAG - NaiveRAG**
- rank-test statistics (`wilcoxon_stat` for paired tests, or `mannwhitney_u_stat` fallback)
- `p_value`
- `cohens_d` effect size
- pairing/test metadata (`pairing_mode`, `test_used`)

If paired alignment is unavailable or rank tests are undefined (for example, all paired
differences equal zero), the report stores `null` and a descriptive `note` instead of NaN.

### Retrieval Metrics
- **Precision > 0.9**: Excellent - Few irrelevant results
- **Recall > 0.85**: Excellent - Most relevant results found
- **F1 > 0.85**: Excellent combined retrieval quality

### Answer Quality
- **ROUGE > 0.4**: Good word-overlap with reference
- **BERTScore > 0.75**: Strong semantic equivalence (only when `bert_score_status` is `computed`)
- **Semantic Similarity > 0.75**: Good semantic match

### Hallucination
- **Hallucination Rate < 0.1**: Excellent - Minimal false claims
- **Grounded Ratio > 0.9**: Excellent - Claims well-supported
- **Fact Consistency > 0.9**: Excellent - Factually consistent

## Analysis Scripts

```python
# Load and analyze results
import json
import pandas as pd

# Load single result
with open("results/evaluation_20250309_103000.json") as f:
    results = json.load(f)

# Get summary metrics
print(f"F1 Score: {results['retrieval_metrics']['f1']:.3f}")
print(f"Hallucination Rate: {results['hallucination_metrics']['hallucination_rate']:.3f}")

# Load per-question results
df = pd.DataFrame(results["per_question_results"])
print(df[["question", "metrics"]].head())

# Compare across experiments
experiment_results = []
for filename in sorted(os.listdir("results")):
    if filename.endswith(".json"):
        with open(f"results/{filename}") as f:
            data = json.load(f)
            experiment_results.append({
                "experiment": data["experiment_id"],
                "f1": data["retrieval_metrics"]["f1"],
                "hallucination": data["hallucination_metrics"]["hallucination_rate"]
            })

df_compare = pd.DataFrame(experiment_results)
print(df_compare)
```

## Plotting Results

```python
import matplotlib.pyplot as plt
import json

# Load results
with open("results/evaluation_20250309_103000.json") as f:
    results = json.load(f)

# Plot metrics
metrics = ['precision', 'recall', 'f1']
values = [
    results['retrieval_metrics']['precision'],
    results['retrieval_metrics']['recall'],
    results['retrieval_metrics']['f1']
]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values)
plt.ylim([0, 1])
plt.ylabel('Score')
plt.title('GraphRAG Retrieval Performance')
plt.show()

# Plot hallucination trend
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Hallucination rate
hallucination_rates = [
    results['hallucination_metrics']['hallucination_rate']
    for e in experiments
]
ax1.plot(experiments, hallucination_rates, marker='o')
ax1.set_ylabel('Hallucination Rate')
ax1.set_title('Hallucination Trend')
ax1.grid(True)

# Grounded ratio
grounded_ratios = [
    results['hallucination_metrics']['grounded_ratio']
    for e in experiments
]
ax2.plot(experiments, grounded_ratios, marker='o', color='green')
ax2.set_ylabel('Grounded Ratio')
ax2.set_title('Grounding Improvement')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Latest Results Summary

### Baseline Experiment (2025-03-09)
- **F1 Score**: 0.90
- **Hallucination Rate**: 0.12
- **Semantic Similarity**: 0.78
- **Response Time**: 0.34s

### Multimodal Ablation Study
1. **Text Only**: F1=0.88, Hall=0.14
2. **Text + Tables**: F1=0.91, Hall=0.11
3. **Text + Tables + Images**: F1=0.90, Hall=0.12

**Conclusion**: Adding table support improves retrieval quality, but image addition shows diminishing returns.

For detailed evaluation methodology, see [../docs/EVALUATION.md](../docs/EVALUATION.md)
