# Evaluation Methodology and Results

Evaluation framework for GraphRAG vs NaiveRAG under aligned corpora and question sets.

## Scope

- Corpora: `attention_paper`, `tesla`, `google`, `spacex`
- Systems: GraphRAG and NaiveRAG
- Scale: 200 questions total (50 per corpus)
- Outputs: aggregate JSON, per-question metrics, significance report, plots

## Metric Families

### Retrieval

- Precision
- Recall
- F1

### Answer Quality

- Semantic similarity
- ROUGE (ROUGE-1 proxy in current reporting)
- BERTScore (with compute status)

### Grounding

- Hallucination rate
- Grounded ratio

### Efficiency

- Average response time

## Statistical Testing

`experiments/significance_analysis.py` reports:

- mean difference computed as GraphRAG - NaiveRAG
- Wilcoxon signed-rank test for paired data
- Mann-Whitney U fallback when pairing unavailable
- Cohen's d effect size

Written to:

- `results/significance_analysis.json` -> `per_corpus_significance`
- `results/significance_analysis.json` -> `per_metric_significance`

## Current Aggregate Snapshot

Source files:

- `results/comprehensive_evaluation.json`
- `results/naiverag_evaluation.json`
- `results/significance_analysis.json`

| Metric | GraphRAG | NaiveRAG |
|---|---:|---:|
| Retrieval F1 | 0.1096 | 0.7195 |
| Hallucination Rate | 0.0033 | 0.0204 |
| Semantic Similarity | 0.5881 | 0.8308 |
| BERTScore | 0.8604 | 0.9011 |
| ROUGE-1 Proxy | 0.2588 | 0.4856 |
| Avg Response Time (s) | 6.0047 | 4.0152 |

## Per-Metric Significance Highlights

From `results/significance_analysis.json` (paired mode, Wilcoxon):

| Metric | Mean Diff (G - N) | Test | Statistic | p-value | Cohen's d |
|---|---:|---|---:|---:|---:|
| Hallucination rate | -0.01708 | Wilcoxon | 6.0 | 0.02598 | -0.1498 |
| Semantic similarity | -0.24265 | Wilcoxon | 403.0 | 5.51e-32 | -1.2229 |
| ROUGE score | -0.22681 | Wilcoxon | 878.0 | 6.95e-29 | -1.1521 |
| BERTScore | -0.04069 | Wilcoxon | 806.0 | 1.66e-29 | -1.1543 |

## Critical Comparability Caveat

Retrieval F1 is currently computed using different retrieval-target definitions for GraphRAG and NaiveRAG. Strict inferential comparison on F1 stays disabled until definitions are harmonized.

## Reproducibility Procedure

```bash
python main.py evaluate --experiment comprehensive
python main.py evaluate --experiment naiverag
python main.py evaluate --experiment significance
python experiments/visualize_results.py
```

## Interpretation Guardrails

1. Hallucination improvements should be interpreted with current prompting/config held fixed.
2. Similarity metric gaps reflect both retrieval strategy and generation behavior.
3. Response-time comparisons include external API/network variance.

