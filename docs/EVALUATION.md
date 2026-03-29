# Evaluation Methodology and Results

## Project Metadata
- Project Title: Beyond Vector Search: Mitigating LLM Hallucinations via Graph-Based Retrieval-Augmented Generation (GraphRAG)
- Authors: Arnav Deshpande; Sarvesh Nimbalkar; Dhruv Gadia; Aadi Rawat
- Organization: Mukesh Patel School of Technology and Management, NMIMS University
- Contact Email: [deshpandearnavn@gmail.com](mailto:deshpandearnavn@gmail.com)
- GitHub Repository: https://github.com/andy1924/Graph-RAG

## Evaluation Scope
The current evaluation compares GraphRAG and NaiveRAG across four corpora:
- attention_paper
- tesla
- google
- spacex

Each system is evaluated on 200 questions total (50 per corpus).

## Metric Definitions
### Retrieval metrics
- Precision, recall, and F1 are computed per question from retrieved items and reference-relevant items.
- GraphRAG and NaiveRAG currently use different retrieval-target definitions; therefore retrieval F1 is not directly comparable across systems.

### Answer-quality metrics
- Semantic similarity
- ROUGE score (ROUGE-1 proxy in current reporting)
- BERTScore (with execution status tracking)

### Grounding metrics
- Hallucination rate
- Grounded ratio

### Efficiency
- Average response time per question

## Statistical Testing
The script experiments/significance_analysis.py reports:
- mean difference as GraphRAG - NaiveRAG
- Wilcoxon signed-rank test when paired alignment is available
- Mann-Whitney U fallback when pairing is unavailable
- Cohen's d effect size

Results are written to results/significance_analysis.json under:
- per_corpus_significance
- per_metric_significance

## Current Aggregate Results
Source files:
- results/comprehensive_evaluation.json
- results/naiverag_evaluation.json
- results/significance_analysis.json

| Metric | GraphRAG | NaiveRAG |
|---|---:|---:|
| Retrieval F1 | 0.1096 | 0.7195 |
| Hallucination Rate | 0.0033 | 0.0204 |
| Semantic Similarity | 0.5881 | 0.8308 |
| BERTScore | 0.8604 | 0.9011 |
| ROUGE-1 proxy | 0.2588 | 0.4856 |
| Avg Response Time (s) | 6.0047 | 4.0152 |

Comparability note:
- Retrieval F1 values are computed from different retrieval definitions; inferential F1 comparison is intentionally disabled in per_metric_significance.

## Per-Metric Significance Summary
From results/significance_analysis.json (paired_by_key, Wilcoxon):

| Metric | Mean Diff (G - N) | Test | Statistic | p-value | Cohen's d |
|---|---:|---|---:|---:|---:|
| Hallucination rate | -0.01708 | Wilcoxon | 6.0 | 0.02598 | -0.1498 |
| Semantic similarity | -0.24265 | Wilcoxon | 403.0 | 5.51e-32 | -1.2229 |
| ROUGE score | -0.22681 | Wilcoxon | 878.0 | 6.95e-29 | -1.1521 |
| BERTScore | -0.04069 | Wilcoxon | 806.0 | 1.66e-29 | -1.1543 |

Interpretation should be constrained to the current task setup and metric definitions.

## Reproducibility Procedure
```bash
python main.py evaluate --experiment comprehensive
python main.py evaluate --experiment naiverag
python main.py evaluate --experiment significance
python experiments/visualize_results.py
```

## Limitations
- Retrieval F1 cross-system inferential comparison remains invalid until retrieval targets are harmonized.
- Hallucination and quality metrics depend on the current prompt and model configuration.
- Response-time measurements include external API latency and are environment-dependent.

