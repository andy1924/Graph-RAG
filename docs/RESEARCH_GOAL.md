# Research Goal and Current Status

## Core Goal

Test whether graph-structured retrieval reduces LLM hallucinations compared to chunk-based retrieval under matched evaluation conditions.

## Working Hypothesis

Entity-relation retrieval from multimodal property graph improves grounding fidelity, especially when evidence is distributed across document sections/modalities.

## Experimental Objectives

Measure GraphRAG vs NaiveRAG on:

- grounding behavior (`hallucination_rate`, `grounded_ratio`)
- answer quality (`semantic_similarity`, `rouge_score`, `bert_score`)
- retrieval behavior (`precision`, `recall`, `retrieval_f1`)
- efficiency (`avg_response_time`)

## Current Evidence (Repository Outputs)

From `results/comprehensive_evaluation.json`, `results/naiverag_evaluation.json`, and `results/significance_analysis.json`:

- GraphRAG shows lower aggregate hallucination rate (0.0033) than NaiveRAG (0.0204).
- NaiveRAG shows higher semantic and lexical similarity in current setup.
- Significance analysis reports statistically significant differences on hallucination, semantic similarity, ROUGE proxy, and BERTScore.

## Methodological Caveat

`retrieval_f1` is not yet cross-system comparable due to differing retrieval target definitions. Inferential F1 comparison remains intentionally disabled.

## Reproducibility Plan

```bash
python main.py ingest --all
python main.py evaluate --experiment comprehensive
python main.py evaluate --experiment naiverag
python main.py evaluate --experiment significance
python experiments/visualize_results.py
```

## Open Risks

1. Retrieval metric non-equivalence limits strict retrieval inference.
2. API/model behavior can shift quality and latency distributions.
3. Corpus/question composition can influence aggregate directionality.

## Next Method Step

Unify retrieval target definitions across GraphRAG and NaiveRAG, then rerun per-metric significance including retrieval F1.

