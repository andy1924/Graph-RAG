# Usage Guide

Operational guide for ingestion, querying, evaluation, and Python API use.

## CLI Surface

Unified launcher routes to script modules:

```bash
python main.py ingest --help
python main.py query --help
python main.py evaluate --help
```

## Ingestion

```bash
# All corpora into both systems (default target=both)
python main.py ingest --all

# Single corpus into GraphRAG only
python main.py ingest --corpus tesla --target graphrag

# Single corpus into NaiveRAG only
python main.py ingest --corpus google --target naiverag
```

Available corpora in current scripts:

- `attention_paper`
- `tesla`
- `google`
- `spacex`

## Query Modes

```bash
# GraphRAG only
python main.py query --mode graphrag

# NaiveRAG only
python main.py query --mode naiverag

# Side-by-side comparison
python main.py query --mode both
```

## Evaluation Modes

```bash
# GraphRAG comprehensive run
python main.py evaluate --experiment comprehensive

# NaiveRAG baseline run
python main.py evaluate --experiment naiverag

# Significance analysis
python main.py evaluate --experiment significance

# Full suite
python main.py evaluate --all
```

## Visualization

```bash
python experiments/visualize_results.py
```

Generated files:

- `results/visual_output/fig1_grouped_bar_chart.png`
- `results/visual_output/fig2_violin_strip_plot.png`
- `results/visual_output/fig3_radar_chart.png`
- `results/visual_output/fig4_heatmap.png`
- `results/visual_output/tab1_aggregate_metrics.csv`
- `results/visual_output/tab1_aggregate_metrics.png`

## Python API Examples

Graph retrieval:

```python
from graphrag.retrieval import GraphRetriever

retriever = GraphRetriever()
answer, metadata = retriever.answer_question("What is the Transformer model?")
print(answer)
print(metadata.get("retrieved_nodes", []))
```

Multimodal retrieval:

```python
from graphrag.retrieval import MultimodalGraphRetriever

retriever = MultimodalGraphRetriever()
answer, metadata = retriever.answer_with_multimodal_context(
    "How does masking work in decoder self-attention?",
    include_modalities=["text", "table", "image"],
)
print(answer)
print(metadata.get("context", ""))
```

Single evaluation call:

```python
from graphrag.evaluation import EvaluationPipeline

pipeline = EvaluationPipeline("example_eval")
metrics = pipeline.evaluate(
    question="Example question",
    generated_answer="Model answer",
    reference_answer="Reference answer",
    retrieved_context="Retrieved context",
    retrieved_items=["node_a", "node_b"],
    relevant_items=["node_a"],
)
print(metrics.retrieval_f1, metrics.hallucination_rate)
```

## Operational Notes

- Run ingestion before query/evaluation on new environment.
- Keep secrets in `.env`, never hardcode in scripts.
- Recompute significance after changing either evaluation JSON file.

## Troubleshooting

- Neo4j auth or routing errors: verify `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.
- Import/package errors: activate env, run `pip install -r requirements.txt`.
- Empty answers or thin context: re-run ingestion and confirm target corpus.

