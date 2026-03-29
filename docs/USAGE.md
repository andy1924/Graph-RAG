# Usage Guide

## Project Metadata
- Project Title: Beyond Vector Search: Mitigating LLM Hallucinations via Graph-Based Retrieval-Augmented Generation (GraphRAG)
- Authors: Arnav Deshpande; Sarvesh Nimbalkar; Dhruv Gadia; Aadi Rawat
- Organization: Mukesh Patel School of Technology and Management, NMIMS University
- Contact Email: [deshpandearnavn@gmail.com](mailto:deshpandearnavn@gmail.com)
- GitHub Repository: https://github.com/andy1924/Graph-RAG

## Command-Line Interface
The root launcher delegates to scripts/:

```bash
python main.py ingest --help
python main.py query --help
python main.py evaluate --help
```

## Ingestion
```bash
# All corpora to both systems
python main.py ingest --all

# Single corpus to GraphRAG only
python main.py ingest --corpus tesla --target graphrag

# Single corpus to NaiveRAG only
python main.py ingest --corpus google --target naiverag
```

## Querying
```bash
# GraphRAG only
python main.py query --mode graphrag

# NaiveRAG only
python main.py query --mode naiverag

# Side-by-side output
python main.py query --mode both
```

## Evaluation
```bash
# GraphRAG comprehensive evaluation
python main.py evaluate --experiment comprehensive

# NaiveRAG baseline evaluation
python main.py evaluate --experiment naiverag

# Significance testing
python main.py evaluate --experiment significance

# Full evaluation suite
python main.py evaluate --all
```

## Visualization
```bash
python experiments/visualize_results.py
```
Generated outputs:
- results/visual_output/fig1_grouped_bar_chart.png
- results/visual_output/fig2_violin_strip_plot.png
- results/visual_output/fig3_radar_chart.png
- results/visual_output/fig4_heatmap.png
- results/visual_output/tab1_aggregate_metrics.csv
- results/visual_output/tab1_aggregate_metrics.png

## Python API Examples
### GraphRAG retrieval
```python
from graphrag.retrieval import GraphRetriever

retriever = GraphRetriever()
answer, metadata = retriever.answer_question("What is the Transformer model?")
print(answer)
print(metadata.get("retrieved_nodes", []))
```

### Multimodal retrieval
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

### Single-instance evaluation
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
- Run ingestion before query/evaluation if graph and baseline stores are empty.
- Keep OpenAI and Neo4j credentials in .env only.
- Re-run significance analysis after updating either evaluation JSON file.

## Troubleshooting
- Neo4j connection errors: verify NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE.
- Missing Python packages: install with pip install -r requirements.txt in the active environment.
- Empty retrieval output: check ingestion completeness and corpus selection.

