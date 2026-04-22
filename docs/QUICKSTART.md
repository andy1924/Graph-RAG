# Quick Start

Minimal path to run full GraphRAG vs NaiveRAG experiment cycle.

## 1) Prerequisites

- Python 3.10+
- Neo4j (Aura or local)
- OpenAI API key

## 2) Install

```bash
git clone https://github.com/andy1924/Graph-RAG
cd Graph-RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Configure Environment

Create `.env` in repository root:

```dotenv
OPENAI_API_KEY=...
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

## 4) Run End-to-End Pipeline

```bash
# Ingest all corpora into GraphRAG + NaiveRAG stores
python main.py ingest --all

# Interactive query (GraphRAG only)
python main.py query --mode graphrag

# Comparative query (GraphRAG + NaiveRAG)
python main.py query --mode both

# Evaluation suite
python main.py evaluate --experiment comprehensive
python main.py evaluate --experiment naiverag
python main.py evaluate --experiment significance

# Visual outputs
python experiments/visualize_results.py
```

## 5) Expected Outputs

- `results/comprehensive_evaluation.json`
- `results/naiverag_evaluation.json`
- `results/significance_analysis.json`
- `results/visual_output/fig1_grouped_bar_chart.png`
- `results/visual_output/fig2_violin_strip_plot.png`
- `results/visual_output/fig3_radar_chart.png`
- `results/visual_output/fig4_heatmap.png`
- `results/visual_output/tab1_aggregate_metrics.csv`
- `results/visual_output/tab1_aggregate_metrics.png`

## 6) Fast Troubleshooting

- Neo4j connection fail: recheck `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.
- Missing package errors: activate virtual env, rerun `pip install -r requirements.txt`.
- Empty retrieval output: run ingestion again for target corpus.

## 7) Common Variants

```bash
# Single corpus
python main.py ingest --corpus tesla

# Ingest only one pipeline
python main.py ingest --corpus google --target naiverag
python main.py ingest --corpus spacex --target graphrag

# Run all evaluation scripts in sequence
python main.py evaluate --all
```

