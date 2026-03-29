# Quick Start

## Project Metadata
- Project Title: Beyond Vector Search: Mitigating LLM Hallucinations via Graph-Based Retrieval-Augmented Generation (GraphRAG)
- Authors: Arnav Deshpande; Sarvesh Nimbalkar; Dhruv Gadia; Aadi Rawat
- Organization: Mukesh Patel School of Technology and Management, NMIMS University
- Contact Email: [deshpandearnavn@gmail.com](mailto:deshpandearnavn@gmail.com)
- GitHub Repository: https://github.com/andy1924/Graph-RAG

## Prerequisites
- Python 3.10+
- Neo4j database (Aura or local)
- OpenAI API key

## Installation
```bash
git clone https://github.com/andy1924/Graph-RAG
cd Graph-RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure Environment
Create or edit `.env` in the repository root with valid credentials:
```dotenv
OPENAI_API_KEY=...
NEO4J_URI=...
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

## Minimal End-to-End Run
```bash
# 1) Ingest all corpora
python main.py ingest --all

# 2) Query GraphRAG
python main.py query --mode graphrag

# 3) Run evaluations
python main.py evaluate --experiment comprehensive
python main.py evaluate --experiment naiverag
python main.py evaluate --experiment significance

# 4) Generate plots and table
python experiments/visualize_results.py
```

## Common Variants
```bash
# Ingest one corpus only
python main.py ingest --corpus attention_paper

# Compare GraphRAG and NaiveRAG interactively
python main.py query --mode both

# Run all evaluation scripts in sequence
python main.py evaluate --all
```

## Expected Output Locations
- results/comprehensive_evaluation.json
- results/naiverag_evaluation.json
- results/significance_analysis.json
- results/visual_output/

## Verification Checklist
- Ingestion completes without connection errors.
- Evaluation JSON files are generated and non-empty.
- significance_analysis.json contains per_metric_significance.
- visual_output contains figures and aggregate table files.

