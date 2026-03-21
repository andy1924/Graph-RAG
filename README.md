# GraphRAG: Graph-Based Retrieval-Augmented Generation

> Reducing LLM hallucinations through structured knowledge graphs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Neo4j](https://img.shields.io/badge/Neo4j-Aura-008CC1.svg)](https://neo4j.com)

## Overview

GraphRAG is a research system that replaces traditional vector-based retrieval (NaiveRAG) with **knowledge graph traversal** for grounding LLM responses. Documents are ingested into a Neo4j knowledge graph, and questions are answered by traversing entity relationships rather than performing similarity search over text chunks.

### Key Contributions

- **Multimodal ingestion** — text, tables, and images from PDFs are extracted, structured into a knowledge graph, and stored in Neo4j
- **Semantic entity selection** — candidate entities are ranked by embedding similarity, then filtered by an LLM for relevance
- **Temporal & numeric filtering** — Cypher queries are augmented with year-aware filters and numeric prioritisation to reduce retrieval noise
- **Two-stage hallucination detection** — cosine similarity screening followed by NLI entailment rescue to accurately measure grounding

## Results

Evaluated across 4 corpora (60 questions total):

| Metric | GraphRAG | NaiveRAG |
|---|---|---|
| **Hallucination Rate** | 23.3% | 6.9% |
| **Semantic Similarity** | 63.7% | 81.1% |
| **Retrieval F1** | 21.3% | — |
| **Avg Response Time** | 4.58s | 4.33s |

Statistical significance analysis (Wilcoxon signed-rank, p = 0.002) confirms the difference in hallucination rates is highly significant.

> **Note:** NaiveRAG achieves lower hallucination through verbatim chunk retrieval, while GraphRAG trades surface-level similarity for structured, relationship-aware context. See [docs/EVALUATION.md](docs/EVALUATION.md) for detailed analysis.

## Quick Start

### Prerequisites

- Python 3.10+
- [Neo4j Aura](https://neo4j.com/cloud/aura/) account (or local Neo4j instance)
- OpenAI API key

### Installation

```bash
git clone https://github.com/yourusername/Graph-RAG.git
cd Graph-RAG

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
# Optional: install test/lint tooling
# pip install -r requirements-dev.txt

cp .env.example .env
# Edit .env with your API keys and Neo4j credentials
```

### Usage

```bash
# Ingest a corpus into both GraphRAG (Neo4j) and NaiveRAG (ChromaDB)
python main.py ingest --corpus attention_paper
python main.py ingest --all

# Interactive query interface
python main.py query --mode graphrag
python main.py query --mode both          # side-by-side comparison

# Run evaluations
python main.py evaluate --experiment comprehensive
python main.py evaluate --experiment naiverag
python main.py evaluate --experiment significance
python main.py evaluate --all
```

## Project Structure

```
Graph-RAG/
├── main.py                        # Unified CLI launcher
├── setup.py                       # Package installation
├── requirements.txt               # Dependencies
├── .env.example                   # Environment template
│
├── src/                           # Core implementation
│   ├── graphrag/                  #   GraphRAG system
│   │   ├── config.py              #     Configuration
│   │   ├── retrieval.py           #     Graph retrieval + LLM answering
│   │   ├── ingestion/             #     PDF → knowledge graph pipeline
│   │   ├── evaluation/            #     Metrics (F1, hallucination, ROUGE)
│   │   └── utils/                 #     Neo4j manager, logger, helpers
│   │
│   └── naiverag/                  #   Baseline vector retrieval
│       ├── config.py              #     Configuration
│       ├── retrieval.py           #     ChromaDB retrieval + LLM answering
│       └── ingestion.py           #     PDF/text → ChromaDB pipeline
│
├── scripts/                       # User-facing CLI tools
│   ├── ingest.py                  #   Data ingestion
│   ├── query.py                   #   Interactive querying
│   └── evaluate.py                #   Evaluation runner
│
├── experiments/                   # Research experiments
│   ├── comprehensive_evaluation.py
│   ├── naiverag_evaluation.py
│   ├── significance_analysis.py
│   └── multimodal_ablation.py
│
├── tests/                         # Test suite
│   ├── test_graphrag.py
│   ├── test_naiverag.py
│   └── test_metrics.py
│
├── data/                          # Data storage
│   ├── raw/                       #   Source documents (.txt)
│   └── preprocessed/              #   Graph JSON exports
│
├── results/                       # Evaluation outputs
│   ├── comprehensive_evaluation.json
│   ├── naiverag_evaluation.json
│   └── significance_analysis.json
│
└── docs/                          # Documentation
    ├── QUICKSTART.md
    ├── ARCHITECTURE.md
    ├── EVALUATION.md
    └── USAGE.md
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Document   │────▶│   Ingestion   │────▶│   Neo4j KG   │
│  (PDF/Text)  │     │   Pipeline    │     │  (Entities + │
└─────────────┘     └──────────────┘     │ Relationships)│
                                          └──────┬───────┘
                                                 │
┌─────────────┐     ┌──────────────┐             │
│   Question   │────▶│  Retrieval    │◀────────────┘
│              │     │  Pipeline     │
└─────────────┘     │  1. Keyword   │
                    │     filter    │
                    │  2. Embedding │
                    │     ranking   │
                    │  3. LLM entity│
                    │     selection │
                    │  4. Cypher    │
                    │     traversal │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   LLM Answer  │
                    │  Generation   │
                    │  (grounded)   │
                    └──────────────┘
```

## Documentation

| Document | Description |
|---|---|
| [Quick Start](docs/QUICKSTART.md) | Installation and first query in 10 minutes |
| [Architecture](docs/ARCHITECTURE.md) | System design and component details |
| [Evaluation](docs/EVALUATION.md) | Metrics, methodology, and result analysis |
| [Usage Guide](docs/USAGE.md) | Advanced configuration and API reference |

## Citation

```bibtex
@software{graphrag2026,
    title   = {GraphRAG: Graph-Based Retrieval-Augmented Generation},
    author  = {Deshpande, Arnav and Nimbalkar, Sarvesh and Rawat, Aadi and Gadia, Dhruv},
    year    = {2026},
    institution = {NMIMS Mumbai}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
