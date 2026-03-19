# Quick Start Guide

Get GraphRAG running in under 10 minutes.

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Neo4j | Aura (cloud) or local 5.x |
| OpenAI API key | GPT-4o-mini access |

## 1. Installation

```bash
git clone https://github.com/yourusername/Graph-RAG.git
cd Graph-RAG

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

## 2. Environment Setup

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```dotenv
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

## 3. Ingest Data

Ingest the Attention paper (included as a test corpus):

```bash
python main.py ingest --corpus attention_paper --target naiverag
```

Ingest all available corpora:

```bash
python main.py ingest --all
```

## 4. Ask a Question

```bash
python main.py query --mode graphrag
```

Example session:

```
✓ GraphRAG retriever loaded

GraphRAG Query Interface  [graphrag mode]
============================================================
Type your question, or 'quit' to exit.

Question: What is the Transformer model?

  GraphRAG → The Transformer is a transduction model that relies entirely
  on self-attention mechanisms, dispensing with recurrence and convolutions.
```

## 5. Compare Systems

Run side-by-side comparison of GraphRAG vs NaiveRAG:

```bash
python main.py query --mode both
```

## 6. Run Evaluations

```bash
# Full evaluation suite
python main.py evaluate --all

# Individual experiments
python main.py evaluate --experiment comprehensive
python main.py evaluate --experiment naiverag
python main.py evaluate --experiment significance
```

Results are saved to `results/`.

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand system design
- Read [EVALUATION.md](EVALUATION.md) for metric definitions and analysis
- Read [USAGE.md](USAGE.md) for advanced configuration
