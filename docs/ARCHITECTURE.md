# System Architecture

Architecture for controlled GraphRAG vs NaiveRAG comparison with shared evaluation layer.

## High-Level Components

- **Ingestion Layer:** builds graph store (Neo4j) and chunk store (Chroma) from aligned corpora.
- **Retrieval Layer:** retrieves contextual evidence via graph traversal or chunk similarity.
- **Generation Layer:** produces answers grounded in retrieved context.
- **Evaluation Layer:** computes per-question and aggregate metrics.
- **Statistics Layer:** tests per-metric significance and effect size.

## Entry-Point Topology

- `main.py`: command router (`ingest`, `query`, `evaluate`).
- `scripts/ingest.py`: corpus ingestion orchestration.
- `scripts/query.py`: interactive query modes.
- `scripts/evaluate.py`: experiment dispatch.

## Package-Level Breakdown

### GraphRAG

- `src/graphrag/ingestion/multimodal_ingestion.py`
  - multimodal parsing and Neo4j persistence.
- `src/graphrag/ingestion/graph_generator.py`
  - graph construction helpers.
- `src/graphrag/retrieval.py`
  - `GraphRetriever`, `SemanticGraphRetriever`, `MultimodalGraphRetriever`.
- `src/graphrag/evaluation/metrics.py`
  - retrieval, quality, hallucination, and pipeline metrics.
- `src/graphrag/utils/`
  - utility wrappers for Neo4j, LLM calls, data access, logging.

### NaiveRAG

- `src/naiverag/ingestion.py`
  - chunking + vector index population.
- `src/naiverag/retrieval.py`
  - baseline retrieval and answer generation.
- `src/naiverag/config.py`
  - baseline config.

### Experiment Modules

- `experiments/comprehensive_evaluation.py`
- `experiments/naiverag_evaluation.py`
- `experiments/significance_analysis.py`
- `experiments/multimodal_ablation.py`
- `experiments/visualize_results.py`

## Data + Artifact Flow

- `data/raw/`: plain-text corpora.
- `data/multiModalPDF/`: PDF corpus inputs.
- `data/preprocessed/`: graph data exports.
- `data/chroma_db/`: baseline vector store state.
- `results/`: JSON outputs from evaluations/statistics.
- `results/visual_output/`: figures and aggregate table.

## Pipeline Sequence

### 1) Ingestion

1. Read corpus files.
2. Parse textual and multimodal units.
3. Build graph entities/relations for Neo4j.
4. Build chunk vectors for NaiveRAG baseline.

### 2) Query + Answer

1. Receive question.
2. Retrieve context.
3. Generate answer conditioned on retrieved evidence.
4. Return answer and metadata.

### 3) Evaluation + Statistics

1. Compute retrieval precision/recall/F1.
2. Compute ROUGE, BERTScore, semantic similarity.
3. Compute grounding signals (hallucination, grounded ratio).
4. Aggregate by corpus and globally.
5. Run significance tests and effect-size reporting.

## Design Constraints

1. `retrieval_f1` is non-equivalent across GraphRAG and NaiveRAG in current implementation.
2. API-dependent calls (LLM/embedding) introduce runtime variance.
3. Metric behavior depends on corpus composition and entity density.

## Reproducibility Notes

- Run GraphRAG and NaiveRAG evaluation before significance analysis.
- Regenerate plots only after refreshing JSON outputs.
- Keep seeds/config stable across reruns for cleaner comparison.

