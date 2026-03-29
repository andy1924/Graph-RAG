# System Architecture

## Project Metadata
- Project Title: Beyond Vector Search: Mitigating LLM Hallucinations via Graph-Based Retrieval-Augmented Generation (GraphRAG)
- Authors: Arnav Deshpande; Sarvesh Nimbalkar; Dhruv Gadia; Aadi Rawat
- Organization: Mukesh Patel School of Technology and Management, NMIMS University
- Contact Email: [deshpandearnavn@gmail.com](mailto:deshpandearnavn@gmail.com)
- GitHub Repository: https://github.com/andy1924/Graph-RAG

## Architectural Scope
The repository implements two retrieval pipelines that share evaluation infrastructure:
- GraphRAG: graph-centric retrieval over Neo4j.
- NaiveRAG: chunk-centric retrieval baseline.

The architecture is organized for controlled experimentation rather than production deployment.

## Code-Level Architecture
### Entry points
- main.py delegates to command-specific scripts.
- scripts/ingest.py orchestrates ingestion for GraphRAG and NaiveRAG.
- scripts/query.py provides interactive query modes.
- scripts/evaluate.py runs experiment modules.

### GraphRAG package
- src/graphrag/ingestion/
  - multimodal_ingestion.py: multimodal ingestion and Neo4j population.
  - graph_generator.py: graph construction utilities.
- src/graphrag/retrieval.py
  - GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever.
- src/graphrag/evaluation/metrics.py
  - retrieval metrics, answer quality metrics, hallucination detection, evaluation pipeline.
- src/graphrag/utils/
  - neo4j_manager.py, llm_client.py, logger.py, data_retriever.py.

### NaiveRAG package
- src/naiverag/ingestion.py: chunk ingestion and vector-store preparation.
- src/naiverag/retrieval.py: baseline retrieval and answering.
- src/naiverag/config.py: baseline-specific configuration.

### Experiment layer
- experiments/comprehensive_evaluation.py: corpus-level GraphRAG evaluation.
- experiments/naiverag_evaluation.py: baseline evaluation with per-question metrics.
- experiments/significance_analysis.py: inferential testing and effect-size reporting.
- experiments/multimodal_ablation.py: modality sensitivity analysis.

## Data and Results Artifacts
- data/raw/: raw text corpora.
- data/multiModalPDF/: PDF source inputs.
- data/preprocessed/: graph JSON exports.
- data/chroma_db/: baseline vector store.
- results/: experiment outputs and significance reports.
- results/visual_output/: generated figures and aggregate table image/CSV.

## Pipeline Description
### Ingestion pipeline
1. Load corpus document(s).
2. Extract text and multimodal units.
3. Build graph entities/relations and persist to Neo4j.
4. Build NaiveRAG chunk index for baseline runs.

### Retrieval and generation pipeline
1. Parse question.
2. Retrieve supporting context:
   - GraphRAG: node-centric graph traversal and multimodal context collection.
   - NaiveRAG: chunk retrieval over indexed text.
3. Generate answer from retrieved context.
4. Return answer plus retrieval metadata.

### Evaluation pipeline
1. Compute retrieval metrics (precision/recall/F1).
2. Compute answer quality metrics (ROUGE, BERTScore, semantic similarity).
3. Compute hallucination and grounding metrics.
4. Aggregate per-corpus and cross-corpus summaries.
5. Run significance analysis for selected metrics.

## Design Constraints
- Retrieval F1 definitions differ between GraphRAG and NaiveRAG in current outputs.
- API-dependent steps (LLM and embedding calls) introduce runtime variance.
- Metric interpretation depends on corpus-specific entity distributions.

## Reproducibility Notes
- Use the same random seeds defined in evaluation scripts.
- Run GraphRAG and NaiveRAG evaluations before significance analysis.
- Regenerate visual outputs only after JSON result files are current.

