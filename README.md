---
**Authors**: Arnav Deshpande, Sarvesh Nimbalkar, Aadi Rawat, Dhruv Gadia  
**Institution**: Mukesh Patel School of Technology and Management, NMIMS, Mumbai  
**License**: MIT License  
**Contact**: deshpandearnavn@gmail.com  
**Last Updated**: March 2026  
---

# GraphRAG: Efficient Multi-Modal Graph-Based Retrieval-Augmented Generation

## Overview

**GraphRAG** is a research-oriented system for mitigating hallucinations in Large Language Models (LLMs) through efficient graph-based retrieval-augmented generation with multi-modal support. This project optimizes knowledge graph construction and traversal for multi-modal queries spanning text, tables, and images.

### Research Goal

To develop an **efficient and effective GraphRAG system** that:
- ✅ Significantly reduces LLM hallucinations through structured knowledge graphs
- ✅ Handles multi-modal documents (text, tables, images) seamlessly
- ✅ Optimizes graph traversal for efficient context retrieval
- ✅ Provides comprehensive evaluation metrics for system analysis

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Multi-Modal Document Input                      │
│        (PDF, Text, Images, Tables)                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│       Ingestion Pipeline                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Multimodal Document Processing                │  │
│  │    - PDF partitioning                            │  │
│  │    - Image extraction & vision-based summary     │  │
│  │    - Table structure inference                   │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2. Graph Document Generation                     │  │
│  │    - LLM-based entity extraction                 │  │
│  │    - Relationship identification                 │  │
│  │    - Metadata preservation                       │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│       Neo4j Knowledge Graph Storage                     │
│  - Entity nodes with properties                         │
│  - Typed relationships                                  │
│  - Multimodal node linking (PRECEDES, MENTIONS)        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│       Retrieval Pipeline                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Entity Extraction from Query                  │  │
│  │    - LLM-based entity recognition                │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 2. Graph Traversal & Context Retrieval           │  │
│  │    - Multi-hop relationship exploration          │  │
│  │    - Multimodal context aggregation              │  │
│  │    - Semantic ranking (optional)                 │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 3. Context-Grounded Answer Generation            │  │
│  │    - LLM with retrieved context                  │  │
│  │    - Reduced hallucination tendency              │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│       Evaluation Pipeline                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • Retrieval Quality (Precision, Recall, F1)      │  │
│  │ • Answer Quality (ROUGE, BERTScore, Semantic)    │  │
│  │ • Hallucination Detection & Measurement          │  │
│  │ • Multimodal Coverage & Relevance                │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. **Multi-Modal Document Processing**
   - PDF partitioning with structure inference
   - Vision-based image summarization (GPT-4 Vision)
   - Intelligent table extraction and description
   - Metadata preservation for traceability

### 2. **Graph-Based Knowledge Extraction**
   - LLM-powered entity and relationship extraction
   - Structured graph representation in Neo4j
   - Relationship type awareness (PRECEDES, MENTIONS, etc.)
   - Efficient node indexing and querying

### 3. **Advanced Retrieval**
   - Entity-based graph anchoring
   - Multi-hop relationship traversal
   - Semantic relevance ranking
   - Multimodal context aggregation

### 4. **Comprehensive Evaluation**
   - **Retrieval Metrics**: Precision, Recall, F1-Score
   - **Answer Quality**: ROUGE, BERTScore, Semantic Similarity
   - **Hallucination Detection**: Unsupported claim identification
   - **Multimodal Analysis**: Coverage and relevance per modality

---

## Project Structure

```
GraphRAG/
├── src/graphrag/                    # Main package
│   ├── __init__.py
│   ├── config.py                    # Configuration management
│   ├── ingestion/                   # Document processing
│   │   ├── graph_generator.py       # Graph extraction from text
│   │   └── multimodal_ingestion.py  # Multimodal processing
│   ├── retrieval/                   # Question answering
│   │   ├── graph_retriever.py       # Base retrieval system
│   │   └── multimodal_retriever.py  # Multimodal retrieval
│   ├── evaluation/                  # Metrics and assessment
│   │   └── metrics.py               # Comprehensive evaluation metrics
│   └── utils/                       # Helper functions
│       └── logger.py                # Logging utilities
│
├── experiments/                     # Research experiments
│   ├── baseline_comparison.py       # RAG vs GraphRAG
│   ├── multimodal_ablation.py       # Modality impact analysis
│   └── hallucination_study.py       # Hallucination measurement
│
├── benchmarks/                      # Benchmark datasets
│   ├── dataset1/                    # Example datasets
│   └── README.md                    # Benchmark documentation
│
├── results/                         # Evaluation results
│   └── README.md                    # Results documentation
│
├── tests/                           # Unit tests
│   └── test_*.py
│
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md
│   ├── EVALUATION.md
│   └── USAGE.md
│
├── README.md                        # This file
├── setup.py                         # Package setup
├── requirements.txt                 # Dependencies
└── .env.example                     # Environment template
```

---

## Installation

### Prerequisites
- Python 3.8+
- Neo4j database (4.0+)
- OpenAI API key
- Tesseract and Poppler (Windows: see setup.txt)

### Setup Steps

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Graph_RAG
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in development mode
   ```

4. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials:
   # - OPENAI_API_KEY=sk-...
   # - NEO4J_URI=bolt://localhost:7687
   # - NEO4J_USERNAME=neo4j
   # - NEO4J_PASSWORD=password
   ```

5. **Verify Neo4j Connection**
   ```bash
   python -c "from graphrag.ingestion import Neo4jGraphIngestor; print('Neo4j ready!')"
   ```

---

## Quick Start

### 1. Generate Graphs from Raw Documents

```python
from graphrag.ingestion import GraphDocumentGenerator

generator = GraphDocumentGenerator()
results = generator.generate_graphs(
    input_path="data/raw",
    output_path="data/preprocessed"
)

print(f"Generated {results['total_nodes']} nodes and {results['total_relationships']} relationships")
```

### 2. Ingest into Neo4j

```python
from graphrag.ingestion import Neo4jGraphIngestor

ingestor = Neo4jGraphIngestor()
results = ingestor.ingest_graph_data("data/preprocessed/graph_data.json")

print(f"Ingested into Neo4j: {results['total_nodes']} nodes")
```

### 3. Query the Graph

```python
from graphrag.retrieval import MultimodalGraphRetriever

retriever = MultimodalGraphRetriever()
answer, metadata = retriever.answer_with_multimodal_context(
    "What are the key benefits of graph-based RAG?"
)

print(f"Answer: {answer}")
print(f"Metadata: {metadata}")
```

### 4. Evaluate System

```python
from graphrag.evaluation import EvaluationPipeline

evaluator = EvaluationPipeline(experiment_id="exp_001")
metrics = evaluator.evaluate(
    question="What is X?",
    generated_answer="Answer generated by system",
    reference_answer="Ground truth answer",
    retrieved_context="Retrieved context from graph",
    retrieved_items=["node1", "node2"],
    relevant_items=["node1", "node2", "node3"]
)

evaluator.save_results(metrics)
```

---

## Usage Examples

### CLI Usage

```bash
# Generate graphs from text documents
python -m graphrag.ingestion.graph_generator data/raw data/preprocessed

# Query the graph
python -m graphrag.retrieval.multimodal_retriever "Your question here?"

# Evaluate system performance
python experiments/comprehensive_evaluation.py
```

### Python API

See [docs/USAGE.md](docs/USAGE.md) for detailed API documentation.

---

## Achieving the Research Goal

### Strategy for Efficient Multi-Modal GraphRAG

#### 1. **Multimodal Efficiency Optimization**
   - **Selective multimodal processing**: Only extract images/tables when needed
   - **Hierarchical traversal**: Start with text nodes, expand to other modalities if needed
   - **Modality-specific embeddings**: Use modality-appropriate models for ranking

#### 2. **Graph Optimization Techniques**
   - **Node indexing**: Index frequently accessed nodes for faster retrieval
   - **Relationship filtering**: Prioritize certain relationship types
   - **Caching mechanisms**: Cache common queries for reduced latency

#### 3. **Semantic Ranking**
   - Rank retrieved nodes by relevance to query
   - Use semantic similarity for relationship-aware traversal
   - Combine BM25 and semantic search

#### 4. **Hallucination Mitigation**
   - Ground all claims in retrieved context
   - Detect unsupported assertions (see evaluation metrics)
   - Force citation generation with retrieved nodes

#### 5. **Experimental Protocol**

**Baseline Comparison**:
- Compare against standard RAG without graphs
- Compare against vector similarity-only retrieval
- Measure improvements in answer accuracy and grounding

**Ablation Studies**:
- Multimodal impact: Text only vs. with tables vs. with images
- Graph depth: Impact of multi-hop vs. single-hop retrieval
- Ranking strategies: Semantic vs. heuristic-based

**Hallucination Measurement**:
- Quantify hallucination rate reduction
- Measure grounding ratio improvement
- Track fact consistency with context

---

## Evaluation Metrics

### Retrieval Quality
- **Precision**: Fraction of retrieved items that are relevant
- **Recall**: Fraction of relevant items that were retrieved
- **F1-Score**: Harmonic mean of precision and recall

### Answer Quality
- **ROUGE-1/2/L**: N-gram overlap with reference answers
- **BERTScore**: Contextual token similarity
- **Semantic Similarity**: Cosine similarity of sentence embeddings

### Hallucination Detection
- **Hallucination Rate**: Percentage of unsupported claims
- **Grounded Ratio**: Percentage of claims supported by context
- **Fact Consistency**: Entity accuracy relative to context

### Multimodal Metrics
- **Text/Table/Image Coverage**: Proportion of each modality in context
- **Modality Relevance**: Relevance score per modality type
- **Coverage Efficiency**: Quality of context relative to size

---

## Experimental Results

Results are stored in `results/` directory with timestamps. Each experiment includes:
- Configuration parameters
- Evaluation metrics
- Performance statistics
- Hallucination analysis

See [results/README.md](results/README.md) for detailed results.

---

## Research Repository Best Practices

This repository follows research best practices:
- ✅ **Modularity**: Independent, testable components
- ✅ **Reproducibility**: Configuration management and experiment logging
- ✅ **Documentation**: Comprehensive docstrings and markdown guides
- ✅ **Evaluation**: Rigorous metrics and benchmarking
- ✅ **Version Control**: Clean git history with meaningful commits
- ✅ **Extensibility**: Designed for future improvements and variations

---

## Contributing

For research contributions:
1. Create a new experiment in `experiments/`
2. Document your hypothesis and methodology
3. Run comprehensive evaluations
4. Update results documentation
5. Submit findings with supporting metrics

---

## Citation

If you use GraphRAG in your research, please cite:

```bibtex
@software{Graph-RAG 2026,
  title={GraphRAG: Efficient Multi-Modal Graph-Based RAG},
  author={Arnav Deshpande,Sarvesh Nimbalkar,Aadi Rawat,Dhruv Gadia},
  year={2026},
  url={https://github.com/andy1924/Graph-RAG}
}
```

---

## References

- [LangChain Graph Transformers](https://python.langchain.com/docs/modules/agents/tools/tools_as_openapi_operations)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Retrieval-Augmented Generation Papers](https://arxiv.org/search/?query=retrieval+augmented+generation&searchtype=all)
- [Hallucination in LLMs](https://arxiv.org/abs/2307.04087)

---

## License

MIT License - See LICENSE file for details

---

## Support

For questions and issues:
- 📧 Email: deshpandearnavn@gmail.com
- 🐛 GitHub Issues: [Create an issue](https://github.com/your-repo/graphrag/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/graphrag/discussions)

---

**Last Updated**: March 2026
