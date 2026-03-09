# GraphRAG Repository Structure Guide

## Quick Navigation

### 📚 Key Documentation
- **[README.md](README.md)** - Project overview and setup
- **[RESEARCH_GOAL.md](RESEARCH_GOAL.md)** - Research objectives and strategy
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and components
- **[docs/EVALUATION.md](docs/EVALUATION.md)** - Evaluation methodology and metrics
- **[docs/USAGE.md](docs/USAGE.md)** - Practical usage guide

### 💻 Source Code
```
src/graphrag/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
│
├── ingestion/               # Document processing pipeline
│   ├── graph_generator.py   # LLM-based graph extraction
│   └── multimodal_ingestion.py  # PDF/image/table processing
│
├── retrieval/               # Question answering pipeline
│   ├── graph_retriever.py   # Base graph-based retrieval
│   └── multimodal_retriever.py  # Enhanced multimodal retrieval
│
├── evaluation/              # Performance metrics
│   └── metrics.py           # Comprehensive evaluation suite
│
└── utils/                   # Helper functions
    └── logger.py            # Logging utilities
```

### 🧪 Experiments & Benchmarks
```
experiments/
├── comprehensive_evaluation.py   # Full system evaluation
└── multimodal_ablation.py        # Modality impact analysis

benchmarks/
├── README.md                     # Benchmark guidelines
└── [dataset folders]/           # Test datasets

results/
├── README.md                     # Results analysis guide
└── [evaluation_*.json]/         # Saved results
```

### ⚙️ Configuration
- **.env.example** - Environment variables template
- **setup.py** - Package installation configuration
- **requirements.txt** - Python dependencies

---

## Getting Started

### 1. Setup (5 minutes)
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install package
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Generate Graphs (10-30 minutes)
```bash
# From Python
python -c "
from graphrag.ingestion import GraphDocumentGenerator
gen = GraphDocumentGenerator()
gen.generate_graphs('data/raw', 'data/preprocessed')
"

# Or via CLI
python -m graphrag.ingestion.graph_generator data/raw data/preprocessed
```

### 3. Ingest into Neo4j (5-10 minutes)
```bash
# From Python
python -c "
from graphrag.ingestion import Neo4jGraphIngestor
ing = Neo4jGraphIngestor()
ing.ingest_graph_data('data/preprocessed/graph_data.json')
"
```

### 4. Query the System (< 1 second)
```bash
python -m graphrag.retrieval.multimodal_retriever "Your question?"
```

### 5. Evaluate Performance (30 minutes)
```bash
python experiments/comprehensive_evaluation.py
# Results saved in results/comprehensive_evaluation.json
```

---

## Research Planning

### Phase 1: Setup ✅ (COMPLETE)
- [x] Refactor code into modular packages
- [x] Implement comprehensive evaluation metrics
- [x] Create benchmark infrastructure
- [x] Document system architecture

### Phase 2: Optimization (IN PROGRESS)
- [ ] Run multimodal ablation studies
- [ ] Optimize graph traversal
- [ ] Implement semantic ranking
- [ ] Measure hallucination reduction

### Phase 3: Validation (PLANNED)
- [ ] Run scaling experiments
- [ ] Compare against baselines
- [ ] Statistical significance testing
- [ ] Generate publication materials

---

## Key Concepts

### Hallucination Reduction 🎯
**Goal**: Reduce LLM tendency to make up facts

**Approach**:
1. Retrieve relevant knowledge graph context
2. Ground answer generation in retrieved facts
3. Detect and flag unsupported claims
4. Measure grounding ratio (% of claims supported)

**Metrics**:
- Hallucination Rate: % of unsupported claims (target: < 10%)
- Grounded Ratio: % of supported claims (target: > 90%)
- Fact Consistency: Entity/fact accuracy

### Multi-Modal Efficiency 🚀
**Goal**: Handle text, tables, and images without excessive latency increase

**Approach**:
1. Selective modality processing (only extract when needed)
2. Hierarchical retrieval (start with text, add modalities if needed)
3. Modality-specific indexing (faster lookup)
4. Lazy loading (defer heavy computation)

**Metrics**:
- Response Time: seconds per query (target: < 1.0s)
- Modality Usage: % of each modality in context
- F1 Score with multimodal context (target: > 0.90)

### Graph Efficiency 📈
**Goal**: Fast and accurate knowledge graph retrieval

**Approach**:
1. Neo4j indexing for O(1) lookups
2. Relationship filtering (high-confidence only)
3. Traversal depth optimization (dynamic limits)
4. Semantic ranking (sort by relevance)

**Metrics**:
- Retrieval Precision: % of relevant results (target: > 0.90)
- Retrieval Recall: % of relevant found (target: > 0.85)
- F1 Score: harmonic mean (target: > 0.85)

---

## Evaluation Metrics Summary

### Answer Quality
| Metric | Meaning | Target |
|--------|---------|--------|
| ROUGE | N-gram overlap with reference | > 0.40 |
| BERTScore | Contextual token similarity | > 0.75 |
| Semantic Similarity | Sentence embedding cosine | > 0.75 |

### Retrieval Quality
| Metric | Meaning | Target |
|--------|---------|--------|
| Precision | Relevant / Retrieved | > 0.90 |
| Recall | Retrieved / Relevant | > 0.85 |
| F1 | Harmonic mean | > 0.85 |

### Hallucination
| Metric | Meaning | Target |
|--------|---------|--------|
| Hall. Rate | Unsupported / Total | < 0.10 |
| Grounded Ratio | Supported / Total | > 0.90 |
| Fact Consistency | Entities Consistent | > 0.90 |

### Efficiency
| Metric | Meaning | Target |
|--------|---------|--------|
| Response Time | Seconds per query | < 1.0s |
| Throughput | Questions / second | > 1.0/s |
| Modality Coverage | % of rich modalities | > 20% |

---

## Common Tasks

### Run Specific Analysis
```python
# Hallucination detection
from graphrag.evaluation import HallucinationDetector
detector = HallucinationDetector()
rate, claims = detector.detect_unsupported_claims(answer, context)

# Multimodal metrics
from graphrag.evaluation import MultimodalMetrics
mm = MultimodalMetrics()
coverage = mm.modality_coverage(context_by_modality)

# Semantic ranking
from graphrag.retrieval import SemanticGraphRetriever
retriever = SemanticGraphRetriever()
ranked = retriever.rank_relationships(blocks, question, top_k=10)
```

### Create Benchmark
```bash
# In benchmarks/my_dataset/
mkdir -p documents
# Add your documents to documents/

# Create questions.json and references.json
# See benchmarks/README.md for format

# Run evaluation
python experiments/benchmark_evaluation.py --dataset benchmarks/my_dataset
```

### Debug Issues
```python
# Enable detailed logging
from graphrag.utils import get_logger
logger = get_logger(__name__, log_file="debug.log", level=logging.DEBUG)

# Check entity extraction
entity = retriever.extract_entity("Your question?")
print(f"Extracted: {entity}")

# Inspect retrieved context
context = retriever.retrieve_context(entity)
print(f"Context length: {len(context)}")
print(context[:500])
```

---

## Performance Expectations

### Default Configuration
- **Graph Generation**: 5-10 sec per 10K words
- **Neo4j Ingestion**: 1-2 sec per 1K nodes
- **Query Latency**: 0.2-0.5 seconds
- **Answer Generation**: 2-5 seconds (LLM dependent)
- **Evaluation Time**: 1-2 seconds per metric

### With Optimization
- **Response Time**: < 1 second (including LLM)
- **Throughput**: 2-3 questions/second
- **Hallucination Rate**: 8-12%
- **F1 Score**: 0.85-0.92

---

## Troubleshooting

### Problem: Neo4j Connection Failed
**Solution**: Verify credentials in .env and Neo4j is running
```bash
# Check if Neo4j is accessible
python -c "from graphrag.ingestion import Neo4jGraphIngestor; print('Connected!')"
```

### Problem: Low Retrieval Quality
**Solution**: Check if documents are ingested
```bash
# Query Neo4j for nodes
python -c "
from graphrag.retrieval import GraphRetriever
r = GraphRetriever()
# Check if graph has any nodes
"
```

### Problem: Slow Response Time
**Solution**: Reduce graph traversal depth
```python
config.retrieval.max_hop_distance = 2  # Default: 3
config.retrieval.relationships_limit = 10  # Default: 15
```

---

## Next Steps

1. **Read** [RESEARCH_GOAL.md](RESEARCH_GOAL.md) for detailed strategy
2. **Setup** environment following [docs/USAGE.md](docs/USAGE.md)
3. **Run** `python experiments/comprehensive_evaluation.py`
4. **Review** results in `results/comprehensive_evaluation.json`
5. **Explore** modality impact with `python experiments/multimodal_ablation.py`
6. **Contribute** by creating new experiments in `experiments/`

---

## File Access by Use Case

### I want to...

#### ...understand the system
- Start: [README.md](README.md)
- Deep dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

#### ...get started quickly
- Setup: [docs/USAGE.md](docs/USAGE.md#installation)
- Examples: [docs/USAGE.md](docs/USAGE.md#basic-usage)

#### ...run experiments
- Quickstart: [docs/USAGE.md](docs/USAGE.md#experiment-running)
- Strategy: [RESEARCH_GOAL.md](RESEARCH_GOAL.md)

#### ...evaluate performance
- Methods: [docs/EVALUATION.md](docs/EVALUATION.md)
- Code: [src/graphrag/evaluation/metrics.py](src/graphrag/evaluation/metrics.py)

#### ...optimize the system
- Roadmap: [RESEARCH_GOAL.md](RESEARCH_GOAL.md#implementation-roadmap)
- Code: [src/graphrag/retrieval/](src/graphrag/retrieval/)

#### ...understand results
- Interpretation: [results/README.md](results/README.md#key-metrics-interpretation)
- Analysis: [results/README.md](results/README.md#analysis-scripts)

---

**Last Updated**: March 9, 2025
**Status**: Research Setup Complete ✅
**Next Phase**: Optimization & Experiments 📋
