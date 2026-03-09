# GraphRAG Quick Reference

## Installation (5 minutes)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt && pip install -e .
cp .env.example .env
# Edit .env with your API keys
```

## Common Commands

### 🔧 Setup
```bash
# Install in development mode
pip install -e .

# Install with evaluation extras
pip install -e ".[evaluation]"

# Run tests
pytest tests/
```

### 📊 Data Processing
```bash
# Generate graphs from raw documents
python -m graphrag.ingestion.graph_generator data/raw data/preprocessed

# Query Neo4j status
python -c "from graphrag.ingestion import Neo4jGraphIngestor; Neo4jGraphIngestor()"
```

### 🔍 Querying
```bash
# Simple query
python -m graphrag.retrieval.graph_retriever "What is X?"

# Multimodal query
python -m graphrag.retrieval.multimodal_retriever "Your question?"
```

### 📈 Evaluation
```bash
# Run comprehensive evaluation
python experiments/comprehensive_evaluation.py

# Run multimodal ablation study
python experiments/multimodal_ablation.py

# Evaluate specific benchmark
python experiments/benchmark_evaluation.py --dataset benchmarks/dataset1
```

### 📚 Python API
```python
# Graph generation
from graphrag.ingestion import GraphDocumentGenerator
gen = GraphDocumentGenerator()
results = gen.generate_graphs("data/raw", "data/preprocessed")

# Graph retrieval
from graphrag.retrieval import GraphRetriever
retriever = GraphRetriever()
answer, metadata = retriever.answer_question("Your question?")

# Evaluation
from graphrag.evaluation import EvaluationPipeline
evaluator = EvaluationPipeline(experiment_id="test_001")
metrics = evaluator.evaluate(
    question="Q?",
    generated_answer="Answer",
    reference_answer="Reference",
    retrieved_context="Context"
)

# Hallucination detection
from graphrag.evaluation import HallucinationDetector
detector = HallucinationDetector()
rate, claims = detector.detect_unsupported_claims(answer, context)
```

## Configuration

### Via Environment Variables (.env)
```
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://localhost:7687
LLM_MODEL=gpt-4o
MAX_HOP_DISTANCE=3
HALLUCINATION_THRESHOLD=0.7
```

### Via Python Config
```python
from graphrag.config import config

# Modify settings
config.retrieval.max_hop_distance = 5
config.model.llm_model = "gpt-4-turbo"
config.evaluation.detection_hallucinations = True

# View settings
print(config.to_dict())
```

## Key Files

| File | Purpose |
|------|---------|
| **README.md** | Overview & setup |
| **RESEARCH_GOAL.md** | Research strategy |
| **STRUCTURE.md** | Navigation guide |
| **docs/ARCHITECTURE.md** | System design |
| **docs/EVALUATION.md** | Metrics & testing |
| **docs/USAGE.md** | Detailed usage |
| **src/graphrag/config.py** | Configuration |
| **src/graphrag/evaluation/metrics.py** | All metrics |

## Metrics Quick Reference

```python
# Retrieval Metrics (0-1, higher=better)
precision = retrieval_metrics.precision(retrieved, relevant)
recall = retrieval_metrics.recall(retrieved, relevant)
f1 = retrieval_metrics.f1_score(precision, recall)

# Answer Quality (0-1, higher=better)
rouge = answer_metrics.rouge_score(generated, reference)
semantic_sim = answer_metrics.semantic_similarity(generated, reference)
bert_score = answer_metrics.bert_score(generated, reference)

# Hallucination (0-1, lower=better for rate, higher=better for grounding)
hall_rate, claims = detector.detect_unsupported_claims(answer, context)
grounding_score = 1.0 - hall_rate

# Multimodal (0-1, higher=better)
coverage = multimodal_metrics.modality_coverage(context_by_modality)
relevance = multimodal_metrics.multimodal_relevance(question, context)
```

## Benchmark Results Targets

| Metric | Target | Comments |
|--------|--------|----------|
| **F1 Score** | > 0.85 | Retrieval quality |
| **Hallucination Rate** | < 0.10 | 10% or less |
| **Grounded Ratio** | > 0.90 | 90%+ grounded |
| **Semantic Similarity** | > 0.75 | Answer quality |
| **Response Time** | < 1.0s | Total latency |

## Troubleshooting

**Neo4j connection fails**
```bash
# Check credentials in .env
# Verify Neo4j is running
# Test: python -c "from graphrag.ingestion import Neo4jGraphIngestor"
```

**Low retrieval quality**
```bash
# Check: Are documents ingested into Neo4j?
# Try: Increase max_hop_distance (config.retrieval.max_hop_distance = 5)
# Try: Reduce relationships_limit (config.retrieval.relationships_limit = 20)
```

**Slow response time**
```bash
# Reduce: max_hop_distance = 2 (default: 3)
# Reduce: relationships_limit = 10 (default: 15)
# Disable: use_semantic_ranking = False
```

**High hallucination rate**
```bash
# Force grounding in LLM prompts
# Lower hallucination_threshold = 0.5
# Use semantic ranking: use_semantic_ranking = True
```

## Learning Path

1. **5 min**: Read [README.md](README.md)
2. **10 min**: Review [STRUCTURE.md](STRUCTURE.md)
3. **15 min**: Study [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
4. **20 min**: Run first experiment: `python experiments/comprehensive_evaluation.py`
5. **30 min**: Try [docs/USAGE.md](docs/USAGE.md) examples
6. **60 min**: Design and run custom experiment

## Research Timeline Example

**Week 1**: Setup & Baseline
- Install package
- Generate graphs from documents
- Run `comprehensive_evaluation.py`
- Document baseline metrics

**Week 2**: Multimodal Optimization
- Run `multimodal_ablation.py`
- Analyze modality impact
- Compare with baseline
- Document findings

**Week 3**: Graph Optimization
- Profile retrieval latency
- Implement indexing improvements
- Measure performance gains
- Compare efficiency metrics

**Week 4**: Evaluation & Analysis
- Run all experiments
- Statistical significance testing
- Create visualizations
- Document conclusions

## Useful Commands

```bash
# List all available metrics
grep "class.*Metrics" src/graphrag/evaluation/metrics.py

# Count lines of code
wc -l src/graphrag/evaluation/metrics.py

# Find all evaluation references
grep -r "EvaluationPipeline" src/

# View latest results
cat results/evaluation_*.json | tail -1

# Monitor progress
watch -n 5 "ls -lh results/"

# Clear results (careful!)
rm results/evaluation_*.json
```

## Documentation Links

- **Overview**: [README.md](README.md)
- **Quick Start**: [docs/USAGE.md](docs/USAGE.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Evaluation**: [docs/EVALUATION.md](docs/EVALUATION.md)
- **Research Strategy**: [RESEARCH_GOAL.md](RESEARCH_GOAL.md)
- **Navigation**: [STRUCTURE.md](STRUCTURE.md)
- **Status**: [COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md)

---

**Pro Tip**: Start with `python experiments/comprehensive_evaluation.py` to see everything working!

Last updated: March 9, 2025
