---
**Authors**: Arnav Deshpande, Sarvesh Nimbalkar, Aadi Rawat, Dhruv Gadia  
**Institution**: Mukesh Patel School of Technology and Management, NMIMS, Mumbai  
**License**: MIT License  
**Contact**: deshpandearnavn@gmail.com  
**Last Updated**: March 2026  
---

# GraphRAG - Usage Guide

This guide covers practical usage of the GraphRAG system.

## Installation

```bash
# Clone and setup
git clone <repo-url>
cd Graph_RAG
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Install package
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

## Basic Usage

### 1. Graph Generation

**Via Python API**:
```python
from graphrag.ingestion import GraphDocumentGenerator

# Initialize generator
generator = GraphDocumentGenerator()

# Generate graphs from raw documents
results = generator.generate_graphs(
    input_path="data/raw",
    output_path="data/preprocessed"
)

print(f"Generated {results['total_nodes']} nodes")
print(f"Generated {results['total_relationships']} relationships")
```

**Via CLI**:
```bash
python -m graphrag.ingestion.graph_generator data/raw data/preprocessed
```

### 2. Neo4j Ingestion

**Via Python**:
```python
from graphrag.ingestion import Neo4jGraphIngestor

ingestor = Neo4jGraphIngestor()
results = ingestor.ingest_graph_data("data/preprocessed/graph_data.json")

print(f"Ingested {results['total_nodes']} nodes into Neo4j")
```

### 3. Graph Querying

**Basic Retrieval**:
```python
from graphrag.retrieval import GraphRetriever

retriever = GraphRetriever()
answer, metadata = retriever.answer_question("What is entity X?")

print(f"Answer: {answer}")
print(f"Entity: {metadata['entity']}")
print(f"Context retrieved: {metadata['context_retrieved']}")
```

**Multimodal Retrieval**:
```python
from graphrag.retrieval import MultimodalGraphRetriever

retriever = MultimodalGraphRetriever()
answer, metadata = retriever.answer_with_multimodal_context(
    "What information do we have about X?"
)

print(f"Answer: {answer}")
print(f"Modalities used: {metadata['modalities_found']}")
print(f"Text blocks: {metadata['text_blocks']}")
print(f"Table blocks: {metadata['table_blocks']}")
print(f"Image blocks: {metadata['image_blocks']}")
```

**Via CLI**:
```bash
python -m graphrag.retrieval.multimodal_retriever "Your question here?"
```

### 4. Evaluation

**Single Query Evaluation**:
```python
from graphrag.evaluation import EvaluationPipeline

evaluator = EvaluationPipeline(experiment_id="test_001")

metrics = evaluator.evaluate(
    question="Test question?",
    generated_answer="System's answer",
    reference_answer="Ground truth answer",
    retrieved_context="Retrieved context from graph",
    retrieved_items=["node1", "node2", "node3"],
    relevant_items=["node1", "node2", "node3", "node4"]
)

print(f"F1 Score: {metrics.retrieval_f1:.3f}")
print(f"ROUGE: {metrics.rouge_score:.3f}")
print(f"Hallucination Rate: {metrics.hallucination_rate:.3f}")

# Save results
evaluator.save_results(metrics)
```

**Batch Evaluation**:
```python
from graphrag.evaluation import EvaluationPipeline

evaluator = EvaluationPipeline(experiment_id="batch_eval")
all_metrics = []

for question, reference in zip(questions, references):
    answer, metadata = retriever.answer_question(question)
    metrics = evaluator.evaluate(
        question=question,
        generated_answer=answer,
        reference_answer=reference,
        retrieved_context=metadata.get("context", "")
    )
    all_metrics.append(metrics)

# Analysis
import pandas as pd
df = pd.DataFrame([m.to_dict() for m in all_metrics])
print(df.describe())
```

## Advanced Usage

### Custom Configuration

```python
from graphrag.config import Config, RetrievalConfig

# Load custom config
config = Config()

# Modify settings
config.retrieval.max_hop_distance = 5
config.retrieval.relationships_limit = 20
config.model.llm_model = "gpt-4-turbo"

# Use custom config in retrievers
retriever = GraphRetriever()
# Retriever uses global config instance
```

### Multimodal Processing

```python
from graphrag.ingestion import MultimodalDocumentProcessor

processor = MultimodalDocumentProcessor()

# Process PDF with multimodal extraction
elements = processor.process_document("data/sample.pdf")

print(f"Extracted {len(elements)} multimodal elements")
for el in elements[:5]:
    print(f"  - {el.id}: {el.type}")
    print(f"    Content preview: {el.content[:100]}...")
```

### Semantic Ranking

```python
from graphrag.retrieval import SemanticGraphRetriever

# Use semantic-aware retriever
retriever = SemanticGraphRetriever()

# Get rankedresults
entity = retriever.extract_entity("your question?")
context_blocks = retriever.retrieve_context(entity)

# Rank by relevance
ranked = retriever.rank_relationships(
    context_blocks,
    "your question?",
    top_k=10
)

print(f"Top relevant contexts:")
for block in ranked[:5]:
    print(f"  {block}")
```

### Custom Evaluation Metrics

```python
from graphrag.evaluation import RetrievalMetrics, AnswerQualityMetrics

# Compute specific metrics
metrics = RetrievalMetrics()

precision = metrics.precision(
    retrieved_items=["A", "B", "C"],
    relevant_items=["A", "B", "C", "D"]
)
print(f"Precision: {precision:.3f}")

recall = metrics.recall(
    retrieved_items=["A", "B", "C"],
    relevant_items=["A", "B", "C", "D"]
)
print(f"Recall: {recall:.3f}")

# Answer quality
answer_metrics = AnswerQualityMetrics()

rouge = answer_metrics.rouge_score(
    generated="System answer",
    reference="Reference answer"
)
print(f"ROUGE-1: {rouge['rouge1']:.3f}")

similarity = answer_metrics.semantic_similarity(
    generated="System answer",
    reference="Reference answer"
)
print(f"Semantic Similarity: {similarity:.3f}")
```

### Hallucination Detection

```python
from graphrag.evaluation import HallucinationDetector

detector = HallucinationDetector()

# Detect unsupported claims
hallucination_rate, ungrounded_claims = detector.detect_unsupported_claims(
    answer="Generated answer text",
    context="Retrieved context",
    threshold=0.7
)

print(f"Hallucination Rate: {hallucination_rate:.3f}")
print(f"Ungrounded Claims:")
for claim in ungrounded_claims:
    print(f"  - {claim}")

# Check fact consistency
consistency = detector.fact_consistency_check(
    answer="Answer with named entities",
    context="Context with named entities"
)
print(f"Fact Consistency: {consistency:.3f}")
```

## Experiment Running

### Run Included Experiments

```bash
# Comprehensive evaluation
python experiments/comprehensive_evaluation.py

# Results are saved in results/comprehensive_evaluation.json
```

### Create Custom Experiment

```python
# experiments/my_experiment.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag.retrieval import GraphRetriever, MultimodalGraphRetriever
from graphrag.evaluation import EvaluationPipeline

def run_experiment():
    # Your experiment logic
    retriever = GraphRetriever()
    evaluator = EvaluationPipeline(experiment_id="my_exp")
    
    question = "Your test question?"
    answer, metadata = retriever.answer_question(question)
    
    metrics = evaluator.evaluate(
        question=question,
        generated_answer=answer,
        reference_answer="Ground truth",
        retrieved_context=metadata.get("context", "")
    )
    
    evaluator.save_results(metrics)
    return metrics

if __name__ == "__main__":
    results = run_experiment()
    print(f"Results saved!")
```

```bash
# Run custom experiment
python experiments/my_experiment.py
```

## Debugging and Logging

### Enable Logging

```python
import logging
from graphrag.utils import get_logger

# Setup logger
logger = get_logger(__name__, log_file="debug.log", level=logging.DEBUG)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Enable Verbose Output

```python
from graphrag.config import config

# All LLM calls will print details
config.ingestion.verbose = True
config.retrieval.verbose = True
```

## Performance Optimization

### Query Optimization

```python
# Tune retrieval parameters for speed vs quality tradeoff
config.retrieval.max_hop_distance = 2  # Reduce depth
config.retrieval.relationships_limit = 10  # Limit results
config.retrieval.use_semantic_ranking = False  # Faster, less accurate
```

### Batch Processing

```python
from graphrag.ingestion import GraphDocumentGenerator

generator = GraphDocumentGenerator()

# Process large document sets efficiently
documents = generator.load_documents("data/raw")
graphs = generator.transform_to_graph(documents)

# Save in batches
batch_size = 100
for i in range(0, len(graphs), batch_size):
    batch = graphs[i:i+batch_size]
    serialized = generator.serialize_graph_documents(batch)
    generator.save_graph_data(serialized, f"data/preprocessed/batch_{i}")
```

## Troubleshooting

### Neo4j Connection Issues

```python
from graphrag.ingestion import Neo4jGraphIngestor

try:
    ingestor = Neo4jGraphIngestor()
    print("✓ Neo4j connection successful")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("Check NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env")
```

### Memory Issues

```python
# Process documents in smaller batches
documents = generator.load_documents("data/raw", batch_size=10)
for batch in documents:
    graphs = generator.transform_to_graph(batch)
    generator.save_graph_data(
        generator.serialize_graph_documents(graphs),
        output_path="data/preprocessed"
    )
```

### Low Retrieval Quality

```python
# Debug retrieval pipeline
entity = retriever.extract_entity("Your question?")
print(f"Extracted entity: {entity}")

context = retriever.retrieve_context(entity, max_hops=2)
print(f"Retrieved context:\n{context[:500]}")

# If empty, check:
# 1. Are documents ingested into Neo4j?
# 2. Is the entity mentioned in the graph?
# 3. Try different entity names manually
```

## Best Practices

1. **Always use configuration**: Adjust hyperparameters per use case
2. **Validate inputs**: Check document quality before ingestion
3. **Monitor metrics**: Track hallucination rate and F1 score
4. **Version results**: Use experiment IDs for reproducibility
5. **Document assumptions**: Note what data/models were used
6. **Save intermediate results**: Don't rely on single runs
7. **Use semantic ranking**: For better answer quality
8. **Evaluate regularly**: Benchmark against baselines

---

For API reference, see docstrings in source files.
For architecture details, see [ARCHITECTURE.md](../docs/ARCHITECTURE.md)
For evaluation methodology, see [EVALUATION.md](../docs/EVALUATION.md)
