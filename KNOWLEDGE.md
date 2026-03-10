# GraphRAG Project Knowledge Base

**Project**: Graph-based Retrieval-Augmented Generation system with multi-modal support
**Purpose**: Reduce LLM hallucinations using graph structure + multimodal retrieval
**Status**: Evaluation framework implemented and running
**Date**: March 10, 2026

---

## Project Structure

```
d:\Graph_RAG/
├── src/graphrag/
│   ├── config.py              # Configuration management
│   ├── retrieval.py           # GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever
│   ├── evaluation/
│   │   ├── metrics.py         # EvaluationPipeline, evaluation metrics
│   │   └── __init__.py
│   ├── ingestion/
│   │   ├── graph_generator.py # Graph generation
│   │   └── multimodal_ingestion.py
│   └── utils/
│       ├── logger.py
│       └── data_retriever.py  # [NEW] Loads relevant items from graph_data.json
├── experiments/
│   ├── comprehensive_evaluation.py  # [UPDATED] Main evaluation script
│   └── multimodal_ablation.py       # Ablation study script
├── data/
│   ├── raw/                   # Google.txt, Microsoft.txt, Nvidia.txt, SpaceX.txt, Tesla.txt
│   ├── multiModalPDF/         # PDF documents
│   └── preprocessed/graph_data.json  # [KEY] Extracted graph nodes (100+ nodes from PDFs)
├── results/                   # Evaluation outputs (JSON, logs)
├── docs/
│   ├── EVALUATION.md          # Evaluation methodology & metrics
│   ├── ARCHITECTURE.md
│   ├── QUICKSTART.md
│   ├── RESEARCH_GOAL.md
│   └── USAGE.md
├── requirements.txt           # Dependencies
├── EVALUATION_GUIDE.md        # [NEW] How to run evaluations
└── KNOWLEDGE.md              # [THIS FILE]
```

---

## Key Components

### 1. Evaluation Metrics (docs/EVALUATION.md)

**Retrieval Metrics:**
- **Precision**: |Retrieved ∩ Relevant| / |Retrieved|
- **Recall**: |Retrieved ∩ Relevant| / |Relevant|
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

**Answer Quality:**
- **ROUGE**: N-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore**: Contextual embedding similarity
- **Semantic Similarity**: Cosine similarity of sentence embeddings

**Hallucination Detection:**
- **Unsupported Claims**: Sentences lacking context support (threshold: 0.7)
- **Hallucination Rate**: (Unsupported sentences) / (Total sentences)
- **Grounded Ratio**: 1 - Hallucination Rate

**Multimodal Metrics:**
- **Modality Coverage**: Proportion from each modality type
- **Modality Relevance**: Semantic relevance to question

---

## Recent Changes & Implementation

### 1. Data Retriever Module (NEW)
**File**: `src/graphrag/utils/data_retriever.py`

**Purpose**: Replace hardcoded placeholder items with actual graph nodes

**Key Classes**:
- `RelevantItemsRetriever`: Loads graph_data.json, extracts nodes, matches to questions
- `get_relevant_items_mapping()`: Batch retrieval for all evaluation questions

**How It Works**:
1. Loads preprocessed graph data from `data/preprocessed/graph_data.json`
2. Extracts all nodes (Transformer, Google Brain, Ashish Vaswani, etc.)
3. For each question, extracts keywords (remove stop words)
4. Scores nodes based on keyword matches in ID and type
5. Returns top N matching nodes as relevant items

**Example**:
```python
from graphrag.utils.data_retriever import get_relevant_items_mapping

items = get_relevant_items_mapping(
    ["What is Transformer?", "Who created it?"],
    question_keywords={"What is Transformer?": ["Transformer", "architecture"]}
)
# Returns: [["Transformer", "Google Brain", ...], ["Ashish Vaswani", "Google", ...]]
```

### 2. Updated Benchmark Dataset
**File**: `experiments/comprehensive_evaluation.py`, class `BenchmarkDataset`

**Questions** (5 Transformer-focused):
1. "What are the main characteristics of the Transformer architecture?"
2. "How does Multi-Head Attention relate to Scaled Dot-Product Attention?"
3. "What is the performance significance of the Transformer model on WMT 2014 English-to-German translation?"
4. "Compare computational complexity per layer of self-attention vs recurrent layers"
5. "What is the impact of masking in decoder's self-attention sub-layer?"

**References**: Actual ground truth answers from Transformer paper with citations

**Relevant Items**: Dynamically retrieved from graph_data.json using data_retriever module

### 3. Retrieval Module (NEW)
**File**: `src/graphrag/retrieval.py`

**Classes**:
- `GraphRetriever`: Basic graph-based retrieval
  - `answer_question(question) → (answer, metadata)`
- `SemanticGraphRetriever`: Semantic-aware retrieval
- `MultimodalGraphRetriever`: Multimodal context support
  - `answer_with_multimodal_context(question, include_modalities) → (answer, metadata)`
  - Supports `["text", "table", "image"]` modalities

### 4. Fixed Issues
- **Unicode Encoding Errors**: Replaced ✓ with [+], ✗ with [!], → with ->
- **Missing Dependencies**: Installed langchain-community, langchain, langchain-core
- **Import Errors**: Updated graphrag/__init__.py to import from retrieval.py

---

## Evaluation Framework

### Running Evaluations

**Basic Command**:
```bash
cd d:\Graph_RAG
python experiments\comprehensive_evaluation.py
```

### What the Script Does

1. **Loads Benchmark Dataset**
   - 5 Transformer-related questions
   - Corresponding reference answers
   - Dynamically retrieves relevant items from graph (via data_retriever)

2. **Baseline Experiment** (~5 min)
   - Runs GraphRetriever on each question
   - Evaluates: F1, Hallucination Rate, Semantic Similarity, Response Time
   - Aggregates metrics

3. **Multimodal Ablation Study** (~15 min)
   - Tests 5 combinations:
     - Text only
     - Text + Table
     - Text + Table + Image
     - Table only
     - Image only
   - For each: evaluates all metrics + modality usage

4. **Saves Results**
   - JSON: `results/comprehensive_evaluation.json`
   - Console: Summary of baseline + best multimodal config
   - Logs: `results/experiment_comprehensive_eval_*.log`

### Output Structure
```json
{
  "timestamp": "2026-03-10 18:11:43",
  "baseline": {
    "experiment": "baseline",
    "num_questions": 5,
    "avg_f1": 0.668,
    "avg_hallucination_rate": 0.0,
    "avg_semantic_similarity": 0.XXX,
    "avg_response_time": X.XXX
  },
  "baseline_details": [
    {
      "question": "...",
      "answer": "...",
      "metrics": {...},
      "response_time": X.XXX
    }
  ],
  "multimodal_ablation": {
    "text": {...metrics...},
    "text+table": {...metrics...},
    "text+table+image": {...metrics...},
    ...
  }
}
```

---

## Alternative: Run Multimodal Ablation Only

**File**: `experiments/multimodal_ablation.py`

**Classes**:
- `MultimodalAblationStudy`: Systematic modality testing
- `run_full_ablation()`: Tests 7 combinations

**Usage**:
```python
from experiments.multimodal_ablation import MultimodalAblationStudy, analyze_ablation_results

study = MultimodalAblationStudy()
results = study.run_full_ablation(questions, references)
analyze_ablation_results(results)
```

---

## Customization Guide

### Update Evaluation Questions
**File**: `experiments/comprehensive_evaluation.py` (lines 29-51)

```python
class BenchmarkDataset:
    def __init__(self):
        self.questions = [
            "Your question 1",
            "Your question 2",
            ...
        ]
        self.references = [
            "Your answer 1",
            "Your answer 2",
            ...
        ]
        # relevant_items auto-loaded from graph data
```

### Custom Keyword Mapping
**File**: `src/graphrag/utils/data_retriever.py` (end of file)

```python
QUESTION_KEYWORDS_MAPPING = {
    "What is Transformer?": ["Transformer", "architecture", "attention"],
    "How does...": ["keyword1", "keyword2"],
}
```

### Adjust Retrieval Parameters
**File**: `src/graphrag/utils/data_retriever.py`, method `get_relevant_items_for_question()`

```python
def get_relevant_items_for_question(
    question: str,
    keywords: List[str] = None,
    num_items: int = 5  # ← Change this
) -> List[str]:
```

---

## Graph Data Overview

**Source**: `data/preprocessed/graph_data.json`

**Content**: Multi-page document with nodes extracted from PDFs

**Key Node Types**:
- **Organization**: Google, Google Brain, University of Toronto
- **Person**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Llion Jones, etc.
- **Technology**: Transformer, Attention mechanisms
- **Task**: WMT 2014 English-to-German Translation, English Parsing

**Relationships**:
- ATTRIBUTION_REQUIRED (Publication → Organization)
- GRANTS (Organization → Permission)
- RELATED_TO (Entity → Entity)

**Total Nodes**: 16+ per page × multiple pages

---

## Performance Interpretation Guidelines

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| F1 Score | > 0.85 | 0.65-0.85 | 0.5-0.65 | < 0.5 |
| Hallucination Rate | < 0.15 | 0.15-0.25 | 0.25-0.40 | > 0.40 |
| Semantic Similarity | > 0.75 | 0.60-0.75 | 0.45-0.60 | < 0.45 |
| ROUGE Score | > 0.4 | 0.2-0.4 | 0.1-0.2 | < 0.1 |

---

## Dependencies

**Core**:
- langchain, langchain-core, langchain-community
- chromadb (vector DB)
- sentence-transformers (semantic similarity)
- torch (for embeddings)

**Optional** (for enhanced metrics):
- nltk (natural language processing)
- rouge-score (ROUGE metric)

**Install**:
```bash
pip install -r requirements.txt
```

---

## File Modification History

| File | Change | Date | Reason |
|------|--------|------|--------|
| `src/graphrag/utils/data_retriever.py` | Created | 2026-03-10 | Dynamic node retrieval from graph data |
| `experiments/comprehensive_evaluation.py` | Updated | 2026-03-10 | Integrated data_retriever, fixed Unicode |
| `src/graphrag/retrieval.py` | Created | 2026-03-10 | Implemented retriever classes |
| `src/graphrag/__init__.py` | Updated | 2026-03-10 | Fixed import paths |
| EVALUATION_GUIDE.md | Created | 2026-03-10 | User guide for running evaluations |

---

## Current Status & Next Steps

✅ **Completed**:
- Data retriever module integrating actual graph nodes
- Benchmark dataset with real Transformer questions
- Comprehensive evaluation framework
- Multimodal ablation study support
- Results JSON output structure

⏳ **Running**:
- comprehensive_evaluation.py (baseline + multimodal tests)

📋 **To Do** (Optional):
- Implement actual LLM answer generation (currently using placeholders)
- Add Neo4j database connection for live graph queries
- Implement actual ROUGE/BERTScore if nltk missing
- Add visualization dashboard for results
- Implement multi-hop graph traversal

---

## Quick Reference Commands

```bash
# Run full evaluation
cd d:\Graph_RAG
python experiments\comprehensive_evaluation.py

# View results
cat results\comprehensive_evaluation.json

# View logs
tail -f results\experiment_comprehensive_eval_*.log

# Check graph data nodes
python -c "
import json
with open('data/preprocessed/graph_data.json') as f:
    data = json.load(f)
    for page in data:
        print(f'Page nodes: {len(page[\"nodes\"])}')"

# Run with specific Python
.\.venv\Scripts\python.exe experiments\comprehensive_evaluation.py

# Install dependencies
pip install -r requirements.txt
```

---

## Contact & Attribution

**Authors**: Arnav Deshpande, Sarvesh Nimbalkar, Aadi Rawat, Dhruv Gadia
**Institution**: Mukesh Patel School of Technology and Management, NMIMS, Mumbai
**License**: MIT License
**Email**: deshpandearnavn@gmail.com
**Last Updated**: 2026-03-10 18:11:43
