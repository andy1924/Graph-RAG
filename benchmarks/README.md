# Benchmarks

This directory contains benchmark datasets for evaluating GraphRAG system performance.

## Dataset Structure

Each benchmark dataset should have:

```
dataset_name/
├── documents/          # Raw source documents
│   ├── doc1.txt
│   ├── doc2.pdf
│   └── ...
│
├── questions.json      # Evaluation questions
└── references.json     # Ground truth answers
```

### Questions Format

```json
{
  "questions": [
    {
      "id": "q1",
      "text": "Question text here?",
      "category": "fact_finding|comparison|reasoning",
      "difficulty": "easy|medium|hard"
    }
  ]
}
```

### References Format

```json
{
  "references": [
    {
      "question_id": "q1",
      "answer": "Ground truth answer text",
      "relevant_docs": ["doc1", "doc2"],
      "entities_mentioned": ["Entity1", "Entity2"]
    }
  ]
}
```

## Available Benchmarks

### Coming Soon
- Tech Paper Benchmark: Questions over computer science papers
- Product Documentation Benchmark: Questions over product manuals
- News Archive Benchmark: Questions over news articles

## Creating Custom Benchmarks

```python
from pathlib import Path
import json

# Organize your documents
dataset_dir = Path("benchmarks/my_dataset")
dataset_dir.mkdir(exist_ok=True)
(dataset_dir / "documents").mkdir(exist_ok=True)

# Create questions
questions = {
    "questions": [
        {
            "id": "q1",
            "text": "Your question?",
            "category": "fact_finding",
            "difficulty": "medium"
        }
    ]
}

with open(dataset_dir / "questions.json", 'w') as f:
    json.dump(questions, f)

# Create reference answers
references = {
    "references": [
        {
            "question_id": "q1",
            "answer": "Reference answer",
            "relevant_docs": ["doc1"],
            "entities_mentioned": ["Entity1"]
        }
    ]
}

with open(dataset_dir / "references.json", 'w') as f:
    json.dump(references, f)

# Copy your documents
# cp your_documents/* dataset_dir/documents/
```

## Benchmark Evaluation

```bash
# Evaluate on a specific benchmark
python experiments/benchmark_evaluation.py --dataset benchmarks/my_dataset

# Compare across multiple benchmarks
python experiments/compare_benchmarks.py --benchmarks "dataset1,dataset2,dataset3"
```

---

See [../docs/EVALUATION.md](../docs/EVALUATION.md) for evaluation methodology.
