# How to Run GraphRAG Model Evaluation

## Summary of What We've Set Up

1. **Data Retriever Module** (`src/graphrag/utils/data_retriever.py`)
   - Automatically loads relevant graph nodes from your preprocessed `graph_data.json`
   - Maps evaluation questions to relevant items using keyword matching
   - Retrieves actual node IDs instead of hardcoded placeholders

2. **Benchmark Dataset** (in `comprehensive_evaluation.py`)
   - Updated with real questions based on your training data (Transformer architecture, attention mechanisms)
   - Updated with actual reference answers
   - Now retrieves `relevant_items` dynamically from your graph data

3. **Evaluation Scripts**
   - **comprehensive_evaluation.py**: Runs baseline + multimodal ablation studies
   - **multimodal_ablation.py**: Detailed multimodal component analysis

---

## Running the Evaluation

### Basic Command
```bash
cd d:\Graph_RAG
python experiments\comprehensive_evaluation.py
```

Or with explicit venv Python:
```bash
.\.venv\Scripts\python.exe experiments\comprehensive_evaluation.py
```

### What It Does

The evaluation runs:

1. **Baseline Experiment**
   - Tests standard graph retrieval on 5 questions
   - Measures: F1 Score, Hallucination Rate, Semantic Similarity, Response Time

2. **Multimodal Ablation Study**
   - Tests different modality combinations:
     - Text only
     - Text + Tables
     - Text + Tables + Images
     - Tables only
     - Images only
   - Compares performance across modalities

3. **Results**
   - Saves results to `results/comprehensive_evaluation.json`
   - Prints summary showing:
     - Baseline metrics
     - Best performing modality configuration
     - Performance comparison

---

## Key Metrics Measured

- **F1 Score**: Harmonic mean of precision and recall for retrieval
- **Hallucination Rate**: Percentage of unsupported claims in generated answers
- **Semantic Similarity**: How well answers match reference answers
- **Response Time**: Query latency
- **Modality Usage**: Breakdown of text/table/image contributions

---

## Customizing Evaluations

### Update Questions
Edit `BenchmarkDataset.__init__()` in `comprehensive_evaluation.py`:

```python
self.questions = [
    "Your custom question 1",
    "Your custom question 2",
    # ...
]

self.references = [
    "Ground truth answer 1",
    "Ground truth answer 2",
    # ...
]
```

### Custom Keyword Mapping
In `data_retriever.py`, update `QUESTION_KEYWORDS_MAPPING`:

```python
QUESTION_KEYWORDS_MAPPING = {
    "What are the main characteristics of the Transformer architecture?": 
        ["Transformer", "architecture", "attention"],
}
```

---

## Output Files

- **Console Output**: Real-time progress and metrics
- **JSON Results**: `results/comprehensive_evaluation.json` - detailed metrics per question
- **Log Files**: `results/experiment_*.log` - full execution logs

---

## Current Status

✓ Data retriever module created and integrated
✓ Benchmark dataset updated with real questions & dynamic item retrieval
✓ Evaluation running successfully
✓ Metrics being calculated (F1, Hallucination, Semantic Similarity)
✓ Results being saved to JSON file

The evaluation will complete and save results in the `results/` directory.

