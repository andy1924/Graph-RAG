# Research Goal and Current Status

## Project Metadata
- Project Title: Beyond Vector Search: Mitigating LLM Hallucinations via Graph-Based Retrieval-Augmented Generation (GraphRAG)
- Authors: Arnav Deshpande; Sarvesh Nimbalkar; Dhruv Gadia; Aadi Rawat
- Organization: Mukesh Patel School of Technology and Management, NMIMS University
- Contact Email: [deshpandearnavn@gmail.com](mailto:deshpandearnavn@gmail.com)
- GitHub Repository: https://github.com/andy1924/Graph-RAG

## Research Objective
The project investigates whether graph-structured retrieval can mitigate hallucination behavior in LLM question answering relative to a chunk-retrieval baseline.

## Experimental Objective
Evaluate GraphRAG and NaiveRAG on the same corpora and question sets, then assess:
- grounding behavior (hallucination rate)
- answer quality (semantic similarity, ROUGE, BERTScore)
- retrieval behavior (precision, recall, retrieval F1)
- runtime characteristics (average response time)

## Current Evidence from Repository Results
Based on results/comprehensive_evaluation.json, results/naiverag_evaluation.json, and results/significance_analysis.json:
- GraphRAG shows lower aggregate hallucination rate (0.0033) than NaiveRAG (0.0204).
- NaiveRAG shows higher semantic and lexical similarity metrics in the current setup.
- Multi-metric significance testing reports statistically significant differences for hallucination, semantic similarity, ROUGE, and BERTScore.

## Methodological Caveat
Retrieval F1 is not currently cross-system comparable because GraphRAG and NaiveRAG use different retrieval targets in evaluation. The significance report explicitly labels this and omits inferential testing for retrieval F1 until harmonized definitions are implemented.

## Limitations
- Retrieval metric non-equivalence across systems.
- Dependence on external API behavior and model configuration.
- Result sensitivity to corpus-specific entity distributions and question design.

## Reproducibility Plan
1. Ingest all corpora.
2. Run comprehensive GraphRAG evaluation.
3. Run NaiveRAG evaluation.
4. Run significance analysis.
5. Generate visual outputs.

Commands:
```bash
python main.py ingest --all
python main.py evaluate --experiment comprehensive
python main.py evaluate --experiment naiverag
python main.py evaluate --experiment significance
python experiments/visualize_results.py
```

## Next Methodological Step
Implement a harmonized retrieval target for both systems to enable valid inferential testing on retrieval F1.

