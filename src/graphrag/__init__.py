"""
GraphRAG: Efficient Multi-Modal Graph-Based Retrieval-Augmented Generation
for Mitigating LLM Hallucinations

This package implements a research-oriented GraphRAG system optimized for
multi-modal queries across documents with text, tables, and images.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

try:
    from .ingestion import graph_generator, multimodal_ingestion
except ImportError as e:
    # Optional ingestion dependencies
    print(f"Warning: Could not import ingestion modules: {e}")
    graph_generator = None
    multimodal_ingestion = None

from .retrieval import GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever
from .evaluation import metrics

__all__ = [
    "graph_generator",
    "multimodal_ingestion",
    "graph_retriever",
    "multimodal_retriever",
    "metrics",
]
