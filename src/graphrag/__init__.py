"""
GraphRAG: Efficient Multi-Modal Graph-Based Retrieval-Augmented Generation
for Mitigating LLM Hallucinations

This package implements a research-oriented GraphRAG system optimized for
multi-modal queries across documents with text, tables, and images.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .ingestion import graph_generator, multimodal_ingestion
from .retrieval import GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever
from .evaluation import metrics

__all__ = [
    "graph_generator",
    "multimodal_ingestion",
    "graph_retriever",
    "multimodal_retriever",
    "metrics",
]
