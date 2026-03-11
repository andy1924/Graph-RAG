"""
NaiveRAG — standard vector RAG pipeline with multimodal PDF support.

Public API:
    from naiverag import ingest_pdf, NaiveRAGRetriever
"""

from .ingestion import ingest_pdf, ingest_directory
from .retrieval import NaiveRAGRetriever
from .config import NaiveRAGConfig, config

__all__ = [
    "ingest_pdf",
    "ingest_directory",
    "NaiveRAGRetriever",
    "NaiveRAGConfig",
    "config",
]
