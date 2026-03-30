"""
Configuration for the NaiveRAG pipeline.
Mirrors the style of src/graphrag/config.py.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class NaiveRAGConfig:
    """Settings for the NaiveRAG ingestion and retrieval pipeline."""

    # --- ChromaDB ---
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "naiverag_docs")

    # --- Embeddings / LLM ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o-mini")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- Ingestion ---
    pdf_dir: str = os.getenv("PDF_DIR", "data/multiModalPDF")
    caption_images: bool = os.getenv("CAPTION_IMAGES", "true").lower() == "true"

    # --- Retrieval ---
    top_k: int = int(os.getenv("TOP_K", "5"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))


# Global singleton
config = NaiveRAGConfig()
