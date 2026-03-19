"""
Multimodal PDF ingestion pipeline for NaiveRAG.

Pipeline:
  1. Extract text blocks per page          (pdfplumber)
  2. Extract tables and serialise to MD    (pdfplumber)
  3. Extract embedded images, caption via  (PyMuPDF + GPT-4o-mini vision)
  4. Chunk text with RecursiveCharacterTextSplitter (LangChain)
  5. Embed chunks and persist in ChromaDB  (LangChain + OpenAIEmbeddings)
"""

import base64
import hashlib
import io
import logging
import os
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from .config import NaiveRAGConfig, config as default_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text_and_tables(pdf_path: str) -> List[Document]:
    """Extract text blocks and tables from every page using pdfplumber."""
    docs: List[Document] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # --- plain text ---
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num,
                        "modality": "text",
                    }
                ))

            # --- tables → markdown ---
            for table_idx, table in enumerate(page.extract_tables(), start=1):
                if not table:
                    continue
                # Build a simple markdown table
                rows = []
                for row_i, row in enumerate(table):
                    cells = [str(c or "").strip() for c in row]
                    rows.append("| " + " | ".join(cells) + " |")
                    if row_i == 0:                        # header separator
                        rows.append("|" + "|".join(["---"] * len(cells)) + "|")
                md_table = "\n".join(rows)
                if md_table.strip():
                    docs.append(Document(
                        page_content=md_table,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num,
                            "modality": "table",
                            "table_index": table_idx,
                        }
                    ))

    logger.info(f"Extracted {len(docs)} text/table documents from {pdf_path}")
    return docs


def _caption_image(image_bytes: bytes, client: OpenAI, model: str) -> str:
    """Send an image to GPT-4o-mini vision and return its caption."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "low",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image concisely in 1-3 sentences. "
                                "Focus on any diagrams, equations, or charts shown."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Image captioning failed: {e}")
        return ""


def _extract_images(
    pdf_path: str,
    client: OpenAI,
    cfg: NaiveRAGConfig,
) -> List[Document]:
    """Extract embedded images from PDF pages and caption them with vision LLM."""
    docs: List[Document] = []
    seen_hashes: set = set()

    pdf_doc = fitz.open(pdf_path)
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                base_image = pdf_doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = base_image["image"]
            # Deduplicate identical images (e.g. logos)
            digest = hashlib.md5(image_bytes).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            # Skip tiny images (icons / decorations)
            if len(image_bytes) < 5000:
                continue

            caption = ""
            if cfg.caption_images:
                caption = _caption_image(image_bytes, client, cfg.vision_model)

            if caption:
                docs.append(Document(
                    page_content=f"[Image on page {page_num + 1}]: {caption}",
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "modality": "image",
                        "image_xref": xref,
                    }
                ))

    pdf_doc.close()
    logger.info(f"Extracted {len(docs)} image documents from {pdf_path}")
    return docs


def _chunk_documents(
    docs: List[Document],
    cfg: NaiveRAGConfig,
) -> List[Document]:
    """Split text/table documents into smaller chunks; images are kept as-is."""
    text_docs = [d for d in docs if d.metadata.get("modality") != "image"]
    image_docs = [d for d in docs if d.metadata.get("modality") == "image"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunked = splitter.split_documents(text_docs)
    all_chunks = chunked + image_docs
    logger.info(
        f"Chunking: {len(text_docs)} text/table docs → {len(chunked)} chunks "
        f"+ {len(image_docs)} image docs = {len(all_chunks)} total"
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str, cfg: NaiveRAGConfig = None) -> int:
    """
    Ingest a single PDF into ChromaDB.

    Extracts text, tables, and images (with captions), chunks the content,
    embeds it with OpenAI, and persists everything in a local ChromaDB collection.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        cfg:      NaiveRAGConfig instance; uses module default if not provided.

    Returns:
        Number of document chunks stored.
    """
    cfg = cfg or default_config
    pdf_path = str(Path(pdf_path).resolve())

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = OpenAI(api_key=cfg.openai_api_key)

    # --- 1. Extract multimodal content ---
    logger.info(f"Ingesting PDF: {pdf_path}")
    text_table_docs = _extract_text_and_tables(pdf_path)
    image_docs = _extract_images(pdf_path, client, cfg)
    all_raw = text_table_docs + image_docs

    # --- 2. Chunk ---
    chunks = _chunk_documents(all_raw, cfg)
    if not chunks:
        logger.warning("No content extracted — nothing to store.")
        return 0

    # --- 3. Embed + store in ChromaDB ---
    embeddings = OpenAIEmbeddings(
        model=cfg.embedding_model,
        openai_api_key=cfg.openai_api_key,
    )

    os.makedirs(cfg.chroma_dir, exist_ok=True)
    vectorstore = Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
        persist_directory=cfg.chroma_dir,
    )

    vectorstore.add_documents(chunks)
    logger.info(
        f"Stored {len(chunks)} chunks in ChromaDB collection "
        f"'{cfg.collection_name}' at '{cfg.chroma_dir}'"
    )
    return len(chunks)


def ingest_directory(directory: str = None, cfg: NaiveRAGConfig = None) -> dict:
    """
    Ingest all PDFs in a directory.

    Args:
        directory: Path to folder containing PDFs. Defaults to cfg.pdf_dir.
        cfg:       NaiveRAGConfig instance.

    Returns:
        Dict mapping filename → chunk count.
    """
    cfg = cfg or default_config
    directory = directory or cfg.pdf_dir
    results = {}

    pdf_files = list(Path(directory).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {directory}")
        return results

    for pdf_file in pdf_files:
        try:
            count = ingest_pdf(str(pdf_file), cfg)
            results[pdf_file.name] = count
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_file.name}: {e}")
            results[pdf_file.name] = 0

    return results


def ingest_text_file(text_path: str, cfg: NaiveRAGConfig = None) -> int:
    """
    Ingest a plain text file into ChromaDB.

    Reads the file, chunks it, embeds with OpenAI, and stores in ChromaDB.

    Args:
        text_path: Path to the .txt file.
        cfg:       NaiveRAGConfig instance; uses module default if not provided.

    Returns:
        Number of document chunks stored.
    """
    cfg = cfg or default_config
    text_path = str(Path(text_path).resolve())

    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")

    logger.info(f"Ingesting text file: {text_path}")

    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if not content.strip():
        logger.warning(f"Empty file: {text_path}")
        return 0

    # Split into page-like sections by double newlines
    sections = [s.strip() for s in content.split("\n\n") if s.strip()]
    docs = []
    for i, section in enumerate(sections):
        docs.append(Document(
            page_content=section,
            metadata={
                "source": os.path.basename(text_path),
                "section": i + 1,
                "modality": "text",
            }
        ))

    # Chunk
    chunks = _chunk_documents(docs, cfg)
    if not chunks:
        logger.warning("No content extracted — nothing to store.")
        return 0

    # Embed + store
    embeddings = OpenAIEmbeddings(
        model=cfg.embedding_model,
        openai_api_key=cfg.openai_api_key,
    )

    os.makedirs(cfg.chroma_dir, exist_ok=True)
    vectorstore = Chroma(
        collection_name=cfg.collection_name,
        embedding_function=embeddings,
        persist_directory=cfg.chroma_dir,
    )

    vectorstore.add_documents(chunks)
    logger.info(
        f"Stored {len(chunks)} chunks from '{os.path.basename(text_path)}' "
        f"in ChromaDB collection '{cfg.collection_name}'"
    )
    return len(chunks)
