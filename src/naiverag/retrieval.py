"""
Retrieval pipeline for NaiveRAG.

Provides NaiveRAGRetriever using the modern LangChain LCEL pipeline
(langchain_core + langchain_openai) — compatible with LangChain >= 0.3 / 1.x.

The class intentionally mirrors the GraphRetriever interface so both can
be passed into the same evaluation harness.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import NaiveRAGConfig, config as default_config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are answering questions based ONLY on the provided context excerpts "
    "from a knowledge base. Use ONLY the facts in the context — do not use "
    "external knowledge. If the context does not contain enough information "
    "to answer, say so explicitly.\n\n"
    "Context:\n{context}"
)

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{question}"),
])


class NaiveRAGRetriever:
    """
    Vector-based retriever backed by ChromaDB.

    Interface deliberately mirrors GraphRetriever so both can be
    passed into the same evaluation harness.
    """

    def __init__(self, cfg: NaiveRAGConfig = None):
        """
        Load the persisted ChromaDB collection and initialise the LLM chain.

        Args:
            cfg: NaiveRAGConfig instance; uses module default if not provided.
        """
        self.cfg = cfg or default_config

        self.embeddings = OpenAIEmbeddings(
            model=self.cfg.embedding_model,
            openai_api_key=self.cfg.openai_api_key,
        )

        self.vectorstore = Chroma(
            collection_name=self.cfg.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.cfg.chroma_dir,
        )

        self.llm = ChatOpenAI(
            model=self.cfg.llm_model,
            temperature=self.cfg.temperature,
            openai_api_key=self.cfg.openai_api_key,
        )

        self._retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.cfg.top_k}
        )

        # LCEL chain:  question → retrieve docs → format context → LLM → string
        self._chain = (
            {
                "context": self._retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | _QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        logger.info(
            f"NaiveRAGRetriever ready — collection='{self.cfg.collection_name}', "
            f"top_k={self.cfg.top_k}, llm='{self.cfg.llm_model}'"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_docs(docs) -> str:
        """Concatenate retrieved document content into a single context string."""
        return "\n\n".join(d.page_content for d in docs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_context(
        self, question: str, k: int = None
    ) -> Tuple[str, List[str], List[str]]:
        """
        Retrieve the top-k most relevant chunks for a question.

        Args:
            question: Natural-language query.
            k:        Number of chunks (overrides config default).

        Returns:
            Tuple of (merged context string, source citation strings,
                       list of retrieved chunk IDs).
        """
        k = k or self.cfg.top_k
        docs = self.vectorstore.similarity_search(question, k=k)

        context = "\n\n".join(d.page_content for d in docs)
        sources = [
            f"  \u2022 [{d.metadata.get('modality', 'text')}] "
            f"{d.metadata.get('source', 'unknown')} \u2014 page {d.metadata.get('page', '?')}"
            for d in docs
        ]
        # Collect unique chunk identifiers for F1 evaluation
        retrieved_ids = []
        for d in docs:
            # Prefer explicit 'id' metadata, fall back to source+page combo
            chunk_id = d.metadata.get(
                "id",
                f"{d.metadata.get('source', 'unknown')}_p{d.metadata.get('page', '?')}_{d.metadata.get('modality', 'text')}"
            )
            if chunk_id not in retrieved_ids:
                retrieved_ids.append(chunk_id)
        return context, sources, retrieved_ids

    def answer_question(
        self, question: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Answer a question using the ChromaDB vector store + LLM (LCEL pipeline).

        Compatible with GraphRetriever.answer_question():
        returns (answer_str, metadata_dict) where metadata contains
        'context', 'sources', 'response_time', and 'num_chunks_retrieved'.

        Args:
            question: Natural-language question.

        Returns:
            Tuple of (answer, metadata).
        """
        try:
            # Retrieve docs for metadata / context string
            retrieved_docs = self._retriever.invoke(question)

            context = "\n\n".join(d.page_content for d in retrieved_docs)
            sources = [
                f"  • [{d.metadata.get('modality', 'text')}] "
                f"{d.metadata.get('source', 'unknown')} — page {d.metadata.get('page', '?')}"
                for d in retrieved_docs
            ]

            # Time the full LLM call
            start_time = time.time()
            answer = self._chain.invoke(question)
            response_time = time.time() - start_time

            metadata: Dict[str, Any] = {
                "context": context,
                "sources": sources,
                "response_time": response_time,
                "num_chunks_retrieved": len(retrieved_docs),
            }

            logger.info(
                f"answer_question completed in {response_time:.3f}s "
                f"({len(retrieved_docs)} chunks retrieved)"
            )
            return answer, metadata

        except Exception as e:
            logger.error(f"answer_question failed: {e}")
            return (
                f"Error: {e}",
                {"context": "", "sources": [], "response_time": 0.0,
                 "num_chunks_retrieved": 0},
            )
