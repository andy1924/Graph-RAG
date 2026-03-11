"""
Compatibility wrapper for multimodal graph retrieval.

This module re-exports the MultimodalGraphRetriever class from src/graphrag/retrieval.py
for backward compatibility with existing scripts.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graphrag.retrieval import MultimodalGraphRetriever


def get_answer_from_context(retriever, question: str, context: str) -> str:
    """Helper function to generate answer from context using the retriever's internal client."""
    from graphrag.retrieval import ask_llm_with_context
    return ask_llm_with_context(question, context, retriever.client)


def main() -> None:
    """
    Interactive query interface for multimodal graph retrieval.
    
    Demonstrates the full RAG pipeline with source citations:
    1. Retrieve entities and relationships from knowledge graph
    2. Display source citations for retrieved facts
    3. Generate and display answer based on retrieved context
    """
    retriever = MultimodalGraphRetriever()
    user_query = "What is positional encoding connected to?"

    print(f"Question: {user_query}\n")
    print("Retrieving context from knowledge graph...")
    context, sources = retriever.get_multimodal_context(user_query)

    if context and sources:
        print("\n[+] SOURCES FROM KNOWLEDGE GRAPH:")
        for source in sources:
            print(source)
        
        print("\n[+] Retrieved Context Summary:")
        print(context)
        
        print("\n[+] Generating Answer...")
        answer = get_answer_from_context(retriever, user_query, context)
        print(f"\n[+] ANSWER:\n{answer}")
    else:
        print("No relevant graph nodes found.")

    retriever.close()


if __name__ == "__main__":
    main()