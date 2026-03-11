"""
Compatibility wrapper for GraphRAG retrieval module.

This module re-exports the functional retrieval classes and functions from src/graphrag/retrieval.py
for backward compatibility with existing scripts.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graphrag.retrieval import (
    GraphRetriever,
    SemanticGraphRetriever,
    MultimodalGraphRetriever,
    get_graph_context,
    ask_llm_with_context
)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main() -> None:
    """
    Main interactive query interface for the GraphRAG system.
    
    Implements the full RAG pipeline:
    1. Accept user query
    2. Retrieve relevant entities and relationships from Neo4j
    3. Display source citations from the knowledge graph
    4. Generate and display answer based on retrieved context
    
    Exits gracefully on user interruption or errors.
    """
    from neo4j import GraphDatabase
    
    # Initialize Clients
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )

    try:
        query = input("Ask your graph a question: ")

        # 1. Get data from Neo4j
        print("\nSearching graph...")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        context, sources = get_graph_context(query, client, driver, database)

        if not context:
            print("No matching relationships found in the graph.")
            return

        # Print sources for transparency
        print("\n[+] SOURCES FROM KNOWLEDGE GRAPH:")
        for source in sources:
            print(source)

        # 2. Get answer from LLM
        print("\n[+] Generating answer...")
        answer = ask_llm_with_context(query, context, client)

        print(f"\n[+] ANSWER:\n{answer}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()