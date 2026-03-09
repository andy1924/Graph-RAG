"""
Graph-based Question Answer System with Knowledge Retrieval and Citations

This module implements a Retrieval-Augmented Generation (RAG) system that:
1. Retrieves relevant entity information from a Neo4j knowledge graph
2. Summarizes relationships into concise facts
3. Generates answers based on graph context with source citations

The system is designed for research-grade question answering over structured
knowledge graphs, with full traceability of information sources.

Author: GraphRAG Research Team
Date: 2026
"""

import os
from typing import Tuple, List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


def get_graph_context(user_query: str, client, driver, database: str) -> Tuple[str, List[str]]:
    """
    Retrieve relevant context from Neo4j knowledge graph with source attribution.
    
    Process:
    1. Fetches all entity names from the graph
    2. Uses LLM to select entities most relevant to the query
    3. Extracts relationships between selected entities
    4. Summarizes facts and returns with source citations
    
    Args:
        user_query (str): User's question to answer
        client: OpenAI API client for entity selection and summarization
        driver: Neo4j database driver
        database (str): Target database name in Neo4j
        
    Returns:
        Tuple[str, List[str]]: 
            - Summarized context facts from the knowledge graph
            - List of source citations showing entities/relationships used
            
    Raises:
        Returns error message tuple if graph query fails
    """
    try:
        with driver.session(database=database) as session:
            # Step A: Get all entity names from graph
            all_entities = session.run("MATCH (n) WHERE '__Entity__' IN labels(n) RETURN n.id LIMIT 500")
            entity_list = [row['n.id'] for row in all_entities]
            
            if not entity_list:
                return "No entities found in the knowledge graph.", []
            
            # Show first 150 entities to LLM for selection
            entity_context = ", ".join(entity_list[:150])
            
            # Step B: Ask LLM which entities are relevant to the query
            select_res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": """You are analyzing a question about a knowledge graph.
Given a question and a list of exact entity names, return the 3-5 most relevant entities.
Return ONLY the entity names, comma-separated. If none are relevant, return 'NONE'."""},
                    {"role": "user", "content": f"Question: {user_query}\n\nEntity names: {entity_context}"}
                ]
            )
            
            selected_entities_str = select_res.choices[0].message.content.strip()
            if selected_entities_str == "NONE":
                return "No relevant entities found for this question.", []
            
            selected_entities = [e.strip() for e in selected_entities_str.split(",")]
            
            # Step C: Query Neo4j for exact entity matches and their relationships
            relations = []
            sources = []  # Track source citations
            cypher = """
            MATCH (n)
            WHERE n.id = $entity
            OPTIONAL MATCH (n)-[r]-(neighbor)
            RETURN n.id AS source, labels(n)[0] AS source_type, type(r) AS rel, neighbor.id AS target, labels(neighbor)[0] AS target_type
            LIMIT 10
            """
            
            for entity in selected_entities:
                results = list(session.run(cypher, {"entity": entity}))
                for row in results:
                    if row.get('rel') and row.get('target'):
                        rel_str = f"{row['source']} ({row.get('source_type', 'Unknown')}) {row['rel']} {row['target']} ({row.get('target_type', 'Unknown')})"
                        relations.append(rel_str)
                        sources.append(f"  • {row['source']} -[{row['rel']}]-> {row['target']}")
                    elif row.get('source'):
                        rel_str = f"{row['source']} is a {row.get('source_type', 'Unknown')}"
                        relations.append(rel_str)
                        sources.append(f"  • {row['source']} ({row.get('source_type', 'Unknown')})")
            
            # Step D: Summarize relationships into concise facts
            if relations:
                summary_prompt = f"""Summarize these knowledge graph facts into 2-3 concise, factual statements relevant to answering: "{user_query}"
Facts:
{chr(10).join(relations)}

Provide ONLY the summary statements, no preamble."""
                
                summary_res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a concise information summarizer. Extract key facts."},
                        {"role": "user", "content": summary_prompt}
                    ]
                )
                context = summary_res.choices[0].message.content.strip()
                return context, sources
            else:
                return "Entities found but no relationships.", []
    
    except Exception as e:
        return f"Error querying graph: {str(e)}", []


def ask_llm_with_context(user_query: str, context: str, client: OpenAI) -> str:
    """
    Generate answer based on retrieved graph context only.
    
    The LLM is constrained to answer ONLY based on the provided knowledge graph
    context. This ensures answers are grounded in verifiable sources and prevents
    hallucination with external knowledge.
    
    Args:
        user_query (str): Original user question
        context (str): Summarized facts retrieved from knowledge graph
        client (OpenAI): OpenAI API client for answer generation
        
    Returns:
        str: Answer grounded in the provided knowledge graph context
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": """You are answering questions based on ONLY the provided knowledge graph context.
Use ONLY the facts provided below. Do not use external knowledge.
Answer directly and specifically based on what the context shows.
If the context does not contain relevant information, say so explicitly."""},
            {"role": "user", "content": f"""Knowledge Graph Context:
{context}

Question: {user_query}

Answer based ONLY on the context above."""}
        ],
        temperature=0
    )
    return response.choices[0].message.content


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
        print("\n📊 SOURCES FROM KNOWLEDGE GRAPH:")
        for source in sources:
            print(source)

        # 2. Get answer from LLM
        print("\n🤖 Generating answer...")
        answer = ask_llm_with_context(query, context, client)

        print(f"\n✅ ANSWER:\n{answer}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()