"""
Multimodal Graph-based Question Answering System

This module extends the basic retrieval system to support multimodal queries
by retrieving context for various entity types and their relationships.

Features:
- Multi-entity context retrieval from Neo4j knowledge graph
- Source attribution for all retrieved facts
- LLM-based entity selection from knowledge graph
- Context summarization with proper citations

The system maintains the RAG paradigm with full traceability of information sources.

Author: GraphRAG Research Team
Date: 2026
"""

import os
from typing import Tuple, List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()


class MultimodalGraphRetriever:
    """
    Retriever for multimodal context from knowledge graphs.
    
    Handles entity selection, relationship extraction, and context summarization
    for research-grade question answering systems.
    """

    def __init__(self) -> None:
        """Initialize the retriever with OpenAI and Neo4j connections."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def get_multimodal_context(self, question: str) -> Tuple[str, List[str]]:
        """
        Retrieve multimodal context from the knowledge graph with source citations.
        
        Process:
        1. Fetches all entities from knowledge graph
        2. Uses LLM to select relevant entities for the question
        3. Extracts relationships for selected entities
        4. Summarizes facts and returns with source attribution
        
        Args:
            question (str): User's question to answer
            
        Returns:
            Tuple[str, List[str]]:
                - Summarized context facts from the knowledge graph
                - List of source citations (entity-relationship-entity triples)
        """
        try:
            with self.driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
                # Step 1: Get all entity names from the graph
                all_entities = session.run("MATCH (n) WHERE '__Entity__' IN labels(n) RETURN n.id LIMIT 500")
                entity_list = [row['n.id'] for row in all_entities]
                
                if not entity_list:
                    return "No entities found in knowledge graph.", []
                
                # Step 2: Ask LLM to select relevant entities
                entity_context = ", ".join(entity_list[:150])
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system",
                         "content": """You are analyzing a question about a knowledge graph from a research paper.
Given a question and a list of exact entity names, return the 3-5 most relevant entities.
Return ONLY the entity names, comma-separated. If none match, return 'NONE'."""},
                        {"role": "user", "content": f"Question: {question}\n\nEntities: {entity_context}"}
                    ]
                )
                
                selected_str = response.choices[0].message.content.strip()
                if selected_str == "NONE":
                    return "No relevant entities found for this question.", []
                
                selected_entities = [e.strip() for e in selected_str.split(",")]
                
                # Step 3: Get relationships for selected entities
                relations = []
                sources = []  # Track citations
                cypher = """
                MATCH (n)
                WHERE n.id = $entity
                OPTIONAL MATCH (n)-[r]-(neighbor)
                RETURN 
                    n.id AS source_id, 
                    labels(n)[0] AS source_type,
                    type(r) AS relationship,
                    neighbor.id AS target_id,
                    labels(neighbor)[0] AS target_type
                LIMIT 10
                """
                
                for entity in selected_entities:
                    results = list(session.run(cypher, {"entity": entity}))
                    
                    for record in results:
                        if record.get('source_id') and record.get('relationship') and record.get('target_id'):
                            fact = f"{record['source_id']} ({record.get('source_type')}) {record['relationship']} {record['target_id']} ({record.get('target_type')})"
                            relations.append(fact)
                            sources.append(f"  • {record['source_id']} -[{record['relationship']}]-> {record['target_id']}")
                        elif record.get('source_id'):
                            fact = f"{record['source_id']} is a {record.get('source_type')}"
                            relations.append(fact)
                            sources.append(f"  • {record['source_id']} ({record.get('source_type')})")
                
                # Step 4: Summarize relationships into condensed facts
                if relations:
                    summary_prompt = f"""Summarize these knowledge graph facts into 2-3 concise, factual statements relevant to: "{question}"
Facts:
{chr(10).join(relations)}

Provide ONLY the summary statements."""
                    
                    summary_response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a concise information summarizer. Extract key facts."},
                            {"role": "user", "content": summary_prompt}
                        ]
                    )
                    context = summary_response.choices[0].message.content.strip()
                    return context, sources
                else:
                    return "Entities found but no relationships.", []
        
        except Exception as e:
            return f"Error retrieving context: {str(e)}", []

    def get_answer_from_context(self, question: str, context: str) -> str:
        """
        Generate answer based on retrieved graph context only.
        
        The LLM is constrained to answer ONLY based on the provided knowledge graph
        context. This ensures answers are grounded in verifiable graph sources.
        
        Args:
            question (str): Original user question
            context (str): Summarized facts from knowledge graph
            
        Returns:
            str: Answer grounded in the provided knowledge graph context
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": """You are answering questions based on ONLY the provided knowledge graph context.
Use ONLY the facts provided below. Do not use external knowledge.
Answer directly and specifically based on what the context shows.
If the context does not contain relevant information, say so explicitly."""},
                {"role": "user", "content": f"""Knowledge Graph Context:
{context}

Question: {question}

Answer based ONLY on the context above."""}
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()


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
        print("\n📊 SOURCES FROM KNOWLEDGE GRAPH:")
        for source in sources:
            print(source)
        
        print("\n💭 Retrieved Context Summary:")
        print(context)
        
        print("\n🤖 Generating Answer...")
        answer = retriever.get_answer_from_context(user_query, context)
        print(f"\n✅ ANSWER:\n{answer}")
    else:
        print("No relevant graph nodes found.")

    retriever.close()


if __name__ == "__main__":
    main()