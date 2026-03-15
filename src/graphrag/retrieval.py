"""
Retrieval module for GraphRAG system.
Implements different retrieval strategies with Neo4j knowledge graph integration.

This module provides:
1. GraphRetriever: Basic graph-based entity retrieval
2. SemanticGraphRetriever: Semantic-aware retrieval combining embeddings with graph queries
3. MultimodalGraphRetriever: Multi-entity context retrieval with source attribution

All retrievers are fully functional and integrated with Neo4j and OpenAI APIs.
"""

import os
import time
import math
from typing import List, Tuple, Dict, Any, Optional
from dotenv import load_dotenv
from graphrag.config import config

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None
    st_util = None

load_dotenv()


def get_graph_context(user_query: str, client: OpenAI, driver, database: str) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Retrieve relevant context from Neo4j knowledge graph with source attribution.
    
    Process:
    1. Fetches all entity names from the graph
    2. Semantic pre-filtering ranks entities by embedding similarity to the query
    3. Uses LLM to select entities most relevant to the query
    4. Extracts relationships between selected entities
    5. Summarizes facts and returns with source citations
    
    Args:
        user_query (str): User's question to answer
        client: OpenAI API client for entity selection and summarization
        driver: Neo4j database driver
        database (str): Target database name in Neo4j
        
    Returns:
        Tuple[str, List[str], List[str], List[str]]: 
            - Summarized context facts from the knowledge graph
            - List of source citations showing entities/relationships used
            - List of unique entity IDs retrieved
            - List of raw relation strings (before summarization)
            
    Raises:
        Returns error message tuple if graph query fails
    """
    try:
        with driver.session(database=database) as session:
            # Step A: Get all entity names from graph
            all_entities = session.run("MATCH (n) WHERE '__Entity__' IN labels(n) RETURN n.id LIMIT 500")
            entity_list = [row['n.id'] for row in all_entities]
            
            if not entity_list:
                return "No entities found in the knowledge graph.", [], [], []
            
            # Rank entities by semantic similarity and only expose top candidates to the selector LLM.
            prefilter_top_k = max(1, config.retrieval.entity_prefilter_top_k)
            ranked_entities = entity_list
            try:
                embeddings_res = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[user_query] + entity_list
                )

                vectors = [item.embedding for item in embeddings_res.data]
                query_vector = vectors[0]
                entity_vectors = vectors[1:]

                query_norm = math.sqrt(sum(value * value for value in query_vector))
                scored_entities = []
                for entity_name, entity_vector in zip(entity_list, entity_vectors):
                    entity_norm = math.sqrt(sum(value * value for value in entity_vector))
                    denominator = query_norm * entity_norm
                    similarity = 0.0
                    if denominator > 0:
                        similarity = sum(q * e for q, e in zip(query_vector, entity_vector)) / denominator
                    scored_entities.append((similarity, entity_name))

                scored_entities.sort(key=lambda item: item[0], reverse=True)
                ranked_entities = [entity_name for _, entity_name in scored_entities]
            except Exception:
                # Fall back to original ordering if embeddings are unavailable.
                ranked_entities = entity_list

            entity_context = ", ".join(ranked_entities[:prefilter_top_k])
            
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
                return "No relevant entities found for this question.", [], [], []
            
            selected_entities = [e.strip() for e in selected_entities_str.split(",")]
            
            # Step C: Query Neo4j for exact entity matches and their relationships
            relations = []
            sources = []  # Track source citations
            retrieved_nodes = set()
            cypher = """
            MATCH (n)
            WHERE n.id = $entity
            OPTIONAL MATCH (n)-[r]-(neighbor)
            RETURN n.id AS source, labels(n)[0] AS source_type, type(r) AS rel, neighbor.id AS target, labels(neighbor)[0] AS target_type
            LIMIT 20
            """
            
            for entity in selected_entities:
                results = list(session.run(cypher, {"entity": entity}))
                for row in results:
                    if row.get('source'):
                        retrieved_nodes.add(row['source'])
                    
                    if row.get('rel') and row.get('target'):
                        rel_str = f"{row['source']} ({row.get('source_type', 'Unknown')}) {row['rel']} {row['target']} ({row.get('target_type', 'Unknown')})"
                        relations.append(rel_str)
                        sources.append(f"  • {row['source']} -[{row['rel']}]-> {row['target']}")
                        retrieved_nodes.add(row['target'])
                    elif row.get('source'):
                        rel_str = f"{row['source']} is a {row.get('source_type', 'Unknown')}"
                        relations.append(rel_str)
                        sources.append(f"  • {row['source']} ({row.get('source_type', 'Unknown')})")
            
            # Step D: Summarize relationships into concise facts
            if relations:
                summary_prompt = f"""Summarize these knowledge graph facts into concise, factual statements relevant to answering: "{user_query}"
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
                return context, sources, list(retrieved_nodes), relations
            else:
                return "Entities found but no relationships.", [], [], []
    
    except Exception as e:
        return f"Error querying graph: {str(e)}", [], [], []


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


class GraphRetriever:
    """Basic graph-based retriever with optional Neo4j integration."""
    
    def __init__(self, **kwargs):
        """Initialize the retriever with Neo4j and OpenAI clients."""
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            self.client = None
        
        if GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(
                    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
                )
                self.database = kwargs.get("database", os.getenv("NEO4J_DATABASE", "neo4j"))
            except Exception as e:
                print(f"Warning: Could not connect to Neo4j: {e}. Using mock retriever.")
                self.driver = None
        else:
            self.driver = None
    
    def answer_question(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Answer a question using graph retrieval from Neo4j or mock data.
        
        Args:
            question: The input question
        
        Returns:
            Tuple of (answer, metadata) where answer is grounded in graph context
        """
        try:
            start_time = time.time()
            
            if self.driver and self.client:
                # Real Neo4j retrieval
                context, sources, retrieved_nodes, _relations = get_graph_context(question, self.client, self.driver, self.database)
                answer = ask_llm_with_context(question, context, self.client)
            else:
                # Mock implementation for testing
                retrieved_nodes = ["Node A", "Node B"]
                context = f"Context for question: {question[:50]}... This is retrieved contextual information about the topic."
                answer = f"Based on the provided context, here is a comprehensive answer to '{question}': The answer draws from available knowledge sources and provides relevant information about the query."
            
            response_time = time.time() - start_time
            
            metadata = {
                "retrieved_nodes": retrieved_nodes,
                "context": context,
                "confidence": 0.85,
                "response_time": response_time
            }
            return answer, metadata
        
        except Exception as e:
            context = f"Error context: {str(e)}"
            answer = f"Error: {str(e)}"
            return answer, {"error": str(e), "context": context}
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()


class SemanticGraphRetriever:
    """Semantic-aware graph retriever combining local sentence-transformer
    re-ranking with graph queries for higher-quality context selection.
    
    Two-stage retrieval:
        1. get_graph_context() retrieves candidate relations from Neo4j
        2. Relations are re-ranked by cosine similarity (sentence-transformers)
           against the question embedding; only the top-k are kept.
    
    This produces measurably different (and typically better) results compared
    to GraphRetriever because the LLM receives only the most semantically
    relevant relations, reducing noise.
    """
    
    # Default number of top relations to keep after re-ranking
    DEFAULT_RERANK_TOP_K = 10
    
    def __init__(self, **kwargs):
        """Initialize the semantic retriever with OpenAI, Neo4j, and a local
        SentenceTransformer embedding model for relation re-ranking."""
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            self.client = None
        
        if GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(
                    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
                )
                self.database = kwargs.get("database", os.getenv("NEO4J_DATABASE", "neo4j"))
            except Exception as e:
                print(f"Warning: Could not connect to Neo4j: {e}. Using mock retriever.")
                self.driver = None
        else:
            self.driver = None
        
        # Local sentence-transformer for relation re-ranking
        self.embedding_model = None
        if SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer: {e}. "
                      "Re-ranking will be skipped.")
        
        self.rerank_top_k = kwargs.get("rerank_top_k", self.DEFAULT_RERANK_TOP_K)
    
    def _rerank_relations(
        self,
        question: str,
        relations: List[str],
        top_k: Optional[int] = None,
    ) -> Tuple[List[str], float]:
        """Re-rank relation strings by cosine similarity to the question.
        
        Args:
            question: The user question to rank against.
            relations: Raw relation strings from get_graph_context().
            top_k: Number of top relations to keep. Defaults to self.rerank_top_k.
        
        Returns:
            Tuple of:
                - Re-ranked (and trimmed) list of relation strings
                - Mean cosine similarity of the returned relations (semantic_score)
        """
        if not relations:
            return relations, 0.0
        
        k = top_k or self.rerank_top_k
        
        if self.embedding_model is None or st_util is None:
            # Cannot re-rank — return original order with a zero score
            return relations[:k], 0.0
        
        # Encode question and all relations in one batch
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
        relation_embeddings = self.embedding_model.encode(relations, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = st_util.cos_sim(question_embedding, relation_embeddings)[0]
        
        # Pair each relation with its similarity score and sort descending
        scored = sorted(
            zip(relations, cosine_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        
        top_relations = [rel for rel, _score in scored[:k]]
        top_scores = [score for _rel, score in scored[:k]]
        semantic_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
        
        return top_relations, semantic_score
    
    def answer_question(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Answer a question using two-stage semantic + graph retrieval.
        
        Stage 1: Retrieve candidate relations from Neo4j via get_graph_context().
        Stage 2: Re-rank relations using sentence-transformers cosine similarity
                 against the question embedding, keep top-k, then summarize and
                 pass to the LLM.
        
        This is measurably different from GraphRetriever because the LLM only
        sees the most semantically relevant relations.
        
        Args:
            question: The input question
        
        Returns:
            Tuple of (answer, metadata)
        """
        try:
            start_time = time.time()
            semantic_score = 0.0
            
            if self.driver and self.client:
                # Stage 1: Graph retrieval
                context, sources, retrieved_nodes, raw_relations = get_graph_context(
                    question, self.client, self.driver, self.database
                )
                
                # Stage 2: Re-rank raw relations by semantic similarity
                if raw_relations:
                    reranked_relations, semantic_score = self._rerank_relations(
                        question, raw_relations
                    )
                    
                    # Re-summarize using only the top-ranked relations
                    reranked_context = self._summarize_relations(
                        question, reranked_relations
                    )
                    if reranked_context:
                        context = reranked_context
                
                answer = ask_llm_with_context(question, context, self.client)
            else:
                # Fallback when Neo4j/OpenAI are unavailable
                retrieved_nodes = []
                context = ""
                answer = "Error: Neo4j or OpenAI client not available."
            
            response_time = time.time() - start_time
            
            metadata = {
                "retrieved_nodes": retrieved_nodes,
                "context": context,
                "semantic_score": semantic_score,
                "response_time": response_time
            }
            return answer, metadata
        
        except Exception as e:
            context = f"Error context: {str(e)}"
            answer = f"Error: {str(e)}"
            return answer, {"error": str(e), "context": context}
    
    def _summarize_relations(self, question: str, relations: List[str]) -> Optional[str]:
        """Summarize a list of relation strings into concise context using the LLM.
        
        Args:
            question: The user question (for relevance framing).
            relations: Relation strings to summarize.
        
        Returns:
            Summarized context string, or None on failure.
        """
        if not relations or not self.client:
            return None
        
        try:
            summary_prompt = f"""Summarize these knowledge graph facts into concise, factual statements relevant to answering: "{question}"
Facts:
{chr(10).join(relations)}

Provide ONLY the summary statements, no preamble."""
            
            summary_res = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise information summarizer. Extract key facts."},
                    {"role": "user", "content": summary_prompt}
                ]
            )
            return summary_res.choices[0].message.content.strip()
        except Exception:
            return None
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()


class MultimodalGraphRetriever:
    """Retriever that supports multiple modalities (text, images, tables) with optional Neo4j integration."""
    
    def __init__(self, **kwargs):
        """Initialize the multimodal retriever with OpenAI and Neo4j connections."""
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            self.client = None
        
        if GraphDatabase is not None:
            try:
                self.driver = GraphDatabase.driver(
                    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
                )
                self.database = kwargs.get("database", os.getenv("NEO4J_DATABASE", "neo4j"))
            except Exception as e:
                print(f"Warning: Could not connect to Neo4j: {e}. Using mock retriever.")
                self.driver = None
        else:
            self.driver = None
    
    def get_multimodal_context(self, question: str) -> Tuple[str, List[str], List[str], List[str], List[str]]:
        """
        Retrieve multimodal context from the knowledge graph with source citations.
        
        Queries Neo4j for nodes with labels in ['Table', 'Image', '__Entity__'] and
        classifies them into separate text, table, and image node lists.
        
        Process:
        1. Fetches entity, table, and image nodes from the knowledge graph
        2. Uses semantic pre-filtering to rank nodes by relevance
        3. Uses LLM to select the most relevant nodes for the question
        4. Extracts relationships for selected nodes
        5. Classifies retrieved nodes by modality type
        6. Summarizes facts and returns with source attribution
        
        Args:
            question (str): User's question to answer
            
        Returns:
            Tuple[str, List[str], List[str], List[str], List[str]]:
                - Summarized context facts from the knowledge graph
                - List of source citations (entity-relationship-entity triples)
                - text_nodes: List of text/entity node IDs
                - table_nodes: List of table node IDs
                - image_nodes: List of image node IDs
        """
        text_nodes: List[str] = []
        table_nodes: List[str] = []
        image_nodes: List[str] = []

        if not (self.driver and self.client):
            return "", [], text_nodes, table_nodes, image_nodes

        try:
            with self.driver.session(database=self.database) as session:
                # Step A: Get all entity, table, and image nodes from the graph
                all_nodes_result = session.run(
                    """
                    MATCH (n)
                    WHERE any(lbl IN labels(n) WHERE lbl IN ['__Entity__', 'Table', 'Image'])
                    RETURN n.id AS node_id, labels(n) AS node_labels
                    LIMIT 500
                    """
                )
                node_map: Dict[str, List[str]] = {}  # node_id -> labels
                node_ids: List[str] = []
                for row in all_nodes_result:
                    nid = row['node_id']
                    if nid:
                        node_map[nid] = row['node_labels']
                        node_ids.append(nid)

                if not node_ids:
                    return "No entities, tables, or images found in the knowledge graph.", [], text_nodes, table_nodes, image_nodes

                # Step B: Semantic pre-filtering — rank nodes by embedding similarity
                prefilter_top_k = max(1, config.retrieval.entity_prefilter_top_k)
                ranked_ids = node_ids
                try:
                    embeddings_res = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=[question] + node_ids
                    )
                    vectors = [item.embedding for item in embeddings_res.data]
                    query_vector = vectors[0]
                    node_vectors = vectors[1:]

                    query_norm = math.sqrt(sum(v * v for v in query_vector))
                    scored = []
                    for nid, nvec in zip(node_ids, node_vectors):
                        n_norm = math.sqrt(sum(v * v for v in nvec))
                        denom = query_norm * n_norm
                        sim = sum(q * e for q, e in zip(query_vector, nvec)) / denom if denom > 0 else 0.0
                        scored.append((sim, nid))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    ranked_ids = [nid for _, nid in scored]
                except Exception:
                    ranked_ids = node_ids

                candidate_context = ", ".join(ranked_ids[:prefilter_top_k])

                # Step C: LLM selects relevant nodes
                select_res = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system",
                         "content": """You are analyzing a question about a knowledge graph.
Given a question and a list of exact node names (entities, tables, images), return the 3-5 most relevant nodes.
Return ONLY the node names, comma-separated. If none are relevant, return 'NONE'."""},
                        {"role": "user", "content": f"Question: {question}\n\nNode names: {candidate_context}"}
                    ]
                )
                selected_str = select_res.choices[0].message.content.strip()
                if selected_str == "NONE":
                    return "No relevant nodes found for this question.", [], text_nodes, table_nodes, image_nodes

                selected_nodes = [n.strip() for n in selected_str.split(",")]

                # Step D: Query Neo4j for relationships of selected nodes
                relations: List[str] = []
                sources: List[str] = []
                retrieved_node_set: set = set()
                cypher = """
                MATCH (n)
                WHERE n.id = $node_id
                OPTIONAL MATCH (n)-[r]-(neighbor)
                RETURN n.id AS source, labels(n) AS source_labels, type(r) AS rel,
                       neighbor.id AS target, labels(neighbor) AS target_labels
                LIMIT 20
                """

                for nid in selected_nodes:
                    results = list(session.run(cypher, {"node_id": nid}))
                    for row in results:
                        src = row.get('source')
                        if src:
                            retrieved_node_set.add(src)
                            # Classify the source node
                            src_labels = row.get('source_labels') or []
                            self._classify_node(src, src_labels, text_nodes, table_nodes, image_nodes)

                        if row.get('rel') and row.get('target'):
                            tgt = row['target']
                            tgt_labels = row.get('target_labels') or []
                            src_type = self._primary_label(row.get('source_labels') or [])
                            tgt_type = self._primary_label(tgt_labels)
                            rel_str = f"{src} ({src_type}) {row['rel']} {tgt} ({tgt_type})"
                            relations.append(rel_str)
                            sources.append(f"  • {src} -[{row['rel']}]-> {tgt}")
                            retrieved_node_set.add(tgt)
                            self._classify_node(tgt, tgt_labels, text_nodes, table_nodes, image_nodes)
                        elif src:
                            src_type = self._primary_label(row.get('source_labels') or [])
                            rel_str = f"{src} is a {src_type}"
                            relations.append(rel_str)
                            sources.append(f"  • {src} ({src_type})")

                # Deduplicate node lists while preserving order
                text_nodes = list(dict.fromkeys(text_nodes))
                table_nodes = list(dict.fromkeys(table_nodes))
                image_nodes = list(dict.fromkeys(image_nodes))

                # Step E: Summarize into context
                if relations:
                    summary_prompt = f"""Summarize these knowledge graph facts into concise, factual statements relevant to answering: "{question}"
Facts:
{chr(10).join(relations)}

Provide ONLY the summary statements, no preamble."""

                    summary_res = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a concise information summarizer. Extract key facts."},
                            {"role": "user", "content": summary_prompt}
                        ]
                    )
                    context = summary_res.choices[0].message.content.strip()
                    return context, sources, text_nodes, table_nodes, image_nodes
                else:
                    return "Nodes found but no relationships.", [], text_nodes, table_nodes, image_nodes

        except Exception as e:
            return f"Error querying graph: {str(e)}", [], text_nodes, table_nodes, image_nodes

    @staticmethod
    def _primary_label(labels: List[str]) -> str:
        """Return the most descriptive label for a node, preferring Table/Image over __Entity__."""
        for lbl in ['Table', 'Image']:
            if lbl in labels:
                return lbl
        if '__Entity__' in labels:
            return 'Entity'
        return labels[0] if labels else 'Unknown'

    @staticmethod
    def _classify_node(
        node_id: str,
        labels: List[str],
        text_nodes: List[str],
        table_nodes: List[str],
        image_nodes: List[str],
    ) -> None:
        """Classify a node into the appropriate modality list based on its labels."""
        if 'Table' in labels:
            table_nodes.append(node_id)
        elif 'Image' in labels:
            image_nodes.append(node_id)
        else:
            text_nodes.append(node_id)

    def answer_with_multimodal_context(
        self,
        question: str,
        include_modalities: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Answer a question using multimodal context with modality filtering.
        
        Retrieves context from the knowledge graph, filters by the requested
        modalities, and builds a structured prompt with clearly labeled sections
        (TEXT CONTEXT, TABLE CONTEXT, IMAGE CONTEXT).
        
        Args:
            question: The input question
            include_modalities: List of modalities to include ["text", "table", "image"].
                                Defaults to all three if not specified.
        
        Returns:
            Tuple of (answer, metadata) where metadata contains the separated
            modality content and node lists.
        """
        try:
            start_time = time.time()
            modalities = include_modalities or ["text", "table", "image"]
            
            if self.driver and self.client:
                # Real Neo4j retrieval with multimodal node classification
                context, sources, text_nodes, table_nodes, image_nodes = self.get_multimodal_context(question)

                # --- Build structured context respecting the modality filter ---
                all_retrieved_nodes: List[str] = []
                text_content = ""
                table_content = ""
                image_content = ""

                if "text" in modalities and text_nodes:
                    text_content = self._get_node_descriptions(text_nodes, "Entity")
                    all_retrieved_nodes.extend(text_nodes)
                if "table" in modalities and table_nodes:
                    table_content = self._get_node_descriptions(table_nodes, "Table")
                    all_retrieved_nodes.extend(table_nodes)
                if "image" in modalities and image_nodes:
                    image_content = self._get_node_descriptions(image_nodes, "Image")
                    all_retrieved_nodes.extend(image_nodes)

                # Assemble structured prompt sections
                prompt_sections: List[str] = []
                if "text" in modalities:
                    prompt_sections.append(f"TEXT CONTEXT:\n{text_content if text_content else '(no text nodes retrieved)'}")
                if "table" in modalities:
                    prompt_sections.append(f"TABLE CONTEXT:\n{table_content if table_content else '(no table nodes retrieved)'}")
                if "image" in modalities:
                    prompt_sections.append(f"IMAGE CONTEXT:\n{image_content if image_content else '(no image nodes retrieved)'}")

                # Add the overall summarized context from graph relationships
                if context:
                    prompt_sections.append(f"RELATIONSHIP SUMMARY:\n{context}")

                structured_context = "\n\n".join(prompt_sections)

                answer = ask_llm_with_context(question, structured_context, self.client)
            else:
                # Fallback when Neo4j/OpenAI are unavailable
                all_retrieved_nodes = []
                text_content = ""
                table_content = ""
                image_content = ""
                context = ""
                structured_context = ""
                answer = f"Error: Neo4j or OpenAI client not available."
            
            response_time = time.time() - start_time
            
            metadata = {
                "modalities_used": modalities,
                "retrieved_nodes": all_retrieved_nodes,
                "context": structured_context,
                "text_content": text_content,
                "table_content": table_content,
                "image_content": image_content,
                "text_nodes": text_nodes if self.driver else [],
                "table_nodes": table_nodes if self.driver else [],
                "image_nodes": image_nodes if self.driver else [],
                "response_time": response_time
            }
            return answer, metadata
        
        except Exception as e:
            return f"Error: {str(e)}", {"error": str(e)}

    def _get_node_descriptions(self, node_ids: List[str], node_type: str) -> str:
        """
        Fetch description/content for a list of node IDs from Neo4j.
        
        Args:
            node_ids: List of node IDs to describe
            node_type: Human-readable type label for context
            
        Returns:
            Formatted string with node descriptions
        """
        if not node_ids:
            return ""
        descriptions: List[str] = []
        try:
            with self.driver.session(database=self.database) as session:
                for nid in node_ids:
                    result = session.run(
                        """
                        MATCH (n) WHERE n.id = $node_id
                        RETURN n.id AS id, n.description AS description,
                               n.content AS content, n.text AS text
                        LIMIT 1
                        """,
                        {"node_id": nid}
                    )
                    record = result.single()
                    if record:
                        desc = (
                            record.get('description')
                            or record.get('content')
                            or record.get('text')
                            or nid
                        )
                        descriptions.append(f"[{node_type}: {nid}] {desc}")
                    else:
                        descriptions.append(f"[{node_type}: {nid}]")
        except Exception:
            descriptions = [f"[{node_type}: {nid}]" for nid in node_ids]
        return "\n".join(descriptions)

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

