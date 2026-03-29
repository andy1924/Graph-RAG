"""
Configuration management for GraphRAG system.
Handles environment variables, model selection, and hyperparameters.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_config_logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for LLM models."""
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o")


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j database."""
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""
    # Graph generation
    graph_model: str = os.getenv("GRAPH_MODEL", "gpt-4o-mini")
    max_nodes_extraction: int = 100
    max_relationships_extraction: int = 150
    
    # Multimodal processing
    extract_images: bool = True
    infer_table_structure: bool = True
    chunking_strategy: str = "by_title"
    
    # Storage
    preprocessed_dir: str = "data/preprocessed"
    extracted_images_dir: str = "data/extracted_images"
    raw_data_dir: str = "data/raw"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    # Graph traversal
    max_hop_distance: int = 3
    relationships_limit: int = 15
    entity_prefilter_top_k: int = 100
    
    # Ranking
    use_semantic_ranking: bool = True
    rank_by_relevance: bool = True
    
    # LLM generation
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Benchmark datasets
    benchmark_dir: str = "benchmarks"
    results_dir: str = "results"
    
    # Evaluation metrics
    use_rouge: bool = True
    use_bert_score: bool = True
    use_semantic_similarity: bool = True
    
    # Hallucination detection
    detect_hallucinations: bool = True
    hallucination_threshold: float = 0.35
    graph_hallucination_threshold: float = 0.22


class Config:
    """Main configuration class combining all settings."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.neo4j = Neo4jConfig()
        self.ingestion = IngestionConfig()
        self.retrieval = RetrievalConfig()
        self.evaluation = EvaluationConfig()

        # --- Log hallucination thresholds at startup ---
        _config_logger.info(
            f"[Config] Evaluation hallucination thresholds: "
            f"hallucination_threshold={self.evaluation.hallucination_threshold}, "
            f"graph_hallucination_threshold={self.evaluation.graph_hallucination_threshold}"
        )
        # Validate thresholds are in a reasonable range
        for name, val in [
            ("hallucination_threshold", self.evaluation.hallucination_threshold),
            ("graph_hallucination_threshold", self.evaluation.graph_hallucination_threshold),
        ]:
            if val < 0.1 or val > 0.8:
                _config_logger.warning(
                    f"[Config] {name}={val} is outside the recommended range "
                    f"[0.1, 0.8]. This may cause all sentences to trivially "
                    f"pass or fail grounding checks."
                )
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "neo4j": self.neo4j.__dict__,
            "ingestion": self.ingestion.__dict__,
            "retrieval": self.retrieval.__dict__,
            "evaluation": self.evaluation.__dict__,
        }


# Global configuration instance
config = Config()
