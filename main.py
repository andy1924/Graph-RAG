#!/usr/bin/env python3
"""
GraphRAG - Graph-based Retrieval-Augmented Generation System

Main entry point for the research-grade GraphRAG system.

This script provides access to both simple and multimodal graph-based
question answering systems with full source attribution.

Usage:
    python main.py --mode simple      # Simple entity-based retrieval
    python main.py --mode multimodal  # Multimodal context retrieval
    python main.py --ingest           # Ingest data from PDF into graph

Author: GraphRAG Research Team
Date: 2026
"""

import argparse
import sys
from typing import Literal


def run_simple_retrieval() -> None:
    """Run the simple retrieval system."""
    from retrieval import main
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nQuery interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error in retrieval system: {e}")
        sys.exit(1)


def run_multimodal_retrieval() -> None:
    """Run the multimodal retrieval system."""
    from multiModalGraphRetrieval import main
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nQuery interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error in multimodal retrieval system: {e}")
        sys.exit(1)


def run_data_ingestion() -> None:
    """Run the data ingestion pipeline."""
    try:
        print("Starting data ingestion pipeline...")
        print("This will load the Attention PDF and ingest it into Neo4j.")
        print("See ingest_attention.py for details.")
        import ingest_attention
        # The script runs on import
    except KeyboardInterrupt:
        print("\n\nIngestion interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error in data ingestion: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point with command-line argument handling."""
    parser = argparse.ArgumentParser(
        description="GraphRAG - Graph-based Retrieval-Augmented Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode simple      # Interactive simple retrieval
  python main.py --mode multimodal  # Interactive multimodal retrieval
  python main.py --ingest           # Ingest PDF into knowledge graph
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "multimodal"],
        default="simple",
        help="Retrieval mode: simple (entity-based) or multimodal (context-based)"
    )
    
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run data ingestion pipeline instead of retrieval"
    )
    
    args = parser.parse_args()
    
    if args.ingest:
        run_data_ingestion()
    elif args.mode == "simple":
        print("=" * 80)
        print("GraphRAG - Simple Entity-Based Retrieval System")
        print("=" * 80)
        run_simple_retrieval()
    elif args.mode == "multimodal":
        print("=" * 80)
        print("GraphRAG - Multimodal Context-Based Retrieval System")
        print("=" * 80)
        run_multimodal_retrieval()


if __name__ == "__main__":
    main()
