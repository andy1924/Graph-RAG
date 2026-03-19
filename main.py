#!/usr/bin/env python3
"""
GraphRAG — Graph-Based Retrieval-Augmented Generation

Main launcher providing unified access to all subsystems.

Usage:
    python main.py ingest   --corpus tesla        # Ingest data
    python main.py query    --mode graphrag        # Interactive queries
    python main.py evaluate --experiment comprehensive  # Run evaluations
    python main.py evaluate --all                  # Run all evaluations

For detailed help on each command:
    python main.py ingest   --help
    python main.py query    --help
    python main.py evaluate --help
"""
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    # Shift argv so argparse in sub-scripts sees the right args
    sys.argv = sys.argv[1:]

    if command == "ingest":
        from scripts.ingest import main as run
        run()
    elif command == "query":
        from scripts.query import main as run
        run()
    elif command == "evaluate":
        from scripts.evaluate import main as run
        run()
    else:
        print(f"Unknown command: '{command}'")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
