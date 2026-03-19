#!/usr/bin/env python3
"""
Evaluation Runner for GraphRAG

Usage:
    python scripts/evaluate.py --experiment comprehensive
    python scripts/evaluate.py --experiment naiverag
    python scripts/evaluate.py --experiment significance
    python scripts/evaluate.py --all
"""
import argparse
import sys
from pathlib import Path

# Allow imports from project root (for experiments/) and src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv()


EXPERIMENTS = {
    "comprehensive": "experiments.comprehensive_evaluation",
    "naiverag": "experiments.naiverag_evaluation",
    "significance": "experiments.significance_analysis",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG evaluations")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all evaluations")

    args = parser.parse_args()

    if args.all:
        to_run = list(EXPERIMENTS.keys())
    elif args.experiment:
        to_run = [args.experiment]
    else:
        parser.print_help()
        return

    for exp_name in to_run:
        print(f"\n{'='*60}")
        print(f"  Running: {exp_name}")
        print(f"{'='*60}\n")

        import importlib

        mod = importlib.import_module(EXPERIMENTS[exp_name])
        mod.main()

        print(f"\n✓ {exp_name} complete")


if __name__ == "__main__":
    main()
