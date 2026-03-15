"""
Comprehensive evaluation experiment for NaiveRAG (vector-based) system.
Mirrors experiments/comprehensive_evaluation.py for head-to-head comparison
with GraphRAG.  Results are saved to a SEPARATE JSON file:
    results/naiverag_evaluation.json
"""

import os
import re
import sys
import time
import json
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag.evaluation import EvaluationPipeline
from graphrag.utils import ExperimentLogger
from naiverag import NaiveRAGRetriever, config as naiverag_config

logger = ExperimentLogger("naiverag_eval")

_CITE_RE = re.compile(r'\s*\[cite:\s*[\d,\s]+\]', re.IGNORECASE)


def _strip_cite_markers(text: str) -> str:
    """Remove all ``[cite: ...]`` markers from *text*."""
    return _CITE_RE.sub('', text).strip()


# ------------------------------------------------------------------ #
# Benchmark dataset (identical questions & references as GraphRAG)
# ------------------------------------------------------------------ #
class NaiveRAGBenchmarkDataset:
    """Benchmark questions and ground-truth references for the Attention paper."""

    def __init__(self):
        self.questions = [
            "What are the main characteristics of the Transformer architecture?",
            "How does Multi-Head Attention relate to Scaled Dot-Product Attention?",
            "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?",
            "Compare the computational complexity per layer of self-attention layers and recurrent layers.",
            "What is the impact of masking in the decoder's self-attention sub-layer?"
        ]

        _raw_refs = [
            "The Transformer is a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely, and using stacked self-attention and point-wise, fully connected layers. [cite: 17, 78]",
            "Multi-Head Attention connects to Scaled Dot-Product Attention by linearly projecting queries, keys, and values h times, and performing the scaled dot-product attention function in parallel on each projected version. [cite: 126, 127]",
            "The Transformer model achieved a new state-of-the-art BLEU score of 28.4 on the WMT 2014 English-to-German translation task, improving over existing best results by over 2 BLEU. [cite: 19]",
            "Self-attention layers have a complexity of O(n^2 * d) per layer, while recurrent layers have a complexity of O(n * d^2), making self-attention faster when sequence length n is smaller than representation dimensionality d. [cite: 163, 187, 188, 189]",
            "Masking impacts the decoder by preventing positions from attending to subsequent positions, ensuring that predictions for position i can depend only on the known outputs at positions less than i, preserving the auto-regressive property. [cite: 88, 89]"
        ]
        self.references = [_strip_cite_markers(r) for r in _raw_refs]

    def __len__(self):
        return len(self.questions)


# ------------------------------------------------------------------ #
# Experiment runner
# ------------------------------------------------------------------ #
def run_naiverag_experiment(
    dataset: NaiveRAGBenchmarkDataset,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run evaluation against the NaiveRAG vector retriever.

    Args:
        dataset: Benchmark dataset with questions and references.

    Returns:
        Tuple of (metrics_summary dict, per_question_results list).
    """
    logger.get_logger().info("=" * 60)
    logger.get_logger().info("Running NAIVERAG EXPERIMENT: Vector-Based Retrieval")
    logger.get_logger().info("=" * 60)

    retriever = NaiveRAGRetriever(naiverag_config)
    evaluator = EvaluationPipeline("naiverag_exp")

    per_question_results: List[Dict[str, Any]] = []
    all_metrics = []

    for i, (question, reference) in enumerate(
        zip(dataset.questions, dataset.references)
    ):
        logger.get_logger().info(
            f"\nQuestion {i + 1}/{len(dataset)}: {question[:50]}..."
        )

        try:
            # --- Retrieve context + generate answer (timed) ---
            start_time = time.time()
            context, sources, retrieved_ids = retriever.get_context(question)
            answer, meta = retriever.answer_question(question)
            response_time = time.time() - start_time

            # Use the context from get_context (raw chunks), NOT the answer
            retrieved_context = context
            assert retrieved_context != answer, (
                "ERROR: retrieved_context must differ from answer "
                "(circular evaluation detected)"
            )

            # --- Evaluate ---
            metrics = evaluator.evaluate(
                question=question,
                generated_answer=answer,
                reference_answer=reference,
                retrieved_context=retrieved_context,
            )

            metrics.avg_response_time = response_time
            all_metrics.append(metrics)

            per_question_results.append({
                "question": question,
                "answer": answer,
                "retrieved_context": retrieved_context,
                "sources": sources,
                "metrics": metrics.to_dict(),
                "response_time": response_time,
            })

            logger.get_logger().info(
                f"[+] Hallucination: {metrics.hallucination_rate:.3f}, "
                f"Semantic Sim: {metrics.semantic_similarity:.3f}, "
                f"ROUGE-1: {metrics.rouge_score:.3f}, "
                f"Response Time: {response_time:.4f}s"
            )

        except Exception as e:
            logger.get_logger().error(
                f"[!] Error processing question {i + 1}: {str(e)}"
            )

    # Aggregate
    n = len(all_metrics)
    metrics_summary: Dict[str, Any] = {
        "experiment": "naiverag",
        "num_questions": len(dataset),
        "questions_evaluated": n,
        "avg_hallucination_rate": (
            sum(m.hallucination_rate for m in all_metrics) / n if n else 0.0
        ),
        "avg_grounded_ratio": (
            sum(m.grounded_ratio for m in all_metrics) / n if n else 0.0
        ),
        "avg_semantic_similarity": (
            sum(m.semantic_similarity for m in all_metrics) / n if n else 0.0
        ),
        "avg_rouge_score": (
            sum(m.rouge_score for m in all_metrics) / n if n else 0.0
        ),
        "avg_response_time": (
            sum(m.avg_response_time for m in all_metrics) / n if n else 0.0
        ),
    }

    return metrics_summary, per_question_results


# ------------------------------------------------------------------ #
# Corpus Adapter
# ------------------------------------------------------------------ #
from experiments.comprehensive_evaluation import get_all_corpora

def save_results(
    per_corpus_naiverag: Dict[str, Tuple],
    output_dir: str = "results",
) -> str:
    """Save multi-corpus NaiveRAG evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_corpus_naiverag": {
            cid: {"summary": data[0], "details": data[1]}
            for cid, data in per_corpus_naiverag.items()
        },
    }

    filepath = os.path.join(output_dir, "naiverag_evaluation.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

    logger.get_logger().info(f"\n[+] Results saved to {filepath}")
    return filepath


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    """Run NaiveRAG evaluation end-to-end."""
    logger.get_logger().info("NaiveRAG Comprehensive Evaluation")
    logger.get_logger().info(f"Experiment ID: {logger.experiment_name}")

    corpora = get_all_corpora()
    logger.get_logger().info(
        f"Loaded {len(corpora)} corpora: {[c.corpus_id for c in corpora]}"
    )

    per_corpus_naiverag: Dict[str, Tuple] = {}

    for corpus in corpora:
        logger.get_logger().info(
            f"\n{'#' * 60}\n"
            f"# CORPUS: {corpus.corpus_id}  ({len(corpus)} questions)\n"
            f"{'#' * 60}"
        )

        summary, details = run_naiverag_experiment(corpus)
        per_corpus_naiverag[corpus.corpus_id] = (summary, details)

    results_file = save_results(per_corpus_naiverag)

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("NAIVERAG EVALUATION SUMMARY")
    print("=" * 60)
    for cid, (summary, _) in per_corpus_naiverag.items():
        print(f"\n--- Corpus: {cid} ---")
        print(f"  Questions Evaluated : {summary['questions_evaluated']}/{summary['num_questions']}")
        print(f"  Hallucination Rate  : {summary['avg_hallucination_rate']:.3f}")
        print(f"  Grounded Ratio      : {summary['avg_grounded_ratio']:.3f}")
        print(f"  Semantic Similarity : {summary['avg_semantic_similarity']:.3f}")
        print(f"  ROUGE-1             : {summary['avg_rouge_score']:.3f}")
    print(f"\n  Results saved to: {results_file}")

    logger.get_logger().info("\n[+] Evaluation complete!")


if __name__ == "__main__":
    main()
