"""
Comprehensive evaluation experiment for NaiveRAG (vector-based) system.
Mirrors experiments/comprehensive_evaluation.py for head-to-head comparison
with GraphRAG.  Results are saved to a SEPARATE JSON file:
    results/naiverag_evaluation.json

Retrieval metric note
---------------------
GraphRAG retrieval returns *named graph-node IDs* (e.g. "Transformer",
"Multi-Head Attention") which are directly compared against the corpus
`relevant_items` lists.

NaiveRAG retrieval returns *text chunks* (PDF page segments).  To make
the retrieval F1 comparable, we use **content-based entity matching**:
for each question we scan the retrieved chunk texts and check which
`relevant_items` entity names appear in the text.  The set of matched
entity names is used as `retrieved_items` for precision/recall/F1.

This means the metric answers: "Of the entities that *should* be covered,
how many are *mentioned* in the retrieved chunks?"  It is a fair
content-level comparison, though not an exact apples-to-apples match
with graph-node retrieval.
"""

import os
import re
import sys
import time
import json
from typing import Dict, List, Tuple, Any

# Add project root and src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag.evaluation import EvaluationPipeline
from graphrag.utils import ExperimentLogger
from naiverag import NaiveRAGRetriever, config as naiverag_config
from experiments.comprehensive_evaluation import (
    get_all_corpora,
    MIN_QA_PER_CORPUS,
    HELDOUT_FRACTION,
    HELDOUT_SEED,
)

logger = ExperimentLogger("naiverag_eval")


# ------------------------------------------------------------------ #
# Content-based entity matching for retrieval metrics
# ------------------------------------------------------------------ #
def _match_entities_in_chunks(
    retrieved_chunk_texts: List[str],
    relevant_entity_names: List[str],
) -> List[str]:
    """Identify which relevant entity names appear in the retrieved chunks.

    This bridges the gap between NaiveRAG (chunk-based retrieval) and the
    graph-node-based ``relevant_items`` used in corpus definitions.

    Matching is case-insensitive and uses word-boundary-aware substring
    search so that e.g. "Transformer" matches "the Transformer model"
    but not "TransformerBlock" (unless that exact entity is in the list).

    Args:
        retrieved_chunk_texts: The page_content strings of retrieved docs.
        relevant_entity_names: The ground-truth relevant entity names
            from the corpus definition.

    Returns:
        De-duplicated list of entity names that were found in **any**
        retrieved chunk.
    """
    if not retrieved_chunk_texts or not relevant_entity_names:
        return []

    # Build a single merged text for efficient scanning
    merged = "\n".join(retrieved_chunk_texts).lower()

    matched: List[str] = []
    for entity in relevant_entity_names:
        # Normalise: lowercase, collapse whitespace
        pattern = re.escape(entity.strip().lower())
        # Allow flexible whitespace between words
        pattern = r"\s+".join(pattern.split(r"\ "))
        if re.search(pattern, merged):
            matched.append(entity)

    return matched


# ------------------------------------------------------------------ #
# Experiment runner (per corpus)
# ------------------------------------------------------------------ #
def run_naiverag_corpus(
    corpus,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run NaiveRAG evaluation on a single corpus.

    Args:
        corpus: A Corpus instance with .questions, .references, .corpus_id.

    Returns:
        Tuple of (metrics_summary dict, per_question_results list).
    """
    logger.get_logger().info("=" * 60)
    logger.get_logger().info(
        f"Running NAIVERAG EXPERIMENT on corpus '{corpus.corpus_id}': "
        "Vector-Based Retrieval"
    )
    logger.get_logger().info("=" * 60)

    retriever = NaiveRAGRetriever(naiverag_config)
    evaluator = EvaluationPipeline(f"naiverag_{corpus.corpus_id}")

    per_question_results: List[Dict[str, Any]] = []
    all_metrics = []
    questions, references, relevant_items = corpus.get_expanded_data(
        min_questions=MIN_QA_PER_CORPUS
    )
    split_info = corpus.get_heldout_split(
        total_questions=len(questions),
        heldout_fraction=HELDOUT_FRACTION,
        seed=HELDOUT_SEED,
    )
    heldout_set = set(split_info.get("heldout_indices", []))

    for i, (question, reference) in enumerate(
        zip(questions, references)
    ):
        logger.get_logger().info(
            f"\nQuestion {i + 1}/{len(questions)}: {question[:50]}..."
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

            # ---- Content-based entity matching for retrieval F1 ----
            # NaiveRAG retrieves text chunks, not named graph nodes.
            # We check which relevant entity names appear in the
            # retrieved chunk texts and use the matches as
            # `retrieved_items`.  See module docstring for rationale.
            relevant = relevant_items[i] if i < len(relevant_items) else []

            # Re-fetch docs to get their page_content for matching
            docs = retriever.vectorstore.similarity_search(
                question, k=retriever.cfg.top_k
            )
            chunk_texts = [d.page_content for d in docs]
            matched_entities = _match_entities_in_chunks(chunk_texts, relevant)

            # --- Evaluate ---
            metrics = evaluator.evaluate(
                question=question,
                generated_answer=answer,
                reference_answer=reference,
                retrieved_context=retrieved_context,
                # Use matched entity names (not chunk IDs) for retrieval F1
                retrieved_items=matched_entities,
                relevant_items=relevant,
            )

            metrics.avg_response_time = response_time
            metrics.corpus_id = corpus.corpus_id
            all_metrics.append(metrics)

            per_question_results.append({
                "question": question,
                "corpus_id": corpus.corpus_id,
                "answer": answer,
                "retrieved_context": retrieved_context,
                "sources": sources,
                "split": "heldout" if i in heldout_set else "train",
                "matched_entities": matched_entities,
                "relevant_entities": relevant,
                "metrics": metrics.to_dict(),
                "response_time": response_time,
            })

            hall_str = (
                f"{metrics.hallucination_rate:.3f}"
                if metrics.hallucination_rate is not None
                else "None"
            )
            logger.get_logger().info(
                f"[+] F1: {metrics.retrieval_f1:.3f}, "
                f"Matched: {len(matched_entities)}/{len(relevant)}, "
                f"Hallucination: {hall_str}, "
                f"Semantic Sim: {metrics.semantic_similarity:.3f}, "
                f"ROUGE-1: {metrics.rouge_score:.3f}, "
                f"Response Time: {response_time:.4f}s"
            )

        except Exception as e:
            logger.get_logger().error(
                f"[!] Error processing question {i + 1}: {str(e)}"
            )

    # Aggregate per-corpus
    n = len(all_metrics)

    # Handle None hallucination rates from previous fix
    valid_hall = [m for m in all_metrics if m.hallucination_rate is not None]
    valid_grounded = [m for m in all_metrics if m.grounded_ratio is not None]

    metrics_summary: Dict[str, Any] = {
        "experiment": "naiverag",
        "corpus_id": corpus.corpus_id,
        "num_questions": len(questions),
        "train_questions": len(split_info.get("train_indices", [])),
        "heldout_questions": len(split_info.get("heldout_indices", [])),
        "heldout_fraction": HELDOUT_FRACTION,
        "heldout_seed": HELDOUT_SEED,
        "questions_evaluated": n,
        "avg_retrieval_f1": (
            sum(m.retrieval_f1 for m in all_metrics) / n if n else 0.0
        ),
        "avg_retrieval_precision": (
            sum(m.retrieval_precision for m in all_metrics) / n if n else 0.0
        ),
        "avg_retrieval_recall": (
            sum(m.retrieval_recall for m in all_metrics) / n if n else 0.0
        ),
        "retrieval_metric_note": (
            "Content-based entity matching: relevant_items entity names "
            "found in retrieved chunk text (case-insensitive substring)"
        ),
        "avg_hallucination_rate": (
            sum(m.hallucination_rate for m in valid_hall) / len(valid_hall)
            if valid_hall else 0.0
        ),
        "hallucination_none_count": n - len(valid_hall),
        "avg_grounded_ratio": (
            sum(m.grounded_ratio for m in valid_grounded) / len(valid_grounded)
            if valid_grounded else 0.0
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
# Save results
# ------------------------------------------------------------------ #
def save_results(
    per_corpus: Dict[str, Dict[str, Any]],
    aggregate: Dict[str, Any],
    all_details: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Save multi-corpus NaiveRAG evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_corpus": per_corpus,
        "aggregate": aggregate,
        "details": all_details,
    }

    filepath = os.path.join(output_dir, "naiverag_evaluation.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.get_logger().info(f"\n[+] Results saved to {filepath}")
    return filepath


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    """Run NaiveRAG evaluation end-to-end across all corpora."""
    logger.get_logger().info("NaiveRAG Comprehensive Evaluation")
    logger.get_logger().info(f"Experiment ID: {logger.experiment_name}")

    corpora = get_all_corpora()
    logger.get_logger().info(
        f"Loaded {len(corpora)} corpora: {[c.corpus_id for c in corpora]}"
    )

    per_corpus: Dict[str, Dict[str, Any]] = {}
    all_details: List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []

    for corpus in corpora:
        expanded_questions, _expanded_references, _expanded_relevant = corpus.get_expanded_data(
            min_questions=MIN_QA_PER_CORPUS
        )
        logger.get_logger().info(
            f"\n{'#' * 60}\n"
            f"# CORPUS: {corpus.corpus_id}  "
            f"({len(expanded_questions)} questions, expanded from {len(corpus.questions)})\n"
            f"{'#' * 60}"
        )

        summary, details = run_naiverag_corpus(corpus)
        per_corpus[corpus.corpus_id] = summary
        all_details.extend(details)
        all_summaries.append(summary)

    # Aggregate across all corpora
    n_total = len(all_summaries)
    total_questions = sum(s["questions_evaluated"] for s in all_summaries)
    aggregate: Dict[str, Any] = {
        "num_corpora": n_total,
        "num_questions": total_questions,
        "avg_retrieval_f1": (
            sum(s.get("avg_retrieval_f1", 0) for s in all_summaries) / n_total
            if n_total else 0.0
        ),
        "avg_hallucination_rate": (
            sum(s["avg_hallucination_rate"] for s in all_summaries) / n_total
            if n_total else 0.0
        ),
        "avg_semantic_similarity": (
            sum(s["avg_semantic_similarity"] for s in all_summaries) / n_total
            if n_total else 0.0
        ),
        "avg_rouge_score": (
            sum(s["avg_rouge_score"] for s in all_summaries) / n_total
            if n_total else 0.0
        ),
        "avg_response_time": (
            sum(s["avg_response_time"] for s in all_summaries) / n_total
            if n_total else 0.0
        ),
    }

    results_file = save_results(per_corpus, aggregate, all_details)

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("NAIVERAG EVALUATION SUMMARY")
    print("=" * 60)
    for cid, summary in per_corpus.items():
        print(f"\n--- Corpus: {cid} ---")
        print(f"  Questions Evaluated : {summary['questions_evaluated']}/{summary['num_questions']}")
        print(f"  Retrieval F1        : {summary.get('avg_retrieval_f1', 0):.3f}")
        print(f"  Retrieval Precision : {summary.get('avg_retrieval_precision', 0):.3f}")
        print(f"  Retrieval Recall    : {summary.get('avg_retrieval_recall', 0):.3f}")
        print(f"  Hallucination Rate  : {summary['avg_hallucination_rate']:.3f}")
        print(f"  Semantic Similarity : {summary['avg_semantic_similarity']:.3f}")
        print(f"  ROUGE-1             : {summary['avg_rouge_score']:.3f}")

    print(f"\n--- Aggregate (all corpora) ---")
    print(f"  Total Questions     : {aggregate['num_questions']}")
    print(f"  Retrieval F1        : {aggregate.get('avg_retrieval_f1', 0):.3f}")
    print(f"  Hallucination Rate  : {aggregate['avg_hallucination_rate']:.3f}")
    print(f"  Semantic Similarity : {aggregate['avg_semantic_similarity']:.3f}")
    print(f"  ROUGE-1             : {aggregate['avg_rouge_score']:.3f}")
    print(f"  Avg Response Time   : {aggregate['avg_response_time']:.3f}s")
    print(f"\n  Results saved to: {results_file}")

    logger.get_logger().info("\n[+] Evaluation complete!")


if __name__ == "__main__":
    main()
