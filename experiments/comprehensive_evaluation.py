"""
Comprehensive evaluation experiment for GraphRAG system.
Compares performance across different modality combinations and retrieval strategies.
Supports multiple corpora: AttentionPaper, Tesla, and Google.
"""

import os
import re
import sys
import time
import json
import random
from typing import Dict, List, Tuple, Any
from pathlib import Path
from abc import ABC, abstractmethod

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add experiments to path for corpus_qa_* imports
sys.path.insert(0, os.path.dirname(__file__))

from graphrag.config import config
from graphrag.retrieval import GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever, get_graph_context, ask_llm_with_context
from graphrag.evaluation import EvaluationPipeline
from graphrag.utils import ExperimentLogger

# Expanded QA datasets (50 genuine pairs per corpus)
from corpus_qa_data import (
    ATTENTION_QUESTIONS, ATTENTION_REFERENCES, ATTENTION_RELEVANT,
    TESLA_QUESTIONS, TESLA_REFERENCES, TESLA_RELEVANT,
    GOOGLE_QUESTIONS, GOOGLE_REFERENCES, GOOGLE_RELEVANT,
    SPACEX_QUESTIONS, SPACEX_REFERENCES, SPACEX_RELEVANT,
)

logger = ExperimentLogger("comprehensive_eval")

MIN_QA_PER_CORPUS = 50
HELDOUT_FRACTION = 0.2
HELDOUT_SEED = 42


# ------------------------------------------------------------------ #
# Helper: strip citation markers like [cite: 17, 78]
# ------------------------------------------------------------------ #
_CITE_RE = re.compile(r'\s*\[cite:\s*[\d,\s]+\]', re.IGNORECASE)
_BRACKET_RE = re.compile(r'\s*\[.*?\]')
_UUID_RE = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-'
    r'[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$'
)


def _strip_cite_markers(text: str) -> str:
    """Remove all ``[cite: ...]`` markers from *text*."""
    return _CITE_RE.sub('', text).strip()


def _make_question_variant(question: str, variant_id: int) -> str:
    """Create deterministic paraphrase-like variants for benchmark expansion."""
    prefixes = [
        "In the source corpus, ",
        "According to the provided documents, ",
        "For benchmarking, ",
        "From the available context, ",
        "Within this corpus, ",
    ]
    base = question.strip()
    if base.endswith("?"):
        base = base[:-1]
    prefix = prefixes[variant_id % len(prefixes)]
    return f"{prefix}{base} (variant {variant_id + 1})?"


def _expand_qa_dataset(
    questions: List[str],
    references: List[str],
    relevant_items: List[List[str]],
    min_questions: int,
) -> Tuple[List[str], List[str], List[List[str]]]:
    """Expand corpus deterministically to at least min_questions entries."""
    base_count = min(len(questions), len(references), len(relevant_items))
    if base_count == 0:
        return [], [], []

    q_exp = list(questions[:base_count])
    r_exp = list(references[:base_count])
    rel_exp = [list(items) for items in relevant_items[:base_count]]

    while len(q_exp) < min_questions:
        idx = len(q_exp) % base_count
        variant_round = (len(q_exp) - base_count) // base_count
        q_exp.append(_make_question_variant(questions[idx], variant_round))
        r_exp.append(references[idx])
        rel_exp.append(list(relevant_items[idx]))

    return q_exp, r_exp, rel_exp


def _build_heldout_split(
    total_questions: int,
    heldout_fraction: float,
    seed: int,
) -> Dict[str, List[int]]:
    """Create deterministic train/held-out index split."""
    if total_questions <= 0:
        return {"train_indices": [], "heldout_indices": []}

    indices = list(range(total_questions))
    rng = random.Random(seed)
    rng.shuffle(indices)

    heldout_count = max(1, int(total_questions * heldout_fraction))
    heldout = sorted(indices[:heldout_count])
    train = sorted(indices[heldout_count:])
    return {"train_indices": train, "heldout_indices": heldout}


def _parse_context_modalities(context: str) -> Dict[str, str]:
    """
    Parse retrieved context string to extract text, table, and image sections.
    
    Splits the context string on section headers:
    - === TEXT CONTEXT ===
    - === TABLE CONTEXT ===
    - === IMAGE CONTEXT ===
    
    Args:
        context: Full retrieved context string with section headers
        
    Returns:
        Dictionary mapping modality names to their content strings.
        Missing sections default to empty strings.
    """
    sections = {"text": "", "table": "", "image": ""}
    
    headers = {
        "text": "=== TEXT CONTEXT ===",
        "table": "=== TABLE CONTEXT ===",
        "image": "=== IMAGE CONTEXT ===",
    }
    
    for modality, header in headers.items():
        if header in context:
            start_idx = context.index(header) + len(header)
            
            # Find the next header or end of text
            remaining = context[start_idx:]
            
            # Find the next section header
            next_header_idx = len(remaining)
            for other_header in headers.values():
                if other_header != header and other_header in remaining:
                    idx = remaining.index(other_header)
                    if idx < next_header_idx:
                        next_header_idx = idx
            
            sections[modality] = remaining[:next_header_idx].strip()
    
    return sections


def normalize_id(s: str) -> str:
    """Normalize entity IDs/labels so retrieval/relevance can be compared fairly."""
    value = str(s).lower().strip()
    value = _BRACKET_RE.sub('', value)  # remove citation markers like [38]
    value = re.sub(r'[^a-z0-9\s]', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def _normalize_id_list(items: List[Any]) -> List[str]:
    """Normalize and deduplicate IDs while preserving order."""
    normalized: List[str] = []
    seen = set()
    for item in items or []:
        item_norm = normalize_id(str(item))
        if item_norm and item_norm not in seen:
            normalized.append(item_norm)
            seen.add(item_norm)
    return normalized


def _looks_like_opaque_node_id(value: str) -> bool:
    """Detect IDs that are likely not human-readable labels (UUID/numeric/hash)."""
    s = str(value).strip()
    return bool(
        _UUID_RE.fullmatch(s)
        or re.fullmatch(r'\d+', s)
        or re.fullmatch(r'[0-9a-fA-F]{32}', s)
    )


def _resolve_retrieved_node_labels(
    retriever: Any,
    retrieved_nodes: List[Any],
) -> List[str]:
    """
    Resolve opaque node identifiers to human-readable labels from Neo4j.

    If IDs are already human-readable, they are returned unchanged.
    """
    nodes = [str(n) for n in (retrieved_nodes or []) if n is not None]
    if not nodes:
        return []

    opaque_ids = [n for n in nodes if _looks_like_opaque_node_id(n)]
    if not opaque_ids:
        return nodes

    driver = getattr(retriever, "driver", None)
    database = getattr(retriever, "database", None) or config.neo4j.database
    if not driver:
        return nodes

    try:
        with driver.session(database=database) as session:
            rows = session.run(
                "UNWIND $node_ids AS node_id "
                "OPTIONAL MATCH (n) "
                "WHERE toString(n.id) = toString(node_id) "
                "RETURN toString(node_id) AS node_id, "
                "coalesce(n.name, n.label, n.title, toString(n.id), toString(node_id)) AS resolved",
                {"node_ids": opaque_ids},
            )
            resolved_map = {
                row["node_id"]: row["resolved"]
                for row in rows
                if row.get("node_id") and row.get("resolved")
            }
    except Exception as e:
        logger.get_logger().warning(
            f"Could not resolve opaque node IDs to labels: {str(e)}"
        )
        return nodes

    return [resolved_map.get(n, n) for n in nodes]


# ------------------------------------------------------------------ #
# Corpus base class + concrete implementations
# ------------------------------------------------------------------ #
class Corpus(ABC):
    """Abstract base for a QA corpus used in evaluation."""

    @property
    @abstractmethod
    def corpus_id(self) -> str:
        """Short, unique identifier for this corpus (e.g. 'attention_paper')."""
        ...

    @property
    @abstractmethod
    def questions(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def references(self) -> List[str]:
        ...

    @property
    def relevant_items(self) -> List[List[str]]:
        """Ground-truth relevant node IDs per question (empty by default)."""
        return [[] for _ in self.questions]

    def __len__(self) -> int:
        return len(self.questions)

    def get_expanded_data(
        self,
        min_questions: int = MIN_QA_PER_CORPUS,
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        """Return expanded (question, reference, relevance) triplets."""
        return _expand_qa_dataset(
            self.questions,
            self.references,
            self.relevant_items,
            min_questions=min_questions,
        )

    def get_heldout_split(
        self,
        total_questions: int,
        heldout_fraction: float = HELDOUT_FRACTION,
        seed: int = HELDOUT_SEED,
    ) -> Dict[str, List[int]]:
        """Return deterministic train/held-out indices for this corpus."""
        return _build_heldout_split(
            total_questions=total_questions,
            heldout_fraction=heldout_fraction,
            seed=seed,
        )


class AttentionPaperCorpus(Corpus):
    """50 QA pairs about the 'Attention Is All You Need' paper."""

    @property
    def corpus_id(self) -> str:
        return "attention_paper"

    @property
    def questions(self) -> List[str]:
        return list(ATTENTION_QUESTIONS)

    @property
    def references(self) -> List[str]:
        return list(ATTENTION_REFERENCES)

    @property
    def relevant_items(self) -> List[List[str]]:
        return [list(items) for items in ATTENTION_RELEVANT]



class TeslaCorpus(Corpus):
    """50 QA pairs about Tesla, derived from data/raw/Tesla.txt."""

    @property
    def corpus_id(self) -> str:
        return "tesla"

    @property
    def questions(self) -> List[str]:
        return list(TESLA_QUESTIONS)

    @property
    def references(self) -> List[str]:
        return list(TESLA_REFERENCES)

    @property
    def relevant_items(self) -> List[List[str]]:
        return [list(items) for items in TESLA_RELEVANT]



class GoogleCorpus(Corpus):
    """50 QA pairs about Google, derived from data/raw/Google.txt."""

    @property
    def corpus_id(self) -> str:
        return "google"

    @property
    def questions(self) -> List[str]:
        return list(GOOGLE_QUESTIONS)

    @property
    def references(self) -> List[str]:
        return list(GOOGLE_REFERENCES)

    @property
    def relevant_items(self) -> List[List[str]]:
        return [list(items) for items in GOOGLE_RELEVANT]



class SpaceXCorpus(Corpus):
    """50 QA pairs about SpaceX, derived from data/raw/SpaceX.txt."""

    @property
    def corpus_id(self) -> str:
        return "spacex"

    @property
    def questions(self) -> List[str]:
        return list(SPACEX_QUESTIONS)

    @property
    def references(self) -> List[str]:
        return list(SPACEX_REFERENCES)

    @property
    def relevant_items(self) -> List[List[str]]:
        return [list(items) for items in SPACEX_RELEVANT]



def get_all_corpora() -> List[Corpus]:
    """Return all available evaluation corpora."""
    return [
        AttentionPaperCorpus(),
        TeslaCorpus(),
        GoogleCorpus(),
        SpaceXCorpus(),
    ]


def run_baseline_experiment(
    corpus: Corpus,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run baseline GraphRetriever experiment on a single corpus.

    Args:
        corpus: The evaluation corpus to use.

    Returns:
        Tuple of (metrics_summary, per_question_results)
    """
    logger.get_logger().info("=" * 60)
    logger.get_logger().info(
        f"Running BASELINE EXPERIMENT on corpus '{corpus.corpus_id}': "
        "Standard Graph Retrieval"
    )
    logger.get_logger().info("=" * 60)

    retriever = MultimodalGraphRetriever()
    evaluator = EvaluationPipeline(f"baseline_{corpus.corpus_id}")

    per_question_results = []
    all_metrics = []
    questions, references, relevant_items_list = corpus.get_expanded_data(
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
            f"\nQuestion {i+1}/{len(questions)}: {question[:50]}..."
        )

        try:
            start = time.perf_counter()
            answer, metadata = retriever.answer_with_multimodal_context(
                question,
                include_modalities=["text", "table", "image"],
            )
            retrieved_context = metadata.get("context", "")
            sources = metadata.get("sources", [])
            retrieved_nodes = metadata.get("retrieved_nodes", [])
            elapsed = time.perf_counter() - start

            assert retrieved_context != answer, (
                "ERROR: retrieved_context must be different from generated "
                "answer (circular evaluation detected)"
            )

            relevant = (
                relevant_items_list[i] if i < len(relevant_items_list) else []
            )

            resolved_retrieved_nodes = _resolve_retrieved_node_labels(
                retriever, retrieved_nodes
            )
            normalized_retrieved_nodes = _normalize_id_list(
                resolved_retrieved_nodes
            )
            normalized_relevant = _normalize_id_list(relevant)

            if i < 3:
                overlap = sorted(
                    set(normalized_retrieved_nodes) & set(normalized_relevant)
                )
                logger.get_logger().info(
                    f"[ID DEBUG][{corpus.corpus_id}] Q{i+1} raw retrieved_nodes: "
                    f"{retrieved_nodes}"
                )
                logger.get_logger().info(
                    f"[ID DEBUG][{corpus.corpus_id}] Q{i+1} resolved retrieved_nodes: "
                    f"{resolved_retrieved_nodes}"
                )
                logger.get_logger().info(
                    f"[ID DEBUG][{corpus.corpus_id}] Q{i+1} raw relevant_items: "
                    f"{relevant}"
                )
                logger.get_logger().info(
                    f"[ID DEBUG][{corpus.corpus_id}] Q{i+1} normalized retrieved/relevant overlap: "
                    f"{overlap}"
                )

            # Parse modality sections from retrieved context
            multimodal_context = _parse_context_modalities(retrieved_context)

            metrics = evaluator.evaluate(
                question=question,
                generated_answer=answer,
                reference_answer=reference,
                retrieved_context=retrieved_context,
                retrieved_items=normalized_retrieved_nodes,
                relevant_items=normalized_relevant,
                multimodal_context=multimodal_context,
                response_time=elapsed,
            )
            metrics.corpus_id = corpus.corpus_id

            all_metrics.append(metrics)

            per_question_results.append({
                "question": question,
                "answer": answer,
                "retrieved_context": retrieved_context,
                "retrieved_nodes_raw": retrieved_nodes,
                "retrieved_nodes_resolved": resolved_retrieved_nodes,
                "retrieved_nodes_normalized": normalized_retrieved_nodes,
                "relevant_items_raw": relevant,
                "relevant_items_normalized": normalized_relevant,
                "split": "heldout" if i in heldout_set else "train",
                "metrics": metrics.to_dict(),
                "response_time": elapsed,
            })

            logger.get_logger().info(
                f"[+] F1: {metrics.retrieval_f1:.3f}, "
                f"Hallucination: {metrics.hallucination_rate:.3f}, "
                f"Response Time: {elapsed:.4f}s"
            )

        except Exception as e:
            logger.get_logger().error(
                f"[!] Error processing question {i+1}: {str(e)}"
            )

    # Aggregate metrics
    n = len(all_metrics)

    # Filter out None hallucination rates for aggregation
    valid_hallucination_metrics = [
        m for m in all_metrics if m.hallucination_rate is not None
    ]
    none_hallucination_count = n - len(valid_hallucination_metrics)
    if none_hallucination_count > 0:
        logger.get_logger().warning(
            f"[{corpus.corpus_id}] {none_hallucination_count}/{n} questions "
            f"returned hallucination_rate=None (detector failed). "
            f"These are excluded from the average."
        )

    avg_hallucination = (
        sum(m.hallucination_rate for m in valid_hallucination_metrics)
        / len(valid_hallucination_metrics)
        if valid_hallucination_metrics else None
    )

    metrics_summary = {
        "experiment": "baseline",
        "corpus_id": corpus.corpus_id,
        "num_questions": len(questions),
        "train_questions": len(split_info.get("train_indices", [])),
        "heldout_questions": len(split_info.get("heldout_indices", [])),
        "heldout_fraction": HELDOUT_FRACTION,
        "heldout_seed": HELDOUT_SEED,
        "avg_f1": sum(m.retrieval_f1 for m in all_metrics) / n if n else 0.0,
        "avg_bert_score": (
            sum(m.bert_score for m in all_metrics) / n if n else 0.0
        ),
        "bert_score_status_counts": {
            "computed": sum(1 for m in all_metrics if m.bert_score_status == "computed"),
            "skipped_missing_dependency": sum(
                1 for m in all_metrics
                if m.bert_score_status == "skipped_missing_dependency"
            ),
            "failed": sum(1 for m in all_metrics if m.bert_score_status == "failed"),
        },
        "avg_hallucination_rate": avg_hallucination if avg_hallucination is not None else 0.0,
        "hallucination_none_count": none_hallucination_count,
        "avg_semantic_similarity": (
            sum(m.semantic_similarity for m in all_metrics) / n if n else 0.0
        ),
        "avg_response_time": (
            sum(m.avg_response_time for m in all_metrics) / n if n else 0.0
        ),
    }

    high_f1_items = [
        r for r in per_question_results
        if r.get("metrics", {}).get("retrieval_f1", 0.0) > 0.5
    ]
    if high_f1_items:
        logger.get_logger().info(
            f"[{corpus.corpus_id}] Retrieval sanity check: "
            f"{len(high_f1_items)}/{len(per_question_results)} questions have F1 > 0.5."
        )
    else:
        logger.get_logger().warning(
            f"[{corpus.corpus_id}] Retrieval sanity check failed: "
            "no question achieved F1 > 0.5 after ID normalization."
        )

    return metrics_summary, per_question_results


def run_multimodal_experiment(
    corpus: Corpus,
    modality_combinations: List[List[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Run multimodal retrieval ablation study on a single corpus.

    Args:
        corpus: The evaluation corpus to use.
        modality_combinations: List of modality combinations to test.

    Returns:
        Dictionary with results for each combination.
    """
    logger.get_logger().info("\n" + "=" * 60)
    logger.get_logger().info(
        f"Running MULTIMODAL ABLATION STUDY on corpus '{corpus.corpus_id}'"
    )
    logger.get_logger().info("=" * 60)

    retriever = MultimodalGraphRetriever()
    results_by_combo = {}
    questions, references, relevant_items_list = corpus.get_expanded_data(
        min_questions=MIN_QA_PER_CORPUS
    )

    for combo in modality_combinations:
        combo_name = "+".join(combo)
        logger.get_logger().info(
            f"\nTesting modality combination: {combo_name}"
        )

        evaluator = EvaluationPipeline(
            f"multimodal_{corpus.corpus_id}_{combo_name}"
        )
        metrics_list = []

        for i, (question, reference) in enumerate(
            zip(questions, references)
        ):
            try:
                start = time.perf_counter()
                answer, metadata = retriever.answer_with_multimodal_context(
                    question, include_modalities=combo
                )
                elapsed = time.perf_counter() - start

                retrieved_context = metadata.get("context", "")
                assert retrieved_context != answer, (
                    "ERROR: retrieved_context must be different from generated "
                    "answer (circular evaluation detected)"
                )

                relevant = (
                    relevant_items_list[i]
                    if i < len(relevant_items_list)
                    else []
                )

                retrieved_nodes = metadata.get("retrieved_nodes", [])
                resolved_retrieved_nodes = _resolve_retrieved_node_labels(
                    retriever, retrieved_nodes
                )
                normalized_retrieved_nodes = _normalize_id_list(
                    resolved_retrieved_nodes
                )
                normalized_relevant = _normalize_id_list(relevant)

                metrics = evaluator.evaluate(
                    question=question,
                    generated_answer=answer,
                    reference_answer=reference,
                    retrieved_context=retrieved_context,
                    retrieved_items=normalized_retrieved_nodes,
                    relevant_items=normalized_relevant,
                    multimodal_context={
                        "text": metadata.get("text_content", ""),
                        "table": metadata.get("table_content", ""),
                        "image": metadata.get("image_content", ""),
                    },
                    response_time=elapsed,
                )
                metrics.corpus_id = corpus.corpus_id

                metrics_list.append(metrics)

            except AssertionError as e:
                logger.get_logger().error(
                    f"[!] Circular evaluation error: {str(e)}"
                )
            except Exception as e:
                logger.get_logger().warning(
                    f"Error processing question: {str(e)}"
                )

        # Aggregate
        if metrics_list:
            n = len(metrics_list)
            results_by_combo[combo_name] = {
                "combination": combo,
                "corpus_id": corpus.corpus_id,
                "num_questions": n,
                "avg_f1": sum(m.retrieval_f1 for m in metrics_list) / n,
                "avg_hallucination": (
                    sum(m.hallucination_rate for m in metrics_list) / n
                ),
                "avg_semantic_sim": (
                    sum(m.semantic_similarity for m in metrics_list) / n
                ),
                "avg_response_time": (
                    sum(m.avg_response_time for m in metrics_list) / n
                ),
                "text_usage": (
                    sum(m.text_modality_usage for m in metrics_list) / n
                ),
                "table_usage": (
                    sum(m.table_modality_usage for m in metrics_list) / n
                ),
                "image_usage": (
                    sum(m.image_modality_usage for m in metrics_list) / n
                ),
            }

            logger.get_logger().info(
                f"  [+] Avg F1: "
                f"{results_by_combo[combo_name]['avg_f1']:.3f}"
            )
            logger.get_logger().info(
                f"  [+] Avg Hallucination: "
                f"{results_by_combo[combo_name]['avg_hallucination']:.3f}"
            )

    return results_by_combo


# ------------------------------------------------------------------ #
# Aggregate helper
# ------------------------------------------------------------------ #
def _aggregate_summaries(
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute aggregate metrics across multiple per-corpus summaries."""
    n = len(summaries)
    if n == 0:
        return {}
    total_questions = sum(s.get("num_questions", 0) for s in summaries)
    return {
        "experiment": "baseline_aggregate",
        "num_corpora": n,
        "total_questions": total_questions,
        "avg_f1": sum(s.get("avg_f1", 0) for s in summaries) / n,
        "avg_bert_score": sum(s.get("avg_bert_score", 0) for s in summaries) / n,
        "bert_score_status_counts": {
            "computed": sum(
                s.get("bert_score_status_counts", {}).get("computed", 0)
                for s in summaries
            ),
            "skipped_missing_dependency": sum(
                s.get("bert_score_status_counts", {}).get(
                    "skipped_missing_dependency", 0
                )
                for s in summaries
            ),
            "failed": sum(
                s.get("bert_score_status_counts", {}).get("failed", 0)
                for s in summaries
            ),
        },
        "avg_hallucination_rate": (
            sum(s.get("avg_hallucination_rate", 0) for s in summaries) / n
        ),
        "avg_semantic_similarity": (
            sum(s.get("avg_semantic_similarity", 0) for s in summaries) / n
        ),
        "avg_response_time": (
            sum(s.get("avg_response_time", 0) for s in summaries) / n
        ),
    }


# ------------------------------------------------------------------ #
# Save results
# ------------------------------------------------------------------ #
def save_results(
    per_corpus_baseline: Dict[str, Tuple],
    aggregate_baseline: Dict[str, Any],
    per_corpus_multimodal: Dict[str, Dict],
    output_dir: str = "results",
) -> str:
    """
    Save comprehensive evaluation results.

    Args:
        per_corpus_baseline: {corpus_id: (summary, details)} dicts.
        aggregate_baseline: Aggregate metrics across all corpora.
        per_corpus_multimodal: {corpus_id: multimodal_results} dicts.
        output_dir: Output directory for results.

    Returns:
        Path to results file.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate_baseline": aggregate_baseline,
        "per_corpus_baseline": {
            cid: {"summary": data[0], "details": data[1]}
            for cid, data in per_corpus_baseline.items()
        },
        "per_corpus_multimodal": per_corpus_multimodal,
    }

    filepath = os.path.join(output_dir, "comprehensive_evaluation.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

    logger.get_logger().info(f"\n[+] Results saved to {filepath}")
    return filepath


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    """Run comprehensive evaluation experiments across all corpora."""
    logger.get_logger().info("GraphRAG Comprehensive Evaluation")
    logger.get_logger().info(f"Experiment ID: {logger.experiment_name}")

    corpora = get_all_corpora()
    logger.get_logger().info(
        f"Loaded {len(corpora)} corpora: "
        f"{[c.corpus_id for c in corpora]}"
    )

    # --- Log evaluation thresholds before starting ---
    logger.get_logger().info(
        f"Hallucination thresholds: "
        f"normal={config.evaluation.hallucination_threshold}, "
        f"graph={config.evaluation.graph_hallucination_threshold}"
    )

    per_corpus_baseline: Dict[str, Tuple] = {}
    per_corpus_multimodal: Dict[str, Dict] = {}
    baseline_summaries: List[Dict[str, Any]] = []

    for corpus in corpora:
        corpus_questions, corpus_references, corpus_relevant = corpus.get_expanded_data(
            min_questions=MIN_QA_PER_CORPUS
        )
        logger.get_logger().info(
            f"\n{'#' * 60}\n"
            f"# CORPUS: {corpus.corpus_id}  "
            f"({len(corpus_questions)} questions, expanded from {len(corpus.questions)})\n"
            f"{'#' * 60}"
        )

        # Baseline
        baseline_results = run_baseline_experiment(corpus)
        per_corpus_baseline[corpus.corpus_id] = baseline_results
        baseline_summaries.append(baseline_results[0])

        # --- Sanity check: all-zero hallucination rates ---
        summary = baseline_results[0]
        details = baseline_results[1]
        avg_hall = summary.get("avg_hallucination_rate", 0.0)
        none_count = summary.get("hallucination_none_count", 0)
        total_q = summary.get("num_questions", 0)

        if none_count == total_q and total_q > 0:
            logger.get_logger().warning(
                f"\n{'!' * 60}\n"
                f"WARNING [{corpus.corpus_id}]: ALL {total_q} hallucination rates are None "
                f"(detector failed on every question). "
                f"Verify the HallucinationDetector is running correctly.\n"
                f"{'!' * 60}"
            )
        elif avg_hall == 0.0 and total_q > 0:
            # Check if literally every valid result is 0.0
            per_q_rates = [
                r.get("metrics", {}).get("hallucination_rate")
                for r in details
            ]
            valid_rates = [r for r in per_q_rates if r is not None]
            if valid_rates and all(r == 0.0 for r in valid_rates):
                logger.get_logger().warning(
                    f"\n{'!' * 60}\n"
                    f"WARNING [{corpus.corpus_id}]: All {len(valid_rates)} hallucination "
                    f"rates are exactly 0.0 \u2014 this is suspicious. "
                    f"Verify the HallucinationDetector is running correctly and "
                    f"the threshold (graph: {config.evaluation.graph_hallucination_threshold}, "
                    f"normal: {config.evaluation.hallucination_threshold}) is reasonable.\n"
                    f"{'!' * 60}"
                )

        # Multimodal comparison pass (text+table+image)
        # Only run multimodal for the attention_paper corpus, which has
        # multimodal source material in this project setup.
        if corpus.corpus_id != "attention_paper":
            per_corpus_multimodal[corpus.corpus_id] = {
                "summary": {
                    "experiment": "multimodal",
                    "corpus_id": corpus.corpus_id,
                    "num_questions": 0,
                    "avg_bert_score": 0.0,
                    "bert_score_status_counts": {
                        "computed": 0,
                        "skipped_missing_dependency": 0,
                        "failed": 0,
                    },
                    "avg_hallucination_rate": 0.0,
                    "avg_semantic_similarity": 0.0,
                    "avg_f1": 0.0,
                    "avg_text_modality_usage": 0.0,
                    "avg_table_modality_usage": 0.0,
                    "avg_image_modality_usage": 0.0,
                    "skipped_reason": "Multimodal run is enabled only for attention_paper.",
                },
                "details": [],
            }
            continue

        # IMPORTANT: use the actual configured Neo4j database, not corpus ID.
        multimodal_retriever = MultimodalGraphRetriever(
            database=config.neo4j.database,
            corpus_id=corpus.corpus_id,
        )
        multimodal_evaluator = EvaluationPipeline(f"multimodal_{corpus.corpus_id}")
        multimodal_results: List[Dict[str, Any]] = []

        for i, (question, reference, relevant) in enumerate(
            zip(corpus_questions, corpus_references, corpus_relevant)
        ):
            logger.get_logger().info(
                f"\nMultimodal question {i+1}/{len(corpus_questions)}: {question[:50]}..."
            )
            try:
                start = time.time()
                answer, metadata = multimodal_retriever.answer_with_multimodal_context(
                    question, include_modalities=["text", "table", "image"]
                )
                elapsed = time.time() - start

                retrieved_context = metadata.get("context", "")
                retrieved_nodes = metadata.get("retrieved_nodes", [])
                resolved_retrieved_nodes = _resolve_retrieved_node_labels(
                    multimodal_retriever, retrieved_nodes
                )
                normalized_retrieved_nodes = _normalize_id_list(
                    resolved_retrieved_nodes
                )
                normalized_relevant = _normalize_id_list(relevant)

                metrics = multimodal_evaluator.evaluate(
                    question=question,
                    generated_answer=answer,
                    reference_answer=reference,
                    retrieved_context=retrieved_context,
                    retrieved_items=normalized_retrieved_nodes,
                    relevant_items=normalized_relevant,
                    multimodal_context={
                        "text": metadata.get("text_content", ""),
                        "table": metadata.get("table_content", ""),
                        "image": metadata.get("image_content", ""),
                    },
                    response_time=elapsed,
                )
                metrics.corpus_id = corpus.corpus_id

                multimodal_results.append({
                    "question": question,
                    "answer": answer,
                    "retrieved_context": retrieved_context,
                    "metrics": metrics.to_dict(),
                    "response_time": elapsed,
                })
            except Exception as e:
                logger.get_logger().warning(
                    f"Error processing multimodal question: {str(e)}"
                )

        multimodal_retriever.close()

        n_mm = len(multimodal_results)
        per_corpus_multimodal[corpus.corpus_id] = {
            "summary": {
                "experiment": "multimodal",
                "corpus_id": corpus.corpus_id,
                "num_questions": n_mm,
                "avg_hallucination_rate": (
                    sum(r["metrics"]["hallucination_rate"] for r in multimodal_results) / n_mm
                    if n_mm else 0.0
                ),
                "avg_semantic_similarity": (
                    sum(r["metrics"]["semantic_similarity"] for r in multimodal_results) / n_mm
                    if n_mm else 0.0
                ),
                "avg_f1": (
                    sum(r["metrics"]["retrieval_f1"] for r in multimodal_results) / n_mm
                    if n_mm else 0.0
                ),
                "avg_bert_score": (
                    sum(r["metrics"].get("bert_score", 0.0) for r in multimodal_results) / n_mm
                    if n_mm else 0.0
                ),
                "bert_score_status_counts": {
                    "computed": sum(
                        1
                        for r in multimodal_results
                        if r["metrics"].get("bert_score_status") == "computed"
                    ),
                    "skipped_missing_dependency": sum(
                        1
                        for r in multimodal_results
                        if r["metrics"].get("bert_score_status")
                        == "skipped_missing_dependency"
                    ),
                    "failed": sum(
                        1
                        for r in multimodal_results
                        if r["metrics"].get("bert_score_status") == "failed"
                    ),
                },
                "avg_text_modality_usage": (
                    sum(r["metrics"]["text_modality_usage"] for r in multimodal_results) / n_mm
                    if n_mm else 0.0
                ),
                "avg_table_modality_usage": (
                    sum(r["metrics"]["table_modality_usage"] for r in multimodal_results) / n_mm
                    if n_mm else 0.0
                ),
                "avg_image_modality_usage": (
                    sum(r["metrics"]["image_modality_usage"] for r in multimodal_results) / n_mm
                    if n_mm else 0.0
                ),
            },
            "details": multimodal_results,
        }

    # Aggregate baseline metrics
    aggregate_baseline = _aggregate_summaries(baseline_summaries)

    # Save results
    results_file = save_results(
        per_corpus_baseline, aggregate_baseline, per_corpus_multimodal
    )

    # ---- Print summary ---- #
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for cid, (summary, _details) in per_corpus_baseline.items():
        print(f"\n--- Corpus: {cid} ---")
        print(f"  F1 Score:            {summary['avg_f1']:.3f}")
        print(f"  BERTScore (F1):      {summary.get('avg_bert_score', 0):.3f}")
        print(
            f"  BERTScore Status:    "
            f"{summary.get('bert_score_status_counts', {})}"
        )
        print(f"  Hallucination Rate:  {summary['avg_hallucination_rate']:.3f}")
        print(f"  Semantic Similarity: {summary['avg_semantic_similarity']:.3f}")
        print(f"  Avg Response Time:   {summary['avg_response_time']:.3f}s")

    print(f"\n--- Aggregate (all corpora) ---")
    print(f"  F1 Score:            {aggregate_baseline.get('avg_f1', 0):.3f}")
    print(f"  BERTScore (F1):      {aggregate_baseline.get('avg_bert_score', 0):.3f}")
    print(
        f"  BERTScore Status:    "
        f"{aggregate_baseline.get('bert_score_status_counts', {})}"
    )
    print(
        f"  Hallucination Rate:  "
        f"{aggregate_baseline.get('avg_hallucination_rate', 0):.3f}"
    )
    print(
        f"  Semantic Similarity: "
        f"{aggregate_baseline.get('avg_semantic_similarity', 0):.3f}"
    )
    print(
        f"  Avg Response Time:   "
        f"{aggregate_baseline.get('avg_response_time', 0):.3f}s"
    )

    # Best multimodal config across all corpora
    all_mm_entries = []
    for cid, mm_dict in per_corpus_multimodal.items():
        summary = mm_dict.get("summary", {}) if isinstance(mm_dict, dict) else {}
        if summary:
            all_mm_entries.append((cid, summary))

    if all_mm_entries:
        best_config = max(
            all_mm_entries,
            key=lambda x: x[1].get('avg_f1', 0.0) - x[1].get('avg_hallucination_rate', 0.0),
        )
        print(f"\nBest Multimodal Configuration:")
        print(f"  Corpus/Modalities: {best_config[0]}/text+table+image")
        print(f"  F1 Score:          {best_config[1].get('avg_f1', 0.0):.3f}")
        print(
            f"  Hallucination Rate: "
            f"{best_config[1].get('avg_hallucination_rate', 0.0):.3f}"
        )

    logger.get_logger().info("\n[+] Evaluation complete!")


if __name__ == "__main__":
    main()
