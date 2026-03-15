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
from typing import Dict, List, Tuple, Any
from pathlib import Path
from abc import ABC, abstractmethod

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag.config import config
from graphrag.retrieval import GraphRetriever, SemanticGraphRetriever, MultimodalGraphRetriever, get_graph_context, ask_llm_with_context
from graphrag.evaluation import EvaluationPipeline
from graphrag.utils import ExperimentLogger
from graphrag.utils.data_retriever import get_relevant_items_mapping, QUESTION_KEYWORDS_MAPPING

logger = ExperimentLogger("comprehensive_eval")


# ------------------------------------------------------------------ #
# Helper: strip citation markers like [cite: 17, 78]
# ------------------------------------------------------------------ #
_CITE_RE = re.compile(r'\s*\[cite:\s*[\d,\s]+\]', re.IGNORECASE)


def _strip_cite_markers(text: str) -> str:
    """Remove all ``[cite: ...]`` markers from *text*."""
    return _CITE_RE.sub('', text).strip()


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


class AttentionPaperCorpus(Corpus):
    """5 QA pairs about the 'Attention Is All You Need' paper."""

    @property
    def corpus_id(self) -> str:
        return "attention_paper"

    @property
    def questions(self) -> List[str]:
        return [
            "What are the main characteristics of the Transformer architecture?",
            "How does Multi-Head Attention relate to Scaled Dot-Product Attention?",
            "What is the performance significance of the Transformer model on the WMT 2014 English-to-German translation task?",
            "Compare the computational complexity per layer of self-attention layers and recurrent layers.",
            "What is the impact of masking in the decoder's self-attention sub-layer?",
        ]

    @property
    def references(self) -> List[str]:
        raw = [
            "The Transformer is a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely, and using stacked self-attention and point-wise, fully connected layers. [cite: 17, 78]",
            "Multi-Head Attention connects to Scaled Dot-Product Attention by linearly projecting queries, keys, and values h times, and performing the scaled dot-product attention function in parallel on each projected version. [cite: 126, 127]",
            "The Transformer model achieved a new state-of-the-art BLEU score of 28.4 on the WMT 2014 English-to-German translation task, improving over existing best results by over 2 BLEU. [cite: 19]",
            "Self-attention layers have a complexity of O(n^2 * d) per layer, while recurrent layers have a complexity of O(n * d^2), making self-attention faster when sequence length n is smaller than representation dimensionality d. [cite: 163, 187, 188, 189]",
            "Masking impacts the decoder by preventing positions from attending to subsequent positions, ensuring that predictions for position i can depend only on the known outputs at positions less than i, preserving the auto-regressive property. [cite: 88, 89]",
        ]
        return [_strip_cite_markers(r) for r in raw]

    @property
    def relevant_items(self) -> List[List[str]]:
        # Retrieve relevant items from actual graph data
        print("\n" + "=" * 60)
        print("Loading relevant items from graph data (Attention Paper)...")
        print("=" * 60)
        items = get_relevant_items_mapping(
            self.questions,
            question_keywords=QUESTION_KEYWORDS_MAPPING,
        )
        print("=" * 60 + "\n")
        return items


class TeslaCorpus(Corpus):
    """5 QA pairs about Tesla, derived from data/raw/Tesla.txt."""

    @property
    def corpus_id(self) -> str:
        return "tesla"

    @property
    def questions(self) -> List[str]:
        return [
            "What are Tesla's main product lines as of November 2024?",
            "What was Tesla's total revenue in 2024?",
            "What is the current status and timeline of Tesla's Full Self-Driving technology?",
            "Who leads Tesla and what is the company's leadership structure?",
            "Who are Tesla's main competitors and partners in the EV market?",
        ]

    @property
    def references(self) -> List[str]:
        return [
            "As of November 2024, Tesla offers six vehicle models: Model S, Model X, Model 3, Model Y, Semi, and Cybertruck. Tesla has also announced plans for a second-generation Roadster, the Cybercab, and the Robovan. Beyond vehicles, Tesla sells energy products including the Powerwall, Megapack, Solar Panels, and Solar Roof.",
            "Tesla reported total revenue of US$97.7 billion in 2024, with an operating income of US$7.1 billion and net income of US$7.1 billion. Total assets stood at US$122.1 billion with total equity of US$72.9 billion.",
            "Tesla's Full Self-Driving (Supervised) is an advanced driver-assistance system classified as SAE Level 2 automation, requiring continuous driver supervision. Since 2013, CEO Elon Musk has repeatedly predicted full autonomy (SAE Level 5) within one to three years, but these goals have not been met. The branding has drawn criticism for potentially misleading consumers. All Tesla vehicles produced after April 2019 include Autopilot.",
            "Tesla is led by CEO Elon Musk, who became chief executive in 2008 and owns approximately 13% of the company. Robyn Denholm serves as chair of the board of directors. The company was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning, and is headquartered in Austin, Texas.",
            "Tesla competes in the battery electric vehicle market, where it held a 17.6% market share in 2024. Key partners include battery suppliers Panasonic, CATL, and LG Energy Solution. Toyota and Daimler were former partners. Between May 2023 and February 2024, nearly all major North American EV manufacturers announced plans to adopt Tesla's North American Charging Standard connector.",
        ]


class GoogleCorpus(Corpus):
    """5 QA pairs about Google, derived from data/raw/Google.txt."""

    @property
    def corpus_id(self) -> str:
        return "google"

    @property
    def questions(self) -> List[str]:
        return [
            "How was Google Search originally developed and what algorithm did it use?",
            "What is Google's strategy for generative artificial intelligence?",
            "What is Google's advertising business model and how significant is it to revenue?",
            "What are Google's key products and services across different categories?",
            "What major antitrust actions has Google faced in the US and EU?",
        ]

    @property
    def references(self) -> List[str]:
        return [
            "Google began in January 1996 as a research project by Larry Page and Sergey Brin at Stanford University. They developed the PageRank algorithm, which determined a website's relevance by the number and importance of pages linking to it, rather than counting search term frequency. The search engine was originally nicknamed BackRub because the system checked backlinks. Google was incorporated on September 4, 1998, funded by an initial $100,000 investment from Andy Bechtolsheim.",
            "Following the success of ChatGPT, Google's senior management issued a code red and directed that all products with more than a billion users must incorporate generative AI within months. In March 2023, Google released Bard (now Gemini), a generative AI chatbot. Google has created the text-to-image model Imagen and the text-to-video model Veo. Google also released NotebookLM for synthesizing documents and developed LearnLM, a family of language models serving as personal AI tutors.",
            "Google generates most of its revenues from advertising, including sales of apps, in-app purchases, digital content products, and YouTube. In 2011, 96% of Google's revenue was derived from advertising programs. The primary advertising methods are AdMob, AdSense, and DoubleClick AdExchange. Google Ads allows advertisers to display advertisements through a cost-per-click scheme, while AdSense allows website owners to display ads and earn money per click.",
            "Google's key products span multiple categories: search (Google Search, News, Shopping), email (Gmail), navigation (Google Maps, Waze, Earth), cloud computing (Google Cloud), web browsing (Chrome), video sharing (YouTube), productivity (Workspace including Docs, Sheets, Slides), operating systems (Android, ChromeOS), hardware (Pixel phones, Nest smart home), AI (Google Assistant, Gemini), and cloud storage (Google Drive).",
            "In August 2024, a US federal judge ruled Google held an illegal monopoly over internet search in violation of Section 2 of the Sherman Antitrust Act. In September 2024, the EU Court of Justice imposed a 2.4 billion euro fine on Google for abusing its dominance in the shopping comparison market. The European Commission also fined Google 4.34 billion euros in 2018 for breaching EU antitrust rules related to Android device constraints, and 1.49 billion euros in 2019 for anti-competitive practices in online advertising.",
        ]


def get_all_corpora() -> List[Corpus]:
    """Return all available evaluation corpora."""
    return [
        AttentionPaperCorpus(),
        TeslaCorpus(),
        GoogleCorpus(),
    ]


# ------------------------------------------------------------------ #
# Experiment runners
# ------------------------------------------------------------------ #
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

    retriever = GraphRetriever()
    evaluator = EvaluationPipeline(f"baseline_{corpus.corpus_id}")

    per_question_results = []
    all_metrics = []
    relevant_items_list = corpus.relevant_items

    for i, (question, reference) in enumerate(
        zip(corpus.questions, corpus.references)
    ):
        logger.get_logger().info(
            f"\nQuestion {i+1}/{len(corpus)}: {question[:50]}..."
        )

        try:
            start = time.perf_counter()
            retrieved_context, sources, retrieved_nodes, _relations = (
                get_graph_context(
                    question,
                    retriever.client,
                    retriever.driver,
                    retriever.database,
                )
            )
            answer = ask_llm_with_context(
                question, retrieved_context, retriever.client
            )
            elapsed = time.perf_counter() - start

            assert retrieved_context != answer, (
                "ERROR: retrieved_context must be different from generated "
                "answer (circular evaluation detected)"
            )

            relevant = (
                relevant_items_list[i] if i < len(relevant_items_list) else []
            )

            metrics = evaluator.evaluate(
                question=question,
                generated_answer=answer,
                reference_answer=reference,
                retrieved_context=retrieved_context,
                retrieved_items=retrieved_nodes,
                relevant_items=relevant,
                response_time=elapsed,
            )
            metrics.corpus_id = corpus.corpus_id

            all_metrics.append(metrics)

            per_question_results.append({
                "question": question,
                "answer": answer,
                "retrieved_context": retrieved_context,
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
    metrics_summary = {
        "experiment": "baseline",
        "corpus_id": corpus.corpus_id,
        "num_questions": len(corpus),
        "avg_f1": sum(m.retrieval_f1 for m in all_metrics) / n if n else 0.0,
        "avg_hallucination_rate": (
            sum(m.hallucination_rate for m in all_metrics) / n if n else 0.0
        ),
        "avg_semantic_similarity": (
            sum(m.semantic_similarity for m in all_metrics) / n if n else 0.0
        ),
        "avg_response_time": (
            sum(m.avg_response_time for m in all_metrics) / n if n else 0.0
        ),
    }

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
    relevant_items_list = corpus.relevant_items

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
            zip(corpus.questions, corpus.references)
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

                metrics = evaluator.evaluate(
                    question=question,
                    generated_answer=answer,
                    reference_answer=reference,
                    retrieved_context=retrieved_context,
                    retrieved_items=metadata.get("retrieved_nodes", []),
                    relevant_items=relevant,
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

    per_corpus_baseline: Dict[str, Tuple] = {}
    per_corpus_multimodal: Dict[str, Dict] = {}
    baseline_summaries: List[Dict[str, Any]] = []

    modality_combos = [
        ["text"],
        ["text", "table"],
        ["text", "table", "image"],
        ["table"],
        ["image"],
    ]

    for corpus in corpora:
        logger.get_logger().info(
            f"\n{'#' * 60}\n"
            f"# CORPUS: {corpus.corpus_id}  "
            f"({len(corpus)} questions)\n"
            f"{'#' * 60}"
        )

        # Baseline
        baseline_results = run_baseline_experiment(corpus)
        per_corpus_baseline[corpus.corpus_id] = baseline_results
        baseline_summaries.append(baseline_results[0])

        # Multimodal ablation
        mm_results = run_multimodal_experiment(corpus, modality_combos)
        per_corpus_multimodal[corpus.corpus_id] = mm_results

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
        print(f"  Hallucination Rate:  {summary['avg_hallucination_rate']:.3f}")
        print(f"  Semantic Similarity: {summary['avg_semantic_similarity']:.3f}")
        print(f"  Avg Response Time:   {summary['avg_response_time']:.3f}s")

    print(f"\n--- Aggregate (all corpora) ---")
    print(f"  F1 Score:            {aggregate_baseline.get('avg_f1', 0):.3f}")
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
        for combo_name, combo_data in mm_dict.items():
            all_mm_entries.append((f"{cid}/{combo_name}", combo_data))

    if all_mm_entries:
        best_config = max(
            all_mm_entries,
            key=lambda x: x[1]['avg_f1'] - x[1]['avg_hallucination'],
        )
        print(f"\nBest Multimodal Configuration:")
        print(f"  Corpus/Modalities: {best_config[0]}")
        print(f"  F1 Score:          {best_config[1]['avg_f1']:.3f}")
        print(
            f"  Hallucination Rate: "
            f"{best_config[1]['avg_hallucination']:.3f}"
        )

    logger.get_logger().info("\n[+] Evaluation complete!")


if __name__ == "__main__":
    main()
