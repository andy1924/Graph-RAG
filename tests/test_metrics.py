"""
Verification test for the improved hallucination detection metric.

Tests:
  1. Factually-correct answer + noisy PDF context  → should be LOW hallucination
  2. Genuinely-fabricated answer + clean context    → should be HIGH hallucination
  3. Context normalisation smoke-test               → concatenated words split correctly

Usage:
    python test_hallucination_metric.py
"""

import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from graphrag.evaluation.metrics import HallucinationDetector


# ------------------------------------------------------------------ #
# Test data (taken from actual naiverag_evaluation.json results)
# ------------------------------------------------------------------ #

NOISY_PDF_CONTEXT = (
    "Figure1: TheTransformer-modelarchitecture.\n"
    "TheTransformerfollowsthisoverallarchitectureusingstackedself-attentionandpoint-wise,fully\n"
    "connectedlayersforboththeencoderanddecoder,shownintheleftandrighthalvesofFigure1,\n"
    "respectively.\n"
    "3.1 EncoderandDecoderStacks\n"
    "Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two\n"
    "sub-layers. Thefirstisamulti-headself-attentionmechanism,andthesecondisasimple,position-\n"
    "wisefullyconnectedfeed-forwardnetwork. Weemployaresidualconnection[11]aroundeachof\n"
    "the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is\n"
    "LayerNorm(x+Sublayer(x)),whereSublayer(x)isthefunctionimplementedbythesub-layer\n"
    "itself. Tofacilitatetheseresidualconnections,allsub-layersinthemodel,aswellastheembedding\n"
    "layers,produceoutputsofdimensiond =512.\n"
    "basedsolelyonattentionmechanisms,dispensingwithrecurrenceandconvolutions\n"
    "entirely.\n"
)

CORRECT_ANSWER = (
    "The main characteristics of the Transformer architecture include: "
    "it follows an encoder-decoder structure using stacked self-attention and "
    "point-wise, fully connected layers. "
    "The encoder is composed of a stack of N = 6 identical layers. "
    "Each layer has two sub-layers: a multi-head self-attention mechanism "
    "and a position-wise fully connected feed-forward network. "
    "Residual connections are employed around each sub-layer, followed by "
    "layer normalization. "
    "All sub-layers produce outputs of dimension d = 512. "
    "The Transformer is based solely on attention mechanisms, dispensing "
    "with recurrence and convolutions entirely."
)

FABRICATED_ANSWER = (
    "The Transformer architecture was invented by Facebook AI Research in 2020. "
    "It uses convolutional neural networks with 128 layers and processes data "
    "through a recurrent gating mechanism that relies on LSTM cells. "
    "The model has 50 billion parameters and is trained on ImageNet. "
    "It requires specialized quantum computing hardware to run efficiently."
)


def test_normalization():
    """Test that _normalize_context properly splits concatenated words."""
    raw = "TheTransformerfollowsthisoverallarchitectureusingstackedself-attention"
    cleaned = HallucinationDetector._normalize_context(raw)
    print(f"[NORM] Raw   : {raw}")
    print(f"[NORM] Clean : {cleaned}")

    # Check that common concatenated words are split
    assert "The Transformer" in cleaned or "The transformer" in cleaned.lower(), \
        f"Expected 'The Transformer' in cleaned text, got: {cleaned}"
    print("[PASS] Normalisation splits concatenated words correctly.\n")


def test_correct_answer_low_hallucination():
    """Factually correct answer against noisy PDF context → low hallucination."""
    print("=" * 60)
    print("TEST: Correct answer vs noisy PDF context")
    print("=" * 60)

    rate, ungrounded = HallucinationDetector.detect_unsupported_claims(
        CORRECT_ANSWER, NOISY_PDF_CONTEXT, threshold=0.7
    )

    print(f"  Hallucination rate : {rate:.3f}")
    print(f"  Ungrounded claims  : {len(ungrounded)}")
    for c in ungrounded:
        print(f"    - {c[:80]}...")

    assert rate < 0.5, (
        f"FAIL: Correct answer should have hallucination_rate < 0.5, got {rate:.3f}"
    )
    print(f"[PASS] Hallucination rate = {rate:.3f} (< 0.5 threshold)\n")


def test_fabricated_answer_high_hallucination():
    """Fabricated answer against clean context → high hallucination."""
    print("=" * 60)
    print("TEST: Fabricated answer vs context")
    print("=" * 60)

    clean_context = (
        "The Transformer is a transduction model based on attention mechanisms. "
        "It uses self-attention and multi-head attention. "
        "The encoder has six identical layers with residual connections."
    )

    rate, ungrounded = HallucinationDetector.detect_unsupported_claims(
        FABRICATED_ANSWER, clean_context, threshold=0.7
    )

    print(f"  Hallucination rate : {rate:.3f}")
    print(f"  Ungrounded claims  : {len(ungrounded)}")
    for c in ungrounded:
        print(f"    - {c[:80]}...")

    assert rate > 0.5, (
        f"FAIL: Fabricated answer should have hallucination_rate > 0.5, got {rate:.3f}"
    )
    print(f"[PASS] Hallucination rate = {rate:.3f} (> 0.5 threshold)\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Hallucination Metric Verification Tests")
    print("=" * 60 + "\n")

    test_normalization()
    test_correct_answer_low_hallucination()
    test_fabricated_answer_high_hallucination()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
