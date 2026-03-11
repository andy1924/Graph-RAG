#!/usr/bin/env python
"""Test script to check which questions are being loaded."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.comprehensive_evaluation import BenchmarkDataset
import experiments.comprehensive_evaluation as eval_module

print(f"\nImported from: {eval_module.__file__}")

# Create dataset
dataset = BenchmarkDataset()

# Print questions
print("\n" + "=" * 60)
print("QUESTIONS IN DATASET:")
print("=" * 60)
for i, q in enumerate(dataset.questions, 1):
    print(f"{i}. {q}")

print("\n" + "=" * 60)
print("REFERENCES IN DATASET:")
print("=" * 60)
for i, r in enumerate(dataset.references, 1):
    print(f"{i}. {r[:100]}...")

print("\n" + "=" * 60)
print("QUESTION KEYWORDS MAPPING KEYS:")
print("=" * 60)
from graphrag.utils.data_retriever import QUESTION_KEYWORDS_MAPPING
for i, q in enumerate(QUESTION_KEYWORDS_MAPPING.keys(), 1):
    print(f"{i}. {q[:50]}...")
