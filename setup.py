"""
Setup configuration for GraphRAG package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphrag",
    version="0.1.0",
    author="Research Team",
    description="Efficient Multi-Modal Graph-Based RAG for Mitigating LLM Hallucinations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/graphrag",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.1",
        "langchain-openai>=0.0.1",
        "langchain-neo4j>=0.0.1",
        "neo4j>=5.0",
        "openai>=1.0",
        "pydantic>=2.0",
        "python-dotenv>=1.0",
        "unstructured>=0.10",
        "pdf2image>=1.16",
        "pytesseract>=0.3",
        "sentence-transformers>=2.2",
        "scipy>=1.10",
        "numpy>=1.24",
    ],
    extras_require={
        "evaluation": [
            "rouge-score>=0.1",
            "bert-score>=0.3",
            "spacy>=3.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "graphrag-generate=graphrag.ingestion.graph_generator:main",
            "graphrag-query=graphrag.retrieval.graph_retriever:main",
            "graphrag-evaluate=graphrag.evaluation.metrics:main",
        ],
    },
)
