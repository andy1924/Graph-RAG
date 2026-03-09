# Quick Start Guide - GraphRAG System

A research-grade Graph-based Retrieval-Augmented Generation system for question answering over structured knowledge graphs with full source attribution.

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and update with your credentials:
```bash
# .env configuration
OPENAI_API_KEY=your_openai_key_here
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=your_database_name
```

## Quick Usage

### Option 1: Simple Entity-Based Retrieval ⭐ (Recommended)
```bash
python main.py --mode simple
```

**Example interaction:**
```
Ask your graph a question: What is self-attention?

📊 SOURCES FROM KNOWLEDGE GRAPH:
  • Self-Attention -[USES]-> Encoder
  • Self-Attention -[RELIES_ON]-> Dot-Product Attention
  • Self-Attention -[APPROACH]-> Parallel Computing

💭 Retrieved Context Summary:
  Self-Attention is an approach in the Transformer model used in both 
  Encoder and Decoder that relies on Dot-Product Attention and supports 
  Parallel Computing...

✅ ANSWER:
  Based on the knowledge graph, Self-Attention is a core mechanism in the 
  Transformer architecture used in both encoder and decoder...
```

### Option 2: Multimodal Context-Based Retrieval
```bash
python main.py --mode multimodal
```

Provides richer context with multiple entity relationships for more comprehensive answers.

### Option 3: Data Ingestion (Load PDF into Graph)
```bash
python main.py --ingest
```

This:
1. Loads PDF document (Attention Is All You Need)
2. Extracts entities and relationships using LLM
3. Creates structured graph representation  
4. Ingests into Neo4j knowledge base

## System Architecture

The GraphRAG system implements **Retrieval-Augmented Generation (RAG)** with full source attribution:

```
1. ENTITY SELECTION
   ↓ All entities fetched from graph
   ↓ LLM selects entities relevant to question

2. RELATIONSHIP EXTRACTION  
   ↓ Graph relationships retrieved for selected entities
   ↓ Formatted as facts for context

3. CONTEXT SUMMARIZATION
   ↓ Relationships summarized into 2-3 concise statements
   ↓ Context optimized for LLM consumption

4. ANSWER GENERATION
   ↓ LLM generates answer ONLY from retrieved context
   ↓ Prevents hallucination, ensures verifiable answers

5. SOURCE ATTRIBUTION
   ↓ Citations show which graph entities support answer
   ↓ Full traceability of information
```

## Key Features

✅ **Research-Grade**: Full source citations and traceability  
✅ **Context-Aware**: Answers grounded in knowledge graph  
✅ **Multimodal**: Supports both simple and rich context retrieval  
✅ **Scalable**: Graph-based approach scales with knowledge size  
✅ **Verifiable**: Every answer backed by graph entities/relationships  

## Files Overview

| File | Purpose |
|------|---------|
| `main.py` | Entry point with CLI argument handling |
| `retrival.py` | Simple entity-based retrieval system |
| `multiModalGraphRetrival.py` | Multimodal context retrieval system |
| `ingest_attention.py` | Data ingestion pipeline for PDFs |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |

## Example Questions

The system works well with questions like:

- "What is self-attention?"
- "What are the components of a Transformer?"
- "What is positional encoding?"
- "What relationships exist between Multi-Head Attention and Layer Normalization?"
- "How does the encoder use embeddings?"

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'neo4j'`  
**Solution**: Activate your virtual environment and run `pip install -r requirements.txt`

**Issue**: `Neo4j connection failed`  
**Solution**: Verify `.env` file has correct URI, username, and password for your Neo4j instance

**Issue**: `OpenAI API key invalid`  
**Solution**: Check `.env` has valid `OPENAI_API_KEY` from your OpenAI account

## Research Applications

This system is designed for:
- Literature mining and knowledge extraction
- Question answering over technical papers
- Knowledge graph construction and querying
- Multimodal document analysis
- Evaluation of RAG system performance

## Further Reading

- See `README.md` for detailed architecture documentation
- See `RESEARCH_GOAL.md` for research objectives
- See `src/graphrag/` for module implementations
