---
**Authors**: Arnav Deshpande, Sarvesh Nimbalkar, Aadi Rawat, Dhruv Gadia  
**Institution**: Mukesh Patel School of Technology and Management, NMIMS, Mumbai  
**License**: MIT License  
**Contact**: deshpandearnavn@gmail.com  
**Last Updated**: March 2026  
---

# GraphRAG - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                           │
│         (CLI / Python API / REST API)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
   ┌────────────┐           ┌────────────┐
   │ Ingestion  │           │ Retrieval  │
   │ Pipeline   │           │ Pipeline   │
   └────────────┘           └────────────┘
        │                           │
        │                           │
    ┌───▼─────────────┐    ┌───────▼──────────┐
    │ Graph Storage   │◄───┤ Query Engine     │
    │ (Neo4j)         │    │ & LLM Back-end   │
    └─────────────────┘    └──────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Layer                         │
│  (Metrics, Benchmarks, Hallucination Detection)            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Ingestion Pipeline

**Purpose**: Convert raw documents into Neo4j knowledge graphs

**Flow**:
```
Raw Documents → Multimodal Processing → Graph Extraction → Neo4j Storage
```

**Modules**:
- `multimodal_ingestion.py`: Extracts text, tables, images from PDFs
- `graph_generator.py`: Converts documents to graph structures

**Key Operations**:
- PDF partitioning with structure inference
- Vision-based image/table summarization
- LLM-powered entity extraction
- Relationship type classification

### 2. Retrieval Pipeline

**Purpose**: Answer user questions using graph context

**Flow**:
```
User Query → Entity Extraction → Graph Traversal → Context Ranking → LLM Answer
```

**Modules**:
- `graph_retriever.py`: Base retrieval system
- `multimodal_retriever.py`: Enhanced with multimodal support

**Key Operations**:
- Query understanding and entity extraction
- Multi-hop relationship traversal
- Context aggregation across modalities
- Semantic relevance ranking

### 3. Evaluation Framework

**Purpose**: Measure system performance on multiple dimensions

**Metrics**:
- Retrieval quality (Precision, Recall, F1)
- Answer quality (ROUGE, BERTScore, Semantic Similarity)
- Hallucination detection and measurement
- Multimodal effectiveness

**Modules**:
- `evaluation/metrics.py`: All evaluation metrics
- `experiments/`: Benchmark and ablation scripts

## Data Flow

### Ingestion

```
Input: PDF/Text Documents
  │
  ├─► Multimodal Processing
  │   ├─► Text Extraction
  │   ├─► Image Extraction & Vision Summarization
  │   └─► Table Extraction & Structure Inference
  │
  ├─► Document Chunking
  │   └─► Maintains document boundaries
  │
  ├─► Graph Extraction (LLM)
  │   ├─► Entity Recognition
  │   ├─► Relationship Extraction
  │   └─► Property Assignment
  │
  ├─► Graph Serialization
  │   └─► JSON format with metadata
  │
  └─► Neo4j Ingestion
      ├─► Creates Entity nodes
      ├─► Creates Relationship edges
      ├─► Indexes for Query Performance
      └─► Source tracking for evaluation

Output: Neo4j Knowledge Graph
```

### Retrieval

```
Input: User Query
  │
  ├─► Query Understanding
  │   └─► LLM-based entity extraction
  │
  ├─► Graph Anchoring
  │   └─► Find initial nodes matching entity
  │
  ├─► Multi-hop Traversal
  │   ├─► Explore relationships up to max depth
  │   ├─► Collect connected nodes
  │   └─► Aggregate multimodal content
  │
  ├─► Context Ranking (Optional)
  │   ├─► Semantic relevance scoring
  │   ├─► Sort by relevance to query
  │   └─► Select top-k results
  │
  ├─► Context Formatting
  │   └─► Organize by modality type
  │
  └─► LLM Answer Generation
      ├─► Prompt with context
      ├─► Grounded generation
      └─► Output answer

Output: Grounded Answer + Metadata
```

## Neo4j Graph Schema

### Node Types

```cypher
Node Labels:
  ├─ Entity (Core entities)
  │   ├─ :Entity:Organization
  │   ├─ :Entity:Person
  │   ├─ :Entity:Location
  │   └─ :Entity:Concept
  │
  ├─ Content (Multimodal content)
  │   ├─ :TextBlock
  │   ├─ :Table
  │   ├─ :Image
  │   └─ :Document
  │
  └─ System
      ├─ :Source (Document source)
      └─ :Metadata
```

### Relationship Types

```cypher
Standard Relationships:
  ├─ :MENTIONS → Entity mentioned in content
  ├─ :RELATED_TO → General relationship
  ├─ :IS_A → Hypernym relationship
  ├─ :PART_OF → Part-whole relationship
  ├─ :CREATED_BY → Creation relationship
  │
Structural Relationships:
  ├─ :PRECEDES → Sequential ordering in document
  ├─ :NEXT → Next in sequence
  └─ :SOURCE → Links to document source

Properties on Relationships:
  ├─ weight: float (Importance/confidence)
  ├─ confidence: float (Extraction confidence)
  └─ type: string (Relationship subtype)
```

### Node Properties

```cypher
Entity Properties:
  ├─ id: String (Unique identifier)
  ├─ name: String
  ├─ type: String (Entity type)
  ├─ description: String
  ├─ extracted_from: List[String] (Source documents)
  └─ properties: Map[String, Any] (Additional attributes)

Content Properties:
  ├─ id: String
  ├─ content: String (Full text content)
  ├─ type: String (text/table/image)
  ├─ page_number: Integer
  ├─ summary: String (For multimodal content)
  └─ metadata: Map[String, Any]

Document Properties:
  ├─ id: String
  ├─ title: String
  ├─ source: String (File path)
  └─ ingestion_time: Datetime
```

## Configuration Management

Configuration is centralized in `config.py`:

```python
Config
  ├─ ModelConfig
  │   ├─ llm_model: str
  │   ├─ embedding_model: str
  │   └─ vision_model: str
  │
  ├─ Neo4jConfig
  │   ├─ uri: str
  │   ├─ username: str
  │   ├─ password: str
  │   └─ database: str
  │
  ├─ IngestionConfig
  │   ├─ graph_model: str
  │   ├─ extract_images: bool
  │   ├─ infer_table_structure: bool
  │   └─ ...
  │
  ├─ RetrievalConfig
  │   ├─ max_hop_distance: int
  │   ├─ relationships_limit: int
  │   ├─ use_semantic_ranking: bool
  │   └─ ...
  │
  └─ EvaluationConfig
      ├─ use_rouge: bool
      ├─ use_bert_score: bool
      ├─ detect_hallucinations: bool
      └─ hallucination_threshold: float
```

## Optimization Strategies

### 1. Graph Optimization

**Indexing**:
```cypher
CREATE INDEX entity_id_idx FOR (n:Entity) ON (n.id);
CREATE INDEX entity_type_idx FOR (n:Entity) ON (n.type);
```

**Query Optimization**:
- Use indexed fields in WHERE clauses
- Limit relationship traversal depth
- Batch operations for bulk ingestion

### 2. Retrieval Optimization

**Caching**:
- Cache frequently accessed nodes
- Store embedding vectors in graph
- Pre-compute popular entities

**Ranking**:
- Sort by relevance (BM25 + semantic)
- Prioritize high-confidence relationships
- Weight relationships by type

### 3. Processing Optimization

**Batching**:
- Process documents in batches
- Bulk Neo4j imports
- Parallel embedding computation

**Vector Stores**:
- Hybrid: Graph + Vector similarity
- Relationship vectors for ranking
- Entity embeddings for semantic search

## Error Handling

```python
PipelineError (all exceptions inherit)
  ├─ IngestionError
  │   ├─ DocumentLoadError
  │   ├─ GraphExtractionError
  │   └─ Neo4jIngestionError
  │
  ├─ RetrievalError
  │   ├─ EntityExtractionError
  │   ├─ GraphTraversalError
  │   └─ LLMAnswerError
  │
  └─ EvaluationError
      ├─ MetricsComputationError
      └─ DatasetLoadError
```

## Monitoring and Logging

```python
Experiment Logger:
  ├─ Configuration logging
  ├─ Pipeline progress tracking
  ├─ Metrics computation
  ├─ Error tracking
  └─ Results persistence
```

## Performance Benchmarks

Typical performance metrics:
- **Graph generation**: 5-10 seconds per 10K words
- **Neo4j ingestion**: 1-2 seconds per 1K nodes/edges
- **Query latency**: 0.2-0.5 seconds per question
- **Answer generation**: 2-5 seconds (LLM dependent)
- **Evaluation**: 1-2 seconds per metric

---

For usage examples, see [USAGE.md](USAGE.md)
For evaluation details, see [EVALUATION.md](EVALUATION.md)
