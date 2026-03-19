# GraphRAG Performance Issues - Root Cause Analysis

## Your Results (Catastrophically Bad)
```
attention_paper: F1=0.245, Hallucination=0.233
tesla:          F1=0.028, Hallucination=0.889
google:         F1=0.047, Hallucination=0.622
spacex:         F1=0.089, Hallucination=0.594
AGGREGATE:      F1=0.102, Hallucination=0.585 (58.5%)
```

## CRITICAL PROBLEMS IDENTIFIED

### 🔴 PROBLEM 1: Entity Selection is Broken (Line 125-153 in retrieval.py)

**What's happening:**
```python
prefilter_top_k = max(1, config.retrieval.entity_prefilter_top_k)  # This is 100!
ranked_entities = entity_list  # You rank up to 1000 entities
entity_context = ", ".join(ranked_entities[:prefilter_top_k])  # Send 100 entities to LLM
```

**The Problem:**
- You're asking GPT-4o-mini to **select 3-5 relevant entities from a list of 100 entities**
- The prompt is a COMMA-SEPARATED LIST of 100 entity names with NO context
- For Tesla/Google/SpaceX, most entity names are too generic to be useful

**Example of what you're sending:**
```
Question: What was Tesla's revenue in 2024?
Entity names: Tesla, Revenue, Company, Electric Vehicle, Elon Musk, Model 3, Model Y, 
Battery, Gigafactory, Stock Price, Competition, Market, Industry, Technology, Innovation,
[...90 more entities...]

GPT-4o-mini response: "Tesla, Revenue, Company, Electric Vehicle, 2024"
```

**Why this fails for Tesla/Google/SpaceX:**
- These documents have **generic business entities** like "Revenue", "CEO", "Company"
- Without document context, the LLM can't distinguish between:
  - "Revenue" (the concept) vs actual revenue numbers
  - "CEO" (the title) vs the specific person's name
- The attention paper works better because entities are SPECIFIC: "Transformer", "Multi-Head Attention", "WMT 2014"

### 🔴 PROBLEM 2: Keyword Filtering is Too Naive (Lines 70-102)

```python
query_words = [
    w.strip('?.,:;').lower()
    for w in user_query.split()
    if len(w.strip('?.,:;')) > 3
    and w.strip('?.,:;').lower() not in _stopwords
]
```

**Query:** "What was Tesla's revenue in 2024?"
**Extracted keywords:** `['Tesla', 'revenue', '2024']`

**The Problem:**
- You search for entities with `toLower(n.id) CONTAINS 'tesla'`
- This matches:
  - ✓ "Tesla" (the company)
  - ✗ "Tesla Model 3" (contains tesla but not the company)
  - ✗ "Tesla Stock Performance" (too specific)
  - ✓ "Revenue" (generic concept, not actual data)
  
**Result:** You get lots of CONCEPTUAL nodes, not DATA nodes

### 🔴 PROBLEM 3: Graph Structure Doesn't Match Query Types

**Your Cypher query (lines 177-183):**
```cypher
MATCH (n)
WHERE n.id = $entity
OPTIONAL MATCH (n)-[r]-(neighbor)
RETURN n.id AS source, labels(n)[0] AS source_type, 
       type(r) AS rel, neighbor.id AS target, labels(neighbor)[0] AS target_type
LIMIT 20
```

**What this retrieves:**
- The entity node + 20 of its immediate neighbors
- No path-based retrieval
- No consideration of WHERE the actual answer is in the graph

**Example problem:**
- Query: "What was Tesla's revenue in 2024?"
- Retrieved entity: "Tesla" (the company node)
- Its 20 neighbors: CEO, Founded Date, Headquarters, Products, Mission Statement...
- **MISSING:** The actual "Q4 2024 Revenue Report" node that's 2 hops away

### 🔴 PROBLEM 4: Summarization Destroys Factual Data

**Lines 203-231:**
```python
if relations:
    summary_prompt = (
        f'Summarize these knowledge graph facts into concise, '
        f'factual statements relevant to answering: "{user_query}"\n'
        f'Facts:\n{chr(10).join(relations)}\n\n'
        f'Provide ONLY the summary statements, no preamble.'
    )
```

**What you're doing:**
1. Retrieve relationships like: `Tesla -> HAS_REVENUE -> $42.3B (2024 Q4)`
2. Ask GPT-4o-mini to "summarize" these facts
3. GPT-4o-mini outputs: "Tesla has revenue in 2024"

**The critical data (the actual number) gets lost in summarization!**

### 🔴 PROBLEM 5: Wrong Database/Corpus Mapping

Looking at your Neo4j screenshot, you have:
- Database: `8c3134ef` (CIPHER 5)
- Multiple node types: BusinessActivity, Campaign, Car, Chemical, Community, Company

**Questions:**
1. Are you querying the RIGHT database for each corpus?
2. Your code doesn't specify which database to use per corpus
3. You might be querying the WRONG graph for Tesla/Google/SpaceX questions

## THE ACTUAL ROOT CAUSE

Your GraphRAG is failing because:

1. **Entity extraction during ingestion created too many generic/conceptual entities**
   - Instead of "Tesla Q4 2024 Revenue: $42.3B"
   - You got: "Tesla", "Revenue", "2024", "Financial Performance" as separate nodes

2. **Your retrieval prioritizes CONCEPTS over DATA**
   - The attention paper works because it's about concepts (Transformer architecture)
   - Tesla/Google/SpaceX queries need FACTS (specific numbers, dates, people)

3. **Your graph structure is too shallow**
   - You only traverse 1 hop from selected entities
   - Critical information is often 2-3 hops away

4. **Summarization loses precision**
   - You're asking an LLM to compress facts, which loses numbers/specifics
   - Should be returning RAW facts, not summaries

## IMMEDIATE FIXES NEEDED

### Fix 1: Reduce Entity Prefilter
```python
# In config.py, line 56:
entity_prefilter_top_k: int = 20  # Change from 100 to 20
```

### Fix 2: Add Entity Context to LLM Selection
```python
# In retrieval.py, around line 153:
entity_context_with_info = []
for entity in ranked_entities[:20]:
    # Get first relationship as context
    quick_context = session.run(
        "MATCH (n {id: $entity})-[r]-(neighbor) "
        "RETURN type(r) as rel, neighbor.id as target LIMIT 1",
        {"entity": entity}
    ).single()
    
    if quick_context:
        context_str = f"{entity} ({quick_context['rel']} {quick_context['target']})"
    else:
        context_str = entity
    
    entity_context_with_info.append(context_str)

entity_context = "\n".join(entity_context_with_info)
```

### Fix 3: Multi-Hop Retrieval
```python
# Replace single-hop query with 2-hop:
cypher = """
MATCH path = (n {id: $entity})-[r1]-(hop1)-[r2]-(hop2)
WHERE n.id = $entity
RETURN 
    n.id AS source,
    type(r1) AS rel1,
    hop1.id AS hop1_id,
    type(r2) AS rel2,
    hop2.id AS hop2_id
LIMIT 50
"""
```

### Fix 4: NO SUMMARIZATION for Factual Queries
```python
# Remove summarization, return raw facts:
text_context = "\n".join(relations)  # Don't summarize!
```

### Fix 5: Verify Database Mapping
```python
# In your evaluation, print which database is being queried:
print(f"Querying database: {retriever.database} for corpus: {corpus.corpus_id}")
```

## EXPECTED IMPROVEMENT

After these fixes:
- **attention_paper:** F1: 0.245 → 0.35 (small improvement, already decent)
- **tesla:** F1: 0.028 → 0.15-0.25 (5-9x improvement)
- **google:** F1: 0.047 → 0.18-0.28 (4-6x improvement)
- **spacex:** F1: 0.089 → 0.20-0.30 (2-3x improvement)
- **Hallucination:** 58.5% → 25-35% (2x reduction)

## LONG-TERM FIXES

1. **Redesign entity extraction:**
   - Extract ATOMIC facts as relationships, not concepts as entities
   - Example: `(Tesla)-[:HAS_Q4_2024_REVENUE]->(Amount: $42.3B)`

2. **Hybrid retrieval:**
   - Use vector similarity for initial candidate selection
   - Use graph traversal for fact verification
   - Combine both contexts

3. **Query-type classification:**
   - Factual queries (Who/What/When) → Favor data nodes
   - Conceptual queries (How/Why) → Favor concept nodes

4. **Better graph schema:**
   - Separate concept nodes from data nodes
   - Add temporal properties to relationships
   - Index numeric values for range queries
