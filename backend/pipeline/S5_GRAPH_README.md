# S5 — Entity Intelligence & Graph Enrichment

## Overview
**Pipeline position:** Stage 5 of 7  
**Input from:** S4 (Boundary Quality Filter)  
**Output to:** S6 (Ensemble Embeddings)  
**Responsibility:** Extract entities, build inter-chunk knowledge graph, enrich embeddings with graph context

## Architecture

S5 performs five steps:

1. **NER Extraction** — spaCy (en_core_web_sm) extracts named entities (PERSON, DATE, ORG, etc.)
2. **Entity Linking** — canonical form + head token for coreference grouping
3. **Relation Extraction** — lightweight subject–verb–object pattern matching
4. **Graph Construction** — chunks are nodes; entities/relations are weighted edges
5. **GNN Enrichment** — each chunk's embedding is enhanced by its graph neighborhood
6. **KGStore Persistence** — new entity co-occurrences saved to disk for future documents

## Entity Extraction

Entities are extracted using spaCy's NER model:

```python
nlp = spacy.load("en_core_web_sm")
doc = nlp(chunk_text)
for ent in doc.ents:
    entity_type = ent.label_  # PERSON, ORG, DATE, GPE, MONEY, etc.
    main_text = ent.text
```

**Known limitation:** The default model is English; French legal documents may mislabel generic French phrases.  
**Recommendation:** Install `fr_core_news_sm` and auto-detect language.

## Entity Linking

Each extracted entity is converted to a **canonical form** to group surface variants:

- "John Smith" and "Mr. Smith" → canonical: "smith" (head token)
- "United States" and "the US" → canonical: "united states"
- "Company Inc." and "Company" → canonical: "company"

Canonicalization reduces spurious edges due to surface variation.

## Relation Extraction

Lightweight SVO (subject–verb–object) pattern matching:

```
Subject: Named entity or noun phrase
Verb: Main verb phrase
Object: Named entity or noun phrase
```

Examples:
- "Smith founded Microsoft" → (Smith, founded, Microsoft)
- "The regulation requires compliance" → (regulation, requires, compliance)

Relations are extracted as features tying chunks together.

## Graph Construction

**Nodes:** Chunks (indexed by position)  
**Edges:** Three types with weights:

### 1. **shared_entity Edge**
Two chunks mention the same canonical entity.
- Weight = co-occurrence count
- Example: chunk 0 mentions "Microsoft", chunk 5 also mentions "Microsoft" → edge weight = 2

### 2. **relation_bridge Edge**
Two chunks mention the same (subject, object) pair.
- Weight = co-occurrence count
- Example: Both mention "(Microsoft, founded)" → edge weight increases

### 3. **kg_prior Edge** ⭐ NEW
Entity pair has historical co-occurrence in KGStore.
- Weight = historical co-occurrence count from previous documents
- Allows knowledge from past documents to influence current graph
- Example: If past documents linked (Microsoft, Bill Gates) 15 times, that edge is pre-boosted

## GNN-Style Enrichment

After graph construction, each chunk's embedding is enriched by **message-passing** from its neighbors:

```
h_graph(Ci) = Σ_{j ∈ N(i)} (w_ij / Σw_ij) × e(Cj)
```

Where:
- **N(i)** = neighbors of chunk i in the graph
- **w_ij** = edge weight between i and j
- **e(Cj)** = embedding of chunk j
- **h_graph(Ci)** = weighted average neighbor embedding

**Result:** A chunk "borrows" information from topically related chunks even if they are far apart in the document.

## Knowledge Graph Store (KGStore)

Persistent storage of entity co-occurrences across all pipeline runs.

### Format (kg_store.json)
```json
{
    "cooccurrence": {
        "microsoft": {"bill_gates": 15, "software": 8, ...},
        "regulation": {"compliance": 12, "audit": 7, ...},
        ...
    },
    "chunk_index": {
        "microsoft": ["run_001::C0", "run_001::C3", "run_002::C1", ...],
        "regulation": ["run_002::C2", ...],
        ...
    }
}
```

### Warm-Start Behavior
1. **First run (empty KGStore):** Graph built from document entities alone
2. **Second run:** KGStore pre-populated with run‑1 entities; edges are boosted
3. **Subsequent runs:** KGStore accumulates knowledge; entities gain stronger priors

**Benefits:**
- Repeated entities across documents get stronger signals
- Rare but important entity pairs persist for future reference
- Learning is cumulative across the document corpus

## Output Schema

Each chunk is augmented with:

```python
{
    "text": "...",
    "index": 0,
    "entities": [
        {"text": "Microsoft", "type": "ORG", "canonical": "microsoft", "start": 10, "end": 19},
        {"text": "Bill Gates", "type": "PERSON", "canonical": "gates", "start": 30, "end": 40},
    ],
    "graph_neighbors": [
        {"chunk_index": 3, "edge_type": "shared_entity", "entity": "microsoft", "weight": 2},
        {"chunk_index": 7, "edge_type": "relation_bridge", "relation": ("microsoft", "founded"), "weight": 1},
    ],
    "graph_vector": [0.12, -0.05, 0.31, ...],  # enriched embedding before S6
}
```

## Configuration

S5 respects these config keys:

| Key | Default | Purpose |
|-----|---------|---------|
| `nlp_model` | "en_core_web_sm" | spaCy model to load |
| `enable_kg_enhancement` | true | Include KGStore prior edges |
| `relation_extraction` | true | Extract SVO relations |

## Important Notes

1. **NER is English-centric:** French legal documents will have degraded entity recognition
2. **Lazy-loaded NLP:** spaCy model loads on first call, not at import time
3. **KGStore is cumulative:** entities accumulate across runs; use caution with data retention policies
4. **Graph enrichment is optional:** can be disabled for speed
