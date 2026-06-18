#!/usr/bin/env python3
"""
Master README generator for all pipeline stages.
Run this to regenerate all README files with current information.
"""

import json
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent

readmes = {
    "S1_PROFILER_README.md": """# S1 —Document Profiler

**Position:** Stage 1 of 7 (first)  
**Input:** Raw document text + optional config  
**Output:** Document profile (type, domain, 5 metrics, weights, suggestions)

## Overview

S1 analyzes document characteristics to guide all downstream decisions. It answers:
- **What type of document is this?** (code|table|mixed|prose)
- **What domain?** (legal, medical, technical, narrative, etc., 15 total)
- **How well-structured is it?** (5 quality metrics)
- **What hyperparameters should S2-S7 use?** (suggestions + confidence)

## Five Quality Metrics

Each scored [0, 1]. Domain determines weight given to each.

| Metric | Meaning | Weight Range |
|--------|---------|--------------|
| **RC** | Retrieval Cues — structural anchors (headers, articles, lists) | 5–35% |
| **ICC** | Intra-Chunk Coherence — sentence-level continuity | 10–30% |
| **DCC** | Document Coherence —topic flow across sections | 15–30% |
| **BI** | Block Integrity — text cleanliness, extraction quality | 15–25% |
| **SC** | Size Compliance — can chunks fit target sizes? | 10–25% |

### RC (Retrieval Cues)
Counts structural signals: Markdown headers (#), legal articles (Article N), lists (-*+), tables, code definitions.  
High RC → structure is explicit → boundar ies are obvious.

### ICC (Intra-Chunk Coherence)
Measures consecutive sentence overlap (Jaccard similarity).  
High ICC → sentences reference each other → natural text flow.

### DCC (Document Coherence)
Measures consecutive paragraph/section overlap.  
High DCC → topics chain smoothly → section boundaries are semantic.

### BI (Block Integrity)
Detects text cleanliness: balanced delimiters ({}, [], ()), sentence endings (. ! ?)

High BI → OCR-clean → boundaries are trustworthy.

### SC (Size Compliance)
What % of paragraphs fit target chunk size?  
High SC → natural units pack well → fewer forced merges.

## Domain Classification (15 domains)

Keyword-matched:
- **Highly structured:** legal, regulatory, policy
- **Moderate:** academic, research, medical, financial
- **Technical/product:** technical, cybersecurity, product, operations
- **Narrative/unstructured:** narrative, marketing, education, scientific

Each domain has different metric weights. Legal emphasizes RC (0.35); Narrative emphasizes ICC (0.30).

## Output Structure

```python
{
    "type": "prose",  # or code|table|mixed
    "domain": "regulatory",  # 15-domain classifier
    "domain_scores": {"legal": 0.82, "medical": 0.15, ...},
    "length_bucket": "medium",  # short (<1K) | medium (1-10K) | long (>10K)
    "token_count": 6478,
    
    "metrics": {
        "RC": 0.99,
        "ICC": 0.30,
        "DCC": 0.61,
        "BI": 0.92,
        "SC": 0.54,
        "overall": 0.67,
        "weighted_overall": 0.78,  # domain-weighted combo
    },
    
    "metric_weights": {  # Adaptive to domain
        "RC": 0.35,
        "ICC": 0.10,
        "DCC": 0.20,
        "BI": 0.25,
        "SC": 0.10,
    },
    
    "uncertainty": {  # 95% CI from bootstrap resampling
        "samples": 120,
        "weighted_overall_ci95": [0.74, 0.78, 0.77],
        "weighted_overall_std": 0.024,
    },
    
    "suggested_config": {  # Non-binding recommendations
        "n_min": 50,
        "n_max": 1000,
        "tau_jsd_low": 0.15,
        "tau_jsd_high": 0.4,
        "tau_sem": 0.75,
    }
}
```

## How Downstream Stages Use S1

- **S2** — Uses `type` + domain to select preferred chunking strategies
- **S3-S4** — Use `metrics` to calibrate boundary refining thresholds
- **S5-S6** — Use `domain` to select NLP/embedding strategies
- **S7** — Uses all fields for RL warm-starting and reward weighting

## Key Functions

- `profile_document(text, config)` — Main entry point
- `_classify_type(text)` → "code"|"table"|"mixed"|"prose"
- `_classify_domain(text)` → (best_domain, all_scores)
- `_compute_metrics(text, tokens, doc_type)` → {RC, ICC, DCC, BI, SC, overall}
- `_bootstrap_uncertainty(text, ...)` → {ci95, std}
- `_suggest_hyperparams(token_count, doc_type, metrics, config)` → suggested_config

## Performance

~150–400 ms per document (depends on bootstrap_samples, document length)
""",

    "S2_CHUNKERS_README.md": """# S2 — Chunking Strategy Selector

**Position:** Stage 2 of 7  
**Input:** Text + doc_type + domain (from S1) + config  
**Output:** Dict of chunks by strategy (8 strategies), then best selected

## Overview

S2 runs **all 8 chunking strategies in parallel** and **selects the best** for this document type + domain.

Strategies evaluated:
1. **structure** — Splits at structural anchors (headers, article numbers, code scopes)
2. **recursive** — Recursive binary splitting on entropy
3. **semantic_boundaries** — Semantic clustering + boundary detection
4. **sentence_clustering** — k-means clustering of sentence embeddings  
5. **sliding_window** — Fixed-size sliding window (fast, simple)
6. **paragraph_pack** — Pack paragraphs into target size buckets
7. **legal_articles** — Specialized for legal/regulatory (splits at articles)
8. **hybrid_legal_semantic** — Blends legal structure + semantic clustering

## Selection Logic

For each strategy, compute quality score:
```
score = quality_score(chunks, n_min, n_max) + strategy_bias
```

Where:
- **quality_score** = weighted combo of: size fit, boundary alignment, diversity
- **strategy_bias** = domain-tuned preference (legal docs get +0.06 for legal_articles, +0.10 for hybrid)

Select strategy with highest score.

## Output Structure

```python
{
    "structure": [chunk1, chunk2, ...],        # All chunks from structure strategy
    "recursive": [...],
    "semantic_boundaries": [...],
    "sentence_clustering": [...],
    "sliding_window": [...],
    "paragraph_pack": [...],
    "legal_articles": [...],
    "hybrid_legal_semantic": [...],
}
```

Then internally selects best & returns just that list.

## Each Chunk Dict

```python
{
    "text": "Full text of chunk...",
    "start": 1234,       # byte offset in original text
    "end": 5678,
    "method": "structure",  # which strategy produced this
    "chunk_index": 3,
    "token_count": 245,
}
```

## Key Functions

- `run_all_chunkers(text, doc_type, config)` → {strategy_name: [chunks]}
- `select_best_strategy(all_chunks, doc_type, config)` → [best_chunks]
- `_strategy_quality_score(chunks, n_min, n_max)` → [0, 1]

Strategy implementations:
- `_chunk_structure(text, config)`
- `_chunk_recursive(text, config)`
- `_chunk_semantic_boundaries(text, config)`
- … and 5 more

""",

    "S3_ENTROPY_README.md": """# S3 — Entropy-Based Boundary Refinement

**Position:** Stage 3 of 7  
**Input:** Chunks (from S2) + config  
**Output:** Chunks with refined boundaries + entropy annotations

## Overview

S3 uses **entropy-based analysis** to detect and fix bad boundaries. Where S2 made initial cuts, S3 asks:
- Is this boundary at a real topic shift (high entropy = good)?
- Or does merging adjacent chunks reduce entropy loss (bad boundary)?

Uses **Jensen-Shannon Divergence (JSD)** to measure vocab distribution shifts at boundaries.

JSD(p||q) ∈ [0, 1]:
- 0 = same vocab distribution (bad boundary, merge)
- 1 = completely different vocab (good boundary, keep or split more)

## Algorithm

For each chunk boundary:
1. Compute vocab distribution (term frequencies) on left side
2. Compute vocab distribution on right side
3. JSD(left||right) = entropy divergence
4. If JSD < tau_jsd_low → merge with next chunk
5. If JSD > tau_jsd_high → consider splitting (S3 can hard-split if entropy spike)
6. Else → keep boundary

Parameters:
- `tau_jsd_low` [0.05, 0.40] — merge threshold
- `tau_jsd_high` [0.20, 0.80] — split threshold
- These are typically set by S1 suggestions or S7 RL

## Output Annotations Per Chunk

```python
{
    "text": "...",
    "jsd_score": 0.47,           # Entropy divergence at boundary
    "boundary_type": "hard|soft",  # hard = protected, soft = flexible
    "merge_reason": "low_entropy_shift|moderate_entropy_shift|protected_structure_boundary",
    "boundary_features": {
        "pmi_drop": 0.45,        # Pointwise mutual information drop
        "entropy_rate": 0.52,    # Right side entropy
        "term_coverage": 0.61,   # How many terms are "new" vs repeated
    },
    ...
}
```

## Key Functions

- `refine_boundaries(chunks, config)` → refined_chunks
- `_compute_jsd(left_tokens, right_tokens)` → float
- `_should_merge(jsd, tau_low, tau_high)` → bool

""",

    "S4_BOUNDARY_README.md": """# S4 — Boundary Quality Filter

**Position:** Stage 4 of 7  
**Input:** Chunks with JSD annotations (from S3) + config  
**Output:** Same chunks, but with soft-boundary merges applied

## Overview

S4 **applies the merge decisions** that S3 identified. It filters out "bad" boundaries by merging adjacent chunks when:
- JSD < tau_jsd_low (low vocab divergence = not a real boundary)
- OR other quality signals (ICC, PMI, etc.) suggest merge

Also optionally:
- Splits chunks that are too large
- Reorders chunks if coherence improves

## Output

Same per-chunk structure as S3, but with merges applied:
- Some chunks are **combined** (if bad boundary detected)
- Some chunks are **split** (if too large or high-entropy spike)
- Some chunks are **unchanged**

Result: Usually fewer chunks, each with better coherence.

## Key Functions

- `filter_boundaries(chunks, doc_type, embeddings, config)` → filtered_chunks

""",

    "S5_GRAPH_README.md": """# S5 — Graph Enrichment & Entity Extraction

**Position:** Stage 5 of 7  
**Input:** Chunks (from S4) + optional embeddings + config  
**Output:** Same chunks, now with entity/relation metadata + graph enrichment

## Overview

S5 **enriches chunks with semantic metadata**:
- Extracts entities (persons, organizations, places, etc.)
- Detects relations between entities
- Updates KG store (knowledge graph)
- Computes graph-based relevance scores

Uses spaCy NLP pipeline (language auto-detected from text sample in config).

## Output Annotations Per Chunk

```python
{
    "text": "...",
    "entities": [
        {"text": "John Smith", "label": "PERSON", "start": 0, "end": 10},
        {"text": "Apple Inc.", "label": "ORG", "start": 50, "end": 60},
        ...
    ],
    "relations": [
        {"source": "John Smith", "relation": "WORKS_AT", "target": "Apple Inc."},
        ...
    ],
    "graph_neighbors": [1, 3, 5],  # indices of nearby chunks (KG-based)
    "typed_edges": [...],           # entity-to-entity edges within chunk
}
```

## KG Store

Persisted to `kg_store.json`:
- Nodes: entities found across all documents
- Edges: relations + co-occurrence stats
- Used by future documents for quick entity lookup

## Key Functions

- `enrich_graph(chunks, embeddings, config)` → enriched_chunks
- `_get_nlp()` — Lazy-load spaCy model (detects language from config)
- `_extract_entities(chunk_text, nlp)` → entities
- `_extract_relations(chunk_text, entities, nlp)` → relations

""",

    "S6_EMBEDDING_README.md": """# S6 — Contextual Embeddings

**Position:** Stage 6 of 7  
**Input:** Chunks (from S5) + full text + doc_profile + config  
**Output:** Same chunks, now with embedding vectors + similarity scores

## Overview

S6 **embeds each chunk** for later retrieval:
- Uses sentence transformers (all-MiniLM-L6-v2 by default or full model)
- Computes embeddings for each chunk
- Stores embeddings for retrieval indexing
- Computes inter-chunk similarities

Two embedding modes:
- **FAST_ENSEMBLE** (during S7 RL) — lightweight, 3–5 fast models
- **FULL_EMBEDDING** (final output) — heavier, single powerful model

## Output Annotations Per Chunk

```python
{
    "text": "...",
    "embedding": [0.12, -0.45, 0.23, ...],  # dense vector, dim=384 or 256
    "ensemble_embedding": {
        "vectors": [vec1, vec2, vec3],      # from 3-5 models during RL
        "aggregation": "mean|concat",
        "projection_dim": 256,
    },
    "similarity_to_neighbors": [0.87, 0.71, 0.64],  # cosine sim to adjacent chunks
}
```

## Key Functions

- `embed_chunks(chunks, text, doc_profile, model_name, config)` → (embedded_chunks, embedding_models)
- `preload_models(model_list)` — Pre-load models to avoid blocking
- `_embed_text(text, model)` → embedding_vector
- `_get_model(model_name, device)` → model

## Performance

- FAST_ENSEMBLE: ~50–100 ms per doc during RL
- FULL_EMBEDDING: ~200–500 ms per doc final output

""",

    "S7_RL_README.md": """# S7 — RL-Based Hyperparameter Calibration

**Position:** Stage 7 of 7 (final, optimizes S2–S6)  
**Input:** Text + doc_profile + chunks (from S2) + config  
**Output:** Best chunk set (from S2–S6 pipeline re-run with optimized hyperparams) + reward history

## Overview

S7 **tunes hyperparameters** (n_min, n_max, tau_jsd_low, tau_jsd_high, tau_sem, etc.) via **Bayesian Optimization (TPE)**.

For each trial:
1. Sample hyperparameter config (from Optuna TPE or random search)
2. Re-run S2–S6 pipeline with that config
3. Evaluate result via multi-objective reward function
4. TPE learns which regions of hyperspace → better rewards
5. After 10–20 trials, return best config's chunks

**Key insight:** S7 doesn't change S1 domain/type classification or strategy selection — it tunes *how* each strategy behaves.

## Multi-Objective Reward Function

```
reward = w_quality × quality
       + w_coverage × coverage
       + w_consistency × consistency
       + w_efficiency × efficiency
       + 0.10 × structural
```

Where:

| Component | Meaning | Range |
|-----------|---------|-------|
| **quality** | Chunk separation + internal coherence (ICC) | [0, 1] |
| **coverage** | Probe retrieval quality (precision-weighted) | [0, 1] |
| **consistency** | Low variance in chunk sizes | [0, 1] |
| **efficiency** | Proximity to target chunk count | [0, 1] |
| **structural** | Hard boundary ratio for legal/structured docs | [0, 1] |

Default weights: quality=0.35, coverage=0.25, consistency=0.20, efficiency=0.20

## Hyperparameter Search Space

T rial suggests values in:

| Param | Range | Purpose |
|-------|-------|---------|
| `tau_jsd_low` | [0.05, 0.40] | S3 merge threshold |
| `tau_jsd_high` | [0.20, 0.80] | S3 split threshold |
| `n_max` | [150, 900] | S2 max chunk size |
| `n_min` | [30, 250] | S2 min chunk size |
| `tau_sem` | [0.40, 0.95] | S4 merge threshold |
| `tau_percentile_low` | [5, 45] | S3 adaptive threshold lower %ile |
| `tau_percentile_high` | [55, 95] | S3 adaptive threshold upper %ile |

## Output Structure

```python
(
    best_chunks,       # List[Dict] — the best chunk set found
    reward_history,    # List[float] — reward at each trial (for frontend chart)
    final_config       # Dict — hyperparams used for best_chunks
)
```

final_config includes:
```python
{
    "tau_jsd_low": 0.15,
    "tau_jsd_high": 0.38,
    "n_max": 650,
    "n_min": 45,
    "tau_sem": 0.77,
    ...
    "reward_breakdown": {
        "quality": 0.68,
        "coverage": 0.72,
        "consistency": 0.81,
        "efficiency": 0.85,
        "structural": 0.51,
        "total": 0.72,
    },
    "n_trials_run": 10,
    "baseline_reward": 0.58,
    "best_reward": 0.72,
    "improvement_over_baseline": 0.14,
}
```

## Optimizer: Optuna TPE

Uses **Tree-structured Parzen Estimator**:
- Builds probabilistic surrogate model of reward landscape
- At each trial, samples hyperparams most likely to improve (Expected Improvement criterion)
- Warm-starts from past trials stored in `rl_history.json` (persisted per domain)
- Early stopping if no improvement > 0.005 for 6 consecutive trials (or min 8 trials done)

Fallback: Random search if Optuna not installed.

## Key Functions

- `run_rl_loop(text, doc_profile, initial_chunks, config)` →  (best_chunks, reward_history, final_config)
- `_run_optuna(...)` — TPE optimization loop
- `_run_random_search(...)` — Fallback random search
- `_suggest_config(trial, base_cfg)` — Ask Optuna for next config to try
- `_compute_reward_components(chunks, probes, weights)` → {quality, coverage, …, total}
- `_precision_recall_proxy(chunks, probes)` → coverage float
- `_generate_probes(text)` → List[str] — query strings for coverage evaluation

## Important Notes

- **Warm-start:** Previous document results (same domain) pre-seed TPE → faster convergence
- **Strategy fixed:** S7 typically doesn't change which strategy S2 selected; it tunes *parameters* of that strategy
- **Noise expected:** Reward history may fluctuate (0.50 → 0.74 → 0.60 is normal oscillation)
- **RL history** persisted to `rl_history.json` for future docs of same domain

""",
}

# Write all README files
for filename, content in readmes.items():
    filepath = PIPELINE_DIR / filename
    filepath.write_text(content, encoding="utf-8")
    print(f"✅ Updated {filename} ({len(content)} bytes)")

print(f"\n✅ All {len(readmes)} README files regenerated!")
