# S2 — Chunking Strategy Selector

## Overview
**Pipeline position:** Stage 2 of 7  
**Input from:** S1 (Document Profiler)  
**Output to:** S7 (RL Hyperparameter Calibration)  
**Responsibility:** Generate multiple expert-quality candidate chunk sets using diverse strategies

## Philosophy

S2 deliberately produces several strong chunking candidates rather than selecting one "best" strategy immediately. This is because:

1. **No single strategy is optimal for all documents** — legal articles benefit from structure-based splits, narrative prose from sentence clustering, technical docs from semantic boundaries.
2. **Downstream stages (S3–S6) score and refine** — S2 avoids making premature optimality claims.
3. **S7 (Bayesian Optimization) benchmarks all strategies** — competing candidates with tuned hyperparameters reveals which truly performs best.

## Eight Chunking Strategies

### 1. **Recursive Character Split**
Recursively bisects text at natural boundaries (.
, .\s, .). Resembles Langchain's recursive splitter.

- **n_min, n_max:** respect minimum/maximum chunk sizes
- **Structure awareness:** avoids splitting code blocks and legal articles  
- **Ideal for:** prose with paragraphs; technical documents with code
- **Typical output:** 40–60 chunks for a 6000-word document

### 2. **Sliding Window**
Fixed-size overlapping windows. Simple baseline.

- **Window size:** n_max
- **Overlap:** configurable (default 15% of n_max)
- **Ideal for:** dense reference documents needing broad context windows
- **Typical output:** ~15–20 chunks for a 6000-word document

### 3. **Structure-Based Split**
Respects document structure:
- Markdown headings (#, ##, ###)
- Legal boundaries (Article N, CHAPITRE, SECTION)
- Code function/class definitions
- Tables (CSV, markdown |...|)

- **n_min, n_max:** constrains chunk size after splits
- **Ideal for:** hybrid documents; legal/regulatory; technical
- **Typical output:** 20–35 chunks

### 4. **Semantic Boundaries**
Splits at peaks of semantic divergence using sentence embeddings.

- Computes cosine distance between consecutive sentence embeddings
- Splits when distance exceeds a threshold (tau_percentile signals)
- **Ideal for:** topic-transitional prose
- **Known limitation:** often produces 100+ micro-chunks because it finds every subtle topic edge; BO cannot tune to improve this
- **Status:** benchmarked but not optimized by S7

### 5. **Sentence Clustering**
Groups sentences into clusters using agglomerative clustering.

- Builds similarity matrix from sentence embeddings
- Clusters based on semantic affinity
- **Ideal for:** homogeneous prose
- **Known limitation:** produces 100+ micro-chunks regardless of parameters; BO cannot tune
- **Status:** benchmarked but not optimized by S7

### 6. **Paragraph Pack**
Groups contiguous paragraphs while respecting size constraints.

- Merges paragraphs greedily until approaching n_max
- Preserves paragraph boundaries
- **Ideal for:** well-formatted documents with clear paragraphs
- **Typical output:** 15–25 chunks

### 7. **Legal Article Split**
Specialized for legal/regulatory documents.

- Respects Article numbers, CHAPITRE breaks, TITRE boundaries
- Split points are locked to structural markers
- **Ideal for:** Tunisian regulations, contracts, compliance documents
- **Typical output:** 20–30 chunks (one per article or logical section)
- **Note:** n_max/n_min are soft constraints; cannot override article boundaries

### 8. **Hybrid Legal Semantic** ⭐ (NEW)
Structure-first + semantic refinement + size packing (custom strategy for this project).

1. **Phase 1 — Structure:** Respect legal/markdown boundaries (same as strategy #3)
2. **Phase 2 — Refinement:** Sub-split chunks using semantic boundaries (strategy #4)
3. **Phase 3 — Packing:** Repack to target size (strategy #6's greedy merge)
4. **Result:** Chunks respect structure but are semantically optimized and size-tuned

- **Tunable params:** n_max, n_min, tau_jsd_low, tau_jsd_high, tau_percentile_*
- **Ideal for:** legal/regulatory documents that need both structure and semantic coherence
- **Typical output:** 20–35 chunks, respecting articles but semantically coherent
- **Domain bias:** +0.10 favorability vs other strategies in S2 benchmark

## Strategy Selection Logic (`select_best_strategy()`)

S2 benchmarks all strategies using `_strategy_quality_score()`:

```
For each strategy S:
    chunks_s ← run strategy S with parameters (n_min, n_max, structure_type)
    score_s ← _strategy_quality_score(chunks_s, n_min, n_max)
    record (S, score_s)
    
winner ← argmax(score_s)
apply domain-tuned bias to winner:
    If domain == "legal" and strategy == "hybrid_legal_semantic":
        score_winner += 0.10
    If domain == "legal" and strategy == "legal_articles":
        score_winner += 0.06
        
best_strategy ← argmax(biased scores)
```

**Why biases?**
- Legal documents have strong prior: structure-respecting strategies (legal_articles, hybrid_legal_semantic) are more useful
- Bias is small (+0.06–0.10) so S2 winner can still be overridden if another strategy scores significantly higher
- Bias helps RL seed better initial hyperparameter ranges

### Quality Score Formula
```
_strategy_quality_score(chunks, n_min, n_max) = 
    0.30 × efficiency
    + 0.25 × consistency
    + 0.20 × size_validity
    + 0.20 × structural_coherence
    + 0.05 × sentence_boundary_respect
```

Where:
- **efficiency** = how close chunk count is to document-derived target (target = total_words / 300)
- **consistency** = 1 - coefficient_of_variation in chunk sizes
- **size_validity** = fraction of chunks within [n_min, n_max]
- **structural_coherence** = fraction of chunks respecting article/section boundaries
- **sentence_boundary_respect** = fraction of chunks avoiding mid-sentence cuts

## Key Functions

### `run_all_chunkers(text, doc_type, config) → Dict[str, List[Dict]]`
Main entry point. Runs all eligible strategies and returns:

```python
{
    "structure": [{"text": "...", "index": 0, ...}, ...],
    "recursive": [...],
    "semantic_boundaries": [...],
    "sentence_clustering": [...],
    "paragraph_pack": [...],
    "legal_articles": [...],  # if doc_type == "legal"
    "hybrid_legal_semantic": [...],  # if doc_type == "legal"
}
```

### `select_best_strategy(all_chunkers, doc_profile, n_min, n_max) → Tuple[str, List[Dict]]`
Returns (strategy_name, best_chunks).

## Configuration

S2 respects these config keys:

| Key | Default | Purpose |
|-----|---------|---------|
| `n_min` | 100 | Minimum tokens per chunk |
| `n_max` | 500 | Maximum tokens per chunk |
| `chunking_strategy` | "auto" | Force a specific strategy (bypass S2 selection) |
| `overlap_tokens` | 0 | Sliding window overlap (advanced) |
| `enable_semantic_boundary` | true | Include semantic_boundaries strategy |
| `enable_sentence_clustering` | false | Include sentence_clustering (slow) |

## Important Notes

1. **Excluded from S7 RL optimization:**
   - semantic_boundaries and sentence_clustering always produce 100+ micro-chunks
   - RL Bayesian Optimization cannot meaningfully tune parameters for these
   - They are benchmarked correctly in S2 and their scores used as baseline

2. **Domain-tuned biases:**
   - Legal documents: +0.10 for hybrid_legal_semantic, +0.06 for legal_articles
   - Regulatory documents: same biases as legal
   - Other domains: no bias (fair competition)

3. **Quality pass:**
   - After chunking, every chunk is filtered:
     - Removed if < n_min or > n_max (after filtering)
     - Removed if entirely lowercase (likely fragment)
     - Merged if both chunks are tiny (< n_min/2)

4. **Adaptive window sizing:**
   - If document is very short (< 2000 words), n_max is reduced
   - If document is code-heavy, n_min is relaxed to allow small code snippets

## Output Schema

Each chunk dict contains:

```python
{
    "text": "...",                  # chunk content
    "index": 0,                     # position in strategy's output
    "strategy": "hybrid_legal_semantic",
    "token_count": 250,
    "has_code": False,
    "has_table": False,
}
```
