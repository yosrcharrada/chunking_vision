# S4 — Advanced Boundary Quality Filter

## Overview
**Pipeline position:** Stage 4 of 7  
**Input from:** S3 (Entropy Boundary Refinement)  
**Output to:** S5 (Entity Intelligence & Graph Enrichment)  
**Responsibility:** Score remaining boundaries using complementary lexical/structural/semantic signals and apply second-pass merge decisions

## Why S4 Exists After S3

S3 uses **distributional entropy signals** to make merge/split/soft decisions. S4 provides a **second opinion** using completely different signals:

- Lexical overlap (BLEU n-gram precision)
- Syntactic continuity (function-word patterns, bilingual French/English)
- Token type diversity (Jaccard set distance)
- Structural continuity (legal article/paragraph flow for prose)
- Semantic similarity (cosine distance or multi-scale lexical)

High S4 score → the two chunks are very similar → **merge candidate**.  
Low S4 score → the boundary is unambiguous → **keep the split**.

## Boundary Score Formula (Corrected)

```
weighted = 0.25×bleu + 0.20×syntactic + 0.15×token_type + 0.20×structural + 0.20×semantic
```

**Why this formula is corrected:**
- Original formula had **double-counting:** syntactic and token_type were computed inside _lexical_boundary_score, then weighted again
- This under-represented pure lexical overlap (BLEU) and over-weighted the composite
- New formula: each component is called ONCE with clean, non-overlapping signals

### Component Breakdown

#### 1. **BLEU (Bigram Lexical Precision)**
Fraction of bigrams in chunk A that appear in chunk B.
- High BLEU = repeated vocabulary = topic continuation
- Low BLEU = new vocabulary = topic shift
- Weight: 0.25

#### 2. **Syntactic Overlap (Bilingual Function Words)**
Overlap of function words (English: "the", "and", "is"; French: "le", "de", "et").
- High overlap = similar grammatical structure = likely topic continuation
- Low overlap = different structure = likely split point
- **Bilingual:** Works for both English and French legal documents (fix from previous version)
- Weight: 0.20

#### 3. **Token Type Diversity (Jaccard Distance)**
Set distance of open-class word types (nouns, verbs, adjectives).
- High Jaccard = similar vocabulary types = topic continuation
- Low Jaccard = different types = topic shift
- Weight: 0.15

#### 4. **Structural Continuity**
Legal prose only. Detects respect for article/paragraph boundaries.
- Legal marker precedence: Article > CHAPITRE > SECTION > numbered items
- High score = boundary respects legal structure
- Low score = arbitrary cut through middle of article
- Weight: 0.20

#### 5. **Semantic Similarity  (Multi-Scale)**
Average of:
- Hash-embedding cosine distance (fast, no model required)
- Multi-scale lexical score (evaluates windows of 1, 2, 3 chunks)
- Weight: 0.20

## Intra-Chunk Coherence (ICC)

Every chunk receives an **ICC (Intra-Chunk Coherence)** score:

```
ICC(chunk) = mean Jaccard(unit_i, unit_i+1) over consecutive units
```

Where **units** are sentences split using a **French-legal-aware splitter:**

- English: split on [.!?]
- French legal: also split on numbered items: 1), 2), 3)
  and on line breaks within numbered lists

**High ICC** (0.7–1.0) = sentences within the chunk are coherent  
**Low ICC** (0.1–0.3) = sentences are diverse (potential merge candidate)

ICC is used later by S7 in the reward function as a quality signal.

## Merge Decision

For each SOFT boundary from S3:

```
s4_score = weighted_boundary_score(chunk_A, chunk_B)
tau_sem = config["tau_sem"]  # e.g., 0.72

if s4_score > tau_sem:
    MERGE chunk_A and chunk_B
else:
    KEEP the boundary
```

**tau_sem is tuned by S7 RL** over range [0.40, 0.95].

## Output Schema

Each chunk is updated with:

```python
{
    "text": "...",
    "index": 0,
    "icc": 0.68,                          # intra-chunk coherence
    "boundary_score": 0.82,               # S4 similarity → previous chunk
    "merge_with_next": False,             # decision outcome
    "s4_details": {                       # component breakdown
        "bleu": 0.75,
        "syntactic": 0.68,
        "token_type": 0.71,
        "structural": 0.62,
        "semantic": 0.89,
        "weighted": 0.74,
    }
}
```

## Key Functions

### `filter_boundaries(chunks, doc_type, kg_chunks, config) → List[Dict]`
Main entry point. Scores all soft boundaries and merges where appropriate.

## Configuration

S4 respects these config keys:

| Key | Default | Purpose |
|-----|---------|---------|
| `tau_sem` | 0.72 | Merge threshold; boundaries > tau_sem are merged |

## Important Notes

1. **Bilingual Support:** Function-word set now includes French ("le", "la", "de", "pour", "que") for French legal documents
2. **Legal Structure Aware:** Prose documents with legal markers (Article, CHAPITRE) get structural continuity scoring
3. **ICC is Domain-Adaptive:** Legal documents expect low ICC (~0.2–0.4); narrative expects high ICC (~0.7+)
4. **No Circular Reward:** All S4 scores are computed independently of downstream stages
