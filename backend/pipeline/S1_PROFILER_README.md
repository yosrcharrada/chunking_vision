# S1 — Document Profiler

## Overview
**Pipeline position:** Entry point (stage 1 of 7)  
**Output to:** S2 (Chunking Strategy Selector)  
**Responsibility:** Analyze document characteristics and provide guidance for downstream stages

## What S1 Does

S1 analyzes a raw document and produces:
- **Document type classification** (prose, code, table, mixed)
- **Domain identification** (legal, medical, academic, financial, technical, narrative, regulatory, marketing, education, cybersecurity, product, operations, policy, research, scientific)
- **Five quality metrics** (RC, ICC, DCC, BI, SC)
- **Domain-specific metric weights** for balanced quality scoring
- **Confidence/uncertainty estimates** via bootstrap sampling
- **Hyperparameter suggestions** based on document properties

## The Five Quality Metrics

### 1. RC — Retrievability Coefficient
Measures document structure maturity and accessibility of key information.

- Detects section headers, code blocks, tables, bullet lists
- Scores 0–1: high RC = well-structured, low RC = amorphous prose
- **Domain impact:** Legal/regulatory/technical documents have naturally high RC (articles, APIs, etc.)

### 2. ICC — Intra-Chunk Coherence  
Sentence-level continuity within a chunk.

- Compares consecutive sentence embeddings (via hash or transformer)
- High ICC (0.7–1.0) = sentences form a tight topic thread
- Low ICC (0.0–0.3) = sentences are semantically diverse
- **Domain impact:** Legal documents often have low ICC (definitions referenced by number, not elaborated); narrative/medical have high ICC (dense explanation)

### 3. DCC — Document Coherence Coefficient
Inter-chunk thematic continuity.

- Measures topic flow: does concept A in chunk i carry forward to chunk j?
- Computed via rolling window of embedding similarities
- High DCC = story-like flow; low DCC = topic-jumping
- **Domain impact:** Research papers (hypothesis → methods → results) have high DCC

### 4. BI — Block Integrity
Resistance to OCR/extraction noise.

- Detects sentence fragments, orphaned punctuation, broken unicode
- High BI (0.8–1.0) = clean text; low BI (0.3–0.5) = scanned/corrupted
- **Domain impact:** Scanned legal PDFs and medical documents often have low BI

### 5. SC — Size Compliance  
Chunk size evenness across the document.

- Low variance in chunk sizes = better usability
- Penalises outliers (1-word vs 900-word chunks)
- High SC (0.7–1.0) = predictable unit sizes; low SC (0.2–0.4) = highly variable

## Domain-Specific Metric Weights

S1 adapts the relative importance of metrics based on detected domain:

| Domain | RC | ICC | DCC | BI | SC | Use Case |
|--------|----|----|-----|----|----|----------|
| **legal** | 0.35 | 0.10 | 0.20 | 0.25 | 0.10 | Articles, clauses, numbered provision structure |
| **regulatory** | 0.35 | 0.10 | 0.20 | 0.25 | 0.10 | Compliance docs, explicit article numbering |
| **academic** | 0.30 | 0.15 | 0.25 | 0.15 | 0.15 | Abstract/methods/results structure |
| **technical** | 0.15 | 0.25 | 0.25 | 0.20 | 0.15 | API docs, procedures, tight coupling of steps |
| **medical** | 0.20 | 0.25 | 0.25 | 0.20 | 0.10 | Clinical reports, diagnosis/treatment sections |
| **narrative** | 0.05 | 0.30 | 0.30 | 0.20 | 0.15 | Fiction/stories, flow is primary coherence |
| **financial** | 0.20 | 0.20 | 0.25 | 0.20 | 0.15 | Irregular tables/prose mix, irregular unit sizes |
| **product** | 0.20 | 0.20 | 0.20 | 0.15 | 0.25 | Backlog, user stories, high variance in sizes |

## Detected Domains

S1 can identify 15+ domains via keyword matching:
- **legal:** statute, clause, agreement, liability, jurisdiction, indemnity
- **medical:** patient, diagnosis, treatment, clinical, therapy, symptom
- **technical:** algorithm, API, repository, deployment, protocol, database
- **financial:** revenue, profit, equity, fiscal, balance sheet, cash flow
- **regulatory:** compliance, regulation, audit, governance, policy
- **academic:** abstract, methodology, citation, hypothesis, literature
- **research:** benchmark, model, inference, evaluation, ablation
- ... and 8 more

## Key Functions

### `profile_document(text, config) → Dict[str, Any]`
Main entry point. Analyzes the text and returns a complete profile dict:

```python
{
    "type": "prose",              # document type
    "domain": "legal",            # best-guess domain
    "domain_scores": {...},       # confidence per domain (normalized floats)
    "token_count": 6478,
    "length_bucket": "medium",    # "short" | "medium" | "long"
    "metrics": {
        "RC": 0.72,
        "ICC": 0.31,
        "DCC": 0.65,
        "BI": 0.89,
        "SC": 0.68,
        "weighted_overall": 0.65,
    },
    "metric_details": {...},      # per-metric breakdowns
    "adaptive_weights": {...},    # domain-adjusted weights used
    "uncertainty": 0.08,          # ±std dev estimate via bootstrap
}
```

### `_classify_type(text) → str`
Detects document structure:
- **"prose":** pure natural language
- **"code":** programming language blocks
- **"table":** CSV/markdown tables
- **"mixed":** prose + code/tables

### `_classify_domain(text, config) → Tuple[str, Dict[str, float]]`
Keyword-based domain detection. Returns:
- Best-matching domain name
- Dictionary of {domain: score} for all 15+ domains

### `_compute_metrics(text, tokens, doc_type) → Dict[str, float]`
Computes all five metrics (RC, ICC, DCC, BI, SC) for the full document.

### `_bootstrap_uncertainty(text, doc_type, weights, n_samples) → float`
Estimates confidence by resampling n_samples subsequences and measuring metric variance.

## Usage

**From command line:**
```bash
python -c "
from s1_profiler import profile_document
text = open('myfile.txt').read()
profile = profile_document(text, {})
print(profile['domain'], profile['metrics'])
"
```

**From another pipeline stage:**
```python
from .s1_profiler import profile_document

config = { ... pipeline config ... }
doc_profile = profile_document(text, config)
print(f"Domain: {doc_profile['domain']}")
print(f"Type: {doc_profile['type']}")
print(f"Overall quality: {doc_profile['metrics']['weighted_overall']}")
```

## Configuration

S1 respects these config keys:

| Key | Default | Purpose |
|-----|---------|---------|
| `bootstrap_samples` | 120 | Number of subsequence samples for uncertainty estimation |
| `domain_threshold` | 0.3 | Min keyword match score to accept a domain |
| `rc_threshold` | 0.5 | Min section density to classify as "structured" |

## Notes
- Domain detection is keyword-based, not learned. False positives possible for mixed documents.
- ICC computation requires embeddings (hash or transformer). Hash embedding is fast but coarse; provide a embedding_model in config for better quality.
- Metrics are normalized to [0, 1], but may cluster near extremes (e.g., legal documents cluster ICC ≈ 0.2–0.3, prose ≈ 0.6–0.8).
- Uncertainty estimates assume stable metric variance across the document.
