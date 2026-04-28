"""
S4 — Advanced Boundary Quality Filter
======================================
Pipeline position : runs AFTER S3 (entropy refinement) and BEFORE S5 (graph).
Responsibility    : score each remaining boundary using a multi-component
                    lexical + structural + semantic signal, then merge adjacent
                    chunks whose combined score exceeds the semantic similarity
                    threshold τ_sem (i.e. they are too similar to keep separate).

Why S4 exists after S3
───────────────────────
S3 uses distributional entropy signals and an LSTM to make merge/hard/soft
decisions.  "Soft" boundaries are passed here for a SECOND opinion using
complementary signals:
  - n-gram overlap (BLEU-inspired lexical continuity)
  - syntactic function-word patterns
  - structural continuity (brace/markup balance for code/mixed docs)
  - semantic score (embedding cosine similarity or cross-encoder)
  - multi-scale boundary score (evaluates windows of 1, 2, 3 chunks)

The composite score is used BOTH to decide whether to merge (score > τ_sem)
AND to annotate each chunk with a "boundary_score" field used by S7's reward.

Score formula
──────────────
    weighted = α·lexical + β·syntactic + γ·token_type + δ·structural + ε·semantic_multiscale
where:
    α = 0.25  (n-gram BLEU-style overlap)
    β = 0.20  (syntactic function-word overlap)
    γ = 0.15  (token type set Jaccard)
    δ = 0.20  (structural continuity — code/markup only, else 0.5)
    ε = 0.20  (average of embedding cosine and multi-scale lexical)

High score → the two chunks are very similar → merge candidate.
Low score  → the boundary is valid → keep the split.

Intra-Chunk Coherence (ICC)
────────────────────────────
Every chunk also receives an "icc" field:
    ICC(chunk) = mean Jaccard(sᵢ, sᵢ₊₁) over consecutive sentence pairs
High ICC → the sentences WITHIN the chunk are topically coherent.
ICC is used by S7 as a quality signal in the RL state vector.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

# ── Composite score weights ───────────────────────────────────────────────────
# These sum to 1.0.  Change them here to re-tune the boundary decision logic.
_ALPHA   = 0.25   # lexical (BLEU n-gram)
_BETA    = 0.20   # syntactic (function-word overlap)
_GAMMA   = 0.15   # token type (set Jaccard)
_DELTA   = 0.20   # structural continuity
_EPSILON = 0.20   # semantic / multi-scale

# Lazy-loaded cross-encoder model (loaded only when embeddings are unavailable)
_cross_encoder = None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def filter_boundaries(
    chunks: List[Dict],
    doc_type: str,
    embeddings: Optional[List[List[float]]],
    config: Dict[str, Any],
) -> List[Dict]:
    """
    Score every boundary and optionally merge high-similarity adjacent chunks.

    Parameters
    ──────────
    chunks     : list of chunk dicts from S3 (each has "text" and boundary metadata)
    doc_type   : "prose" | "code" | "table" | "mixed"  (affects structural scoring)
    embeddings : optional list of embedding vectors (one per chunk, same order)
                 If provided, semantic score uses cosine similarity.
                 If None, falls back to cross-encoder or hash-based similarity.
    config     : pipeline config dict

    Returns
    ───────
    List of chunk dicts with added fields:
      boundary_score      — composite similarity score ∈ [0,1] (high=similar=merge candidate)
      icc                 — intra-chunk coherence score ∈ [0,1]
      boundary_breakdown  — dict of individual component scores for debugging
    Chunks whose boundary_score > τ_sem AND whose combined size ≤ 1.5×n_max are merged.
    """
    if not chunks:
        return chunks

    # Read thresholds from config
    tau_sem     = float(config.get("tau_sem", 0.75))         # merge-similarity threshold
    n_max       = int(config.get("n_max", 500))              # max tokens per chunk
    merge_weight = float(config.get("boundary_merge_weight", 1.0))  # global scale on score

    # Initialise output list with the first chunk (no left neighbour to compare)
    result: List[Dict] = [dict(chunks[0])]
    result[0]["boundary_score"]    = 0.0   # first chunk has no left boundary
    result[0]["icc"]               = _compute_icc(result[0]["text"])
    result[0]["boundary_breakdown"] = {}

    # Iterate from chunk index 1 onward: compare each chunk with its predecessor
    for idx in range(1, len(chunks)):
        prev = result[-1]             # the chunk currently at the end of result[]
        curr = dict(chunks[idx])      # the chunk we are evaluating

        # ── Component scores ─────────────────────────────────────────────
        # 1. Lexical: BLEU-inspired n-gram precision + syntactic + token type
        lexical    = _lexical_boundary_score(prev["text"], curr["text"], doc_type)

        # 2. Structural: brace/markup balance for code/mixed; 0.5 for prose
        structural = _structural_continuity_score(prev["text"], curr["text"], doc_type)

        # 3. Semantic: embedding cosine if available, else cross-encoder, else hash
        semantic   = _semantic_score(prev["text"], curr["text"], embeddings, idx)

        # 4. Multi-scale: lexical score computed at windows 1, 2, 3 chunks wide
        #    captures context beyond the immediate pair
        multiscale = _multi_scale_boundary_score(chunks, idx, doc_type)

        # ── Weighted composite score ──────────────────────────────────────
        # Note: _syntactic_overlap and _token_type_match are called again here
        # (they were already called inside _lexical_boundary_score) to allow
        # their contributions to be individually reported in boundary_breakdown.
        weighted = (
            _ALPHA   * lexical
            + _BETA  * _syntactic_overlap(prev["text"], curr["text"], doc_type)
            + _GAMMA * _token_type_match(prev["text"], curr["text"])
            + _DELTA * structural
            + _EPSILON * ((semantic + multiscale) / 2.0)
        )

        # Apply the global merge_weight scale and clamp to [0, 1]
        decision_score = float(np.clip(weighted * merge_weight, 0.0, 1.0))

        # ── Annotate current chunk ────────────────────────────────────────
        curr["boundary_score"] = round(decision_score, 4)
        curr["icc"]            = _compute_icc(curr["text"])
        curr["boundary_breakdown"] = {
            "lexical":     round(float(lexical),     4),
            "structural":  round(float(structural),  4),
            "semantic":    round(float(semantic),    4),
            "multiscale":  round(float(multiscale),  4),
            "weighted":    round(float(decision_score), 4),
        }

        # ── Merge decision ────────────────────────────────────────────────
        # Merge if: score > τ_sem (very similar) AND combined size fits
        prev_wc = len(prev["text"].split())
        curr_wc = len(curr["text"].split())

        if decision_score > tau_sem and (prev_wc + curr_wc) <= n_max * 1.5:
            # Merge: absorb curr into the previous chunk in result[]
            result[-1]["text"]               = prev["text"] + "\n\n" + curr["text"]
            result[-1]["end"]                = curr.get("end", prev.get("end", 0))
            result[-1]["boundary_score"]     = round(decision_score, 4)
            result[-1]["boundary_breakdown"] = curr["boundary_breakdown"]
            # ICC of the merged chunk will be recomputed if needed downstream;
            # for now, keep the previous chunk's ICC
        else:
            # Keep the split: append curr as a separate chunk
            result.append(curr)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Component score functions
# ─────────────────────────────────────────────────────────────────────────────

def _lexical_boundary_score(text1: str, text2: str, doc_type: str) -> float:
    """
    Composite lexical score combining three sub-signals:
      0.4 × BLEU bigram precision
      0.3 × syntactic function-word overlap
      0.3 × token type set Jaccard

    High score = the two texts share a lot of vocabulary → similar topic.
    """
    bleu = _ngram_precision(text1, text2, n=2)   # bigram BLEU-style precision
    syn  = _syntactic_overlap(text1, text2, doc_type)
    ttm  = _token_type_match(text1, text2)
    return float(0.4 * bleu + 0.3 * syn + 0.3 * ttm)


def _ngram_precision(text1: str, text2: str, n: int = 2) -> float:
    """
    Clipped n-gram precision: the fraction of n-grams in text1 that also
    appear in text2, clipped so that each n-gram in text2 is counted at most
    as many times as it appears there.

    This is the BLEU unigram/bigram precision component (Papineni et al. 2002).
    High precision → text1's n-grams are well-covered by text2 → similar content.

    Falls back to Jaccard token overlap when either text is shorter than n.
    """
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    # Fallback for very short texts
    if len(tokens1) < n or len(tokens2) < n:
        s1, s2 = set(tokens1), set(tokens2)
        union = s1 | s2
        return len(s1 & s2) / len(union) if union else 0.0

    # Build n-gram frequency counters
    ngrams1 = Counter(_ngrams(tokens1, n))
    ngrams2 = Counter(_ngrams(tokens2, n))

    # Clipped count: for each n-gram in text1, count min(count_in_1, count_in_2)
    clipped = sum(min(c, ngrams2[ng]) for ng, c in ngrams1.items())
    total   = sum(ngrams1.values())
    return clipped / total if total else 0.0


def _ngrams(tokens: List[str], n: int):
    """Extract all n-grams from a token list as tuples."""
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _syntactic_overlap(text1: str, text2: str, doc_type: str) -> float:
    """
    Measure overlap of syntactically functional tokens between the two texts.

    For CODE: overlaps ALL tokens including operators and brackets (AST-level).
    For PROSE: overlaps a set of English function words (determiners, auxiliaries,
               prepositions) that signal grammatical continuity.

    Rationale: if two adjacent chunks share the same function-word pattern,
    they are likely continuation text (same syntactic frame → merge candidate).
    If they diverge, they may be starting a new grammatical context.
    """
    if doc_type == "code":
        # For code: include operators and punctuation in the token set
        t1 = set(re.findall(r"[+\-*/%&|^~<>=!;:,.()\[\]{}]|\b\w+\b", text1))
        t2 = set(re.findall(r"[+\-*/%&|^~<>=!;:,.()\[\]{}]|\b\w+\b", text2))
        union = t1 | t2
        return len(t1 & t2) / len(union) if union else 0.0

    # For prose: use English function words as syntactic markers
    func = {
        "the", "a", "an", "is", "was", "are", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "of", "in",
        "on", "at", "by", "for", "with", "about", "as", "to",
    }

    # Extract function tokens from each text
    t1 = [w for w in _tokenize(text1) if w in func]
    t2 = [w for w in _tokenize(text2) if w in func]

    # Clipped count (same logic as BLEU)
    c1, c2 = Counter(t1), Counter(t2)
    shared = sum(min(c1[w], c2[w]) for w in c1)
    total  = max(len(t1), len(t2))
    return shared / total if total else 0.0


def _token_type_match(text1: str, text2: str) -> float:
    """
    Jaccard similarity of the TYPE (vocabulary) sets of the two texts:
        J(V1, V2) = |V1 ∩ V2| / |V1 ∪ V2|

    Unlike _ngram_precision (which counts INSTANCES), this measures what
    fraction of the unique vocabulary is shared.

    High Jaccard → the two chunks talk about the same concepts (merge candidate).
    """
    t1    = set(_tokenize(text1))
    t2    = set(_tokenize(text2))
    union = t1 | t2
    return len(t1 & t2) / len(union) if union else 0.0


def _structural_continuity_score(text1: str, text2: str, doc_type: str) -> float:
    """
    Detect structural continuity between text1 and text2.

    Only meaningful for CODE, MIXED, and TABLE doc types.
    For PROSE, returns a neutral 0.5.

    Three sub-checks combined:
    1. Brace balance delta: if text1 has unmatched { }, it likely continues
       into text2 → high continuity.
    2. Markdown bridge: if text2 starts with a heading, it opens a new block
       → lower continuity (0.6).
    3. XML/HTML bridge: if both texts contain tags, they may be part of the
       same markup structure → higher continuity (0.8 vs 0.4).
    """
    if doc_type not in {"code", "mixed", "table"}:
        return 0.5   # neutral for prose — structural continuity not applicable

    # Brace imbalance: open braces not closed in text1 suggest continuation
    brace_delta  = (abs(text1.count("{") - text1.count("}"))
                    + abs(text2.count("{") - text2.count("}")))

    # Markdown: if text2 opens a new heading, it's a new block (lower continuity)
    markdown_bridge = 1.0 if re.search(r"^#{1,6}\s", text2, re.MULTILINE) else 0.6

    # XML/HTML: shared tags suggest structural continuity
    xml_bridge = (0.8 if ("<" in text1 and ">" in text1
                          and "<" in text2 and ">" in text2) else 0.4)

    # Continuity from brace balance: 0 imbalance → 1.0, 6+ imbalance → 0.0
    cont = 1.0 - min(1.0, brace_delta / 6.0)

    return float(np.clip((cont + markdown_bridge + xml_bridge) / 3.0, 0.0, 1.0))


def _semantic_score(
    text1: str,
    text2: str,
    embeddings: Optional[List[List[float]]],
    idx: int,
) -> float:
    """
    Semantic similarity between text1 and text2.

    Priority:
    1. If embeddings[] is provided → cosine similarity between embedding vectors.
       Most accurate and cheapest at inference time.
    2. Cross-encoder (CrossEncoder/stsb-distilroberta-base) if available.
       Slower but captures deep semantic similarity.
    3. Hash-embedding cosine fallback (always available, no dependencies).

    Returns a value ∈ [0, 1].  High = semantically similar = merge candidate.
    """
    # Option 1: use pre-computed embeddings from S6 (or empty list from RL loop)
    if embeddings and (idx - 1) < len(embeddings) and idx < len(embeddings):
        return _cosine_similarity(embeddings[idx - 1], embeddings[idx])

    # Option 2: cross-encoder model (lazy-loaded)
    ce = _cross_encoder_similarity(text1, text2)
    if ce is not None:
        return ce

    # Option 3: lightweight hash-embedding cosine fallback
    return _fallback_semantic(text1, text2)


def _multi_scale_boundary_score(
    chunks: List[Dict],
    idx: int,
    doc_type: str,
) -> float:
    """
    Multi-scale boundary score: evaluate the boundary at windows of 1, 2, 3
    chunks on each side.

    Rationale: a single-boundary lexical score can be noisy.  Aggregating
    larger context windows smooths out sentence-level vocabulary variation.
    If the MACRO context (3-chunk window) also shows high similarity, the
    boundary is likely spurious.

    Returns the mean lexical boundary score across the three window sizes.
    """
    windows = [1, 2, 3]
    scores  = []

    for w in windows:
        # Build left context (up to w chunks before idx)
        left  = " ".join(c["text"] for c in chunks[max(0, idx - w):idx]).strip()
        # Build right context (up to w chunks from idx onward)
        right = " ".join(c["text"] for c in chunks[idx:min(len(chunks), idx + w)]).strip()

        if not left or not right:
            continue  # skip if context is empty (document boundary)

        scores.append(_lexical_boundary_score(left, right, doc_type))

    return float(np.mean(scores)) if scores else 0.5


def _cross_encoder_similarity(text1: str, text2: str) -> Optional[float]:
    """
    Cross-encoder similarity using stsb-distilroberta-base.

    A cross-encoder reads the PAIR of texts jointly (unlike bi-encoders that
    embed each text independently), producing a more accurate similarity score.

    Lazy-loaded: the model is downloaded and cached on first use.
    Returns None if the model is unavailable (triggers hash-embedding fallback).
    Normalised from [-1, 1] to [0, 1].
    """
    global _cross_encoder
    try:
        if _cross_encoder is None:
            from sentence_transformers import CrossEncoder  # noqa: E402
            _cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")
        # Truncate inputs to 800 chars to keep inference fast
        score = float(_cross_encoder.predict([(text1[:800], text2[:800])])[0])
        # Normalise: model outputs ∈ [-1, 1] → remap to [0, 1]
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
    except Exception:
        return None   # signal to caller that this option is unavailable


def _fallback_semantic(text1: str, text2: str) -> float:
    """
    Hash-embedding cosine similarity: always-available semantic fallback.
    Uses 128-dim hash vectors (higher dim than S3's drift hash for better precision).
    """
    v1 = _hash_embedding(text1)
    v2 = _hash_embedding(text2)
    return _cosine_similarity(v1.tolist(), v2.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_icc(text: str) -> float:
    """
    Intra-Chunk Coherence (ICC).

    Measures how semantically consistent the sentences WITHIN a chunk are.
    Computed as the mean pairwise Jaccard similarity between consecutive
    sentence token sets:

        ICC = mean_{i} Jaccard(tokens(sᵢ), tokens(sᵢ₊₁))

    High ICC (→1) = the chunk's sentences share a lot of vocabulary → coherent.
    Low ICC (→0)  = the chunk's sentences are topically scattered → incoherent.

    Used by S7's reward function as a chunk quality signal.
    Returns 0.5 (neutral) for single-sentence chunks.
    """
    # Split text into sentences on sentence-ending punctuation followed by space
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if len(sentences) < 2:
        return 0.5   # can't compute pairwise overlap with only one sentence

    overlaps: List[float] = []
    for i in range(len(sentences) - 1):
        a     = set(_tokenize(sentences[i]))
        b     = set(_tokenize(sentences[i + 1]))
        union = a | b
        if union:
            overlaps.append(len(a & b) / len(union))

    return float(np.mean(overlaps)) if overlaps else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Cosine similarity: cos(v1, v2) = (v1 · v2) / (‖v1‖ · ‖v2‖)

    Returns 0.0 if either vector is the zero vector (undefined cosine).
    """
    a  = np.array(v1, dtype=np.float32)
    b  = np.array(v2, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _hash_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    128-dim hash embedding (higher precision than S3's 64-dim drift embedding).
    Normalized to unit L2 norm before returning.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokenize(text):
        vec[hash(tok) % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec   # unit normalisation


def _tokenize(text: str) -> List[str]:
    """
    Extract all word tokens via a word-boundary regex.  Lowercased.
    Shared by all functions in this module.
    """
    return re.findall(r"\b\w+\b", text.lower())