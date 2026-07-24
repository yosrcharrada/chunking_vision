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
  - syntactic function-word patterns (French + English)
  - structural continuity (legal article/paragraph structure for prose)
  - semantic score (embedding cosine similarity or hash fallback)
  - multi-scale boundary score (evaluates windows of 1, 2, 3 chunks)

FIXES IN THIS VERSION
─────────────────────
1. DOUBLE-COUNTING BUG REMOVED
   The original formula computed:
     weighted = α·lexical + β·syntactic_overlap + γ·token_type_match + δ·structural + ε·semantic
   But _lexical_boundary_score internally already called both syntactic_overlap
   AND token_type_match.  This meant syntactic got weight α×0.3 + β = 0.275
   instead of its intended 0.20, and token_type got α×0.3 + γ = 0.225 instead
   of 0.15.  BLEU (the unique part of lexical) was under-represented at only
   α×0.4 = 0.10.

   Fix: the composite score now uses clean, non-overlapping components:
     weighted = α·bleu + β·syntactic + γ·token_type + δ·structural + ε·semantic
   where bleu, syntactic, and token_type are each called ONCE.

2. SYNTACTIC FUNCTION WORDS NOW BILINGUAL (French + English)
   The original _syntactic_overlap used an English-only function word set.
   For French legal documents ("le", "la", "de", "dans", "pour", "que" etc.),
   it found ZERO matching tokens → always returned 0.0 → β contributed nothing.
   Fix: unified French+English function word set so the signal works for both.

3. STRUCTURAL SCORE NOW WORKS FOR LEGAL PROSE
   The original _structural_continuity_score returned a hardcoded 0.5 for all
   prose documents.  French legal and regulatory documents have rich structural
   signals (ARTICLE headers, numbered paragraphs 1), 2), 3), CHAPITRE breaks,
   lettered sub-items a-, b-) that can all indicate strong structural boundaries.
   Fix: legal structural detection for prose that uses these signals properly.

4. ICC SENTENCE SPLITTER IS FRENCH-LEGAL-AWARE
   The original used only [.!?] to split sentences.  French legal text primarily
   structures paragraphs with numbered items (1), 2), 3)) and line breaks, not
   terminal punctuation.  The old splitter often produced 1–3 giant "sentences"
   per chunk → ICC ≈ 0.5 (neutral default) rather than a real measurement.
   Fix: extended splitter that recognises French legal paragraph markers.

Score formula (corrected, no double-counting)
──────────────────────────────────────────────
    weighted = α·bleu + β·syntactic + γ·token_type + δ·structural + ε·semantic
where:
    α = 0.25  (n-gram BLEU bigram precision — unique lexical overlap)
    β = 0.20  (syntactic function-word overlap — French + English)
    γ = 0.15  (token type Jaccard — unique vocabulary overlap)
    δ = 0.20  (structural continuity — legal-aware for prose)
    ε = 0.20  (average of hash-embedding cosine and multi-scale lexical)

High score → the two chunks are very similar → merge candidate.
Low score  → the boundary is valid → keep the split.

Intra-Chunk Coherence (ICC)
────────────────────────────
Every chunk also receives an "icc" field:
    ICC(chunk) = mean Jaccard(sᵢ, sᵢ₊₁) over consecutive unit pairs
    where units are French-legal-aware paragraph splits.
High ICC → the sentences WITHIN the chunk are topically coherent.
ICC is used by S7 as a quality signal in the reward function.
"""

import math
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

    # S4 semantic-similarity mode: "cosine" (classical, the old way) or
    # "qcosine" (the Fitouhi–Bouzeffour q-cosine of the inter-chunk angle, base = q²).
    # sim_mode ALWAYS controls which formula computes the `semantic` value below
    # (used both as one ingredient of the composite score AND, in kernel-gate
    # mode, as the merge decision itself).
    #
    # s4_gate_mode controls the DECISION RULE, independent of sim_mode:
    #   "kernel"    (default when s4_similarity is present) — merge iff the
    #               (q-)cosine similarity alone exceeds τ_sem.  The kernel
    #               directly drives the merge; the other 5 signals are still
    #               computed and reported for diagnostics but do not vote.
    #   "composite" — merge iff the ORIGINAL 6-signal weighted blend exceeds
    #               τ_sem, exactly as the pipeline always worked, with the ONE
    #               ingredient swapped: cosine or q-cosine per sim_mode.  This
    #               isolates "does swapping the semantic ingredient inside the
    #               real, deployed S4 formula help" — a more faithful ablation
    #               than the kernel-direct comparison, since nothing about the
    #               decision RULE changes, only that one ingredient.
    # When s4_similarity is absent entirely: legacy behaviour, byte-for-byte
    # unchanged (composite gate, semantic=cosine) — every earlier experiment /
    # table that never set the flag is reproduced exactly.
    _explicit_gate_mode = config.get("s4_gate_mode")
    if _explicit_gate_mode is not None:
        kernel_gate = str(_explicit_gate_mode).lower() == "kernel"
    else:
        kernel_gate = "s4_similarity" in config
    sim_mode = str(config.get("s4_similarity", "cosine")).lower()
    if sim_mode not in {"cosine", "qcosine"}:
        sim_mode = "cosine"
    # q_s4_kernel is the DECOUPLED gene the S7 GA tunes specifically for this
    # kernel (independent of S3's q_entropy_param, which drives D_q).  Any
    # caller that predates this gene (all the earlier S4-only ablations,
    # bench_s4_similarity.py, run_benchmark.py's kernel-swap block, the
    # already-published qcosine_sim_tables.tex, etc.) never sets it, so they
    # fall back to q_entropy_param exactly as before -- byte-for-byte
    # unchanged behaviour.
    q_param = float(config.get("q_s4_kernel", config.get("q_entropy_param", 1.0)))

    # The Fitouhi--Bouzeffour series (eqs. 5-6) is only rigorously defined for
    # base=q^2 in (0,1); base=1 is a removable singularity, justified in the
    # source paper ONLY as the q->1^- classical limit (cos(x;q^2) -> cos(x)).
    # Because our convention squares q, q=-1 maps to that SAME base=1 point --
    # a redundant duplicate of the q=+1 anchor that the paper never considers
    # (it only treats q>0).  Per instructor guidance, q=-1 is excluded for the
    # q-cosine kernel: floor q away from exactly -1 so the formula never
    # evaluates that point, while q=+1 remains the valid classical-limit
    # anchor.  This floor affects ONLY the qcosine kernel below -- S3's D_q
    # (Tsallis entropy) is untouched and still spans the full q in [-1, 1].
    _QCOS_MIN_Q = -0.999
    if sim_mode == "qcosine" and q_param <= _QCOS_MIN_Q:
        q_param = _QCOS_MIN_Q

    # Entropy-gated q-cosine (experimental, OFF by default): couple the kernel's
    # deformation to S3's per-boundary strength (chunk["boundary_signal"]).  A
    # global q + hard threshold is just cosine at a shifted τ; making the base
    # boundary-local is the only way the q-cosine can express a merge rule cosine
    # cannot.  In practice this was found NEUTRAL-to-slightly-worse on the corpus
    # (boundary_signal and the adjacent cosine are correlated, so the coupling is
    # largely redundant), hence it defaults off.  Reduces to the plain global
    # q-cosine when disabled or when every boundary signal is equal.
    couple_local = bool(config.get("s4_qcos_couple", False))
    _sigs = [float(c.get("boundary_signal", 0.0)) for c in chunks]
    _max_sig = max(_sigs) if any(s > 1e-9 for s in _sigs) else 1.0

    # q-cosine VALLEY mechanism (experimental): instead of a pairwise hard gate,
    # score each boundary relative to its neighbours so the kernel detects genuine
    # similarity valleys (topic breaks) rather than absolute similarity.  This is
    # non-monotone in a single cosine, and the neighbour averaging of the
    # nonlinear q-cosine carries a q-dependent Jensen gap, so q matters beyond a
    # threshold shift.  λ=0 ⇒ the plain pairwise gate.  Precompute the static
    # consecutive q-cosine sequence over the INPUT chunks (independent of merges).
    window_mech = bool(config.get("s4_qcos_window", False)) and sim_mode == "qcosine"
    qcos_lambda = float(config.get("s4_qcos_lambda", 0.5))
    qcos_seq = None
    if window_mech:
        qcos_seq = [0.0] * len(chunks)
        for j in range(1, len(chunks)):
            raw_j = _semantic_score(chunks[j - 1]["text"], chunks[j]["text"], embeddings, j)
            qcos_seq[j] = _q_cosine_similarity(raw_j, q_param)

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
        # Each component is computed ONCE and contributes to the weighted sum
        # exactly once.  The original code called _syntactic_overlap and
        # _token_type_match inside _lexical_boundary_score AND again explicitly
        # in the weighted sum, double-counting their contribution.

        # BLEU bigram precision — unique to lexical component
        bleu       = _ngram_precision(prev["text"], curr["text"], n=2)

        # Syntactic function-word overlap — French + English (fixed from EN-only)
        syntactic  = _syntactic_overlap(prev["text"], curr["text"], doc_type)

        # Token-type Jaccard — unique vocabulary overlap
        token_type = _token_type_match(prev["text"], curr["text"])

        # Structural: legal-aware for prose, brace/markup for code/mixed
        structural = _structural_continuity_score(prev["text"], curr["text"], doc_type)

        # Semantic: embedding cosine if available, else hash-embedding cosine
        semantic_cos = _semantic_score(prev["text"], curr["text"], embeddings, idx)

        # q-analog of cosine similarity (instructor's q² convention).  In
        # "qcosine" mode the raw cosine is replaced by the Fitouhi–Bouzeffour q-cosine of the
        # inter-chunk angle; in "cosine" mode the classical value is kept verbatim.
        if sim_mode == "qcosine":
            if couple_local:
                # base_i interpolates between q² (strong S3 boundary → conservative)
                # and 1 (weak boundary → plain cosine).  curr's boundary_signal is
                # the shift that opened the boundary between prev and curr.
                s_i = min(1.0, max(0.0, float(curr.get("boundary_signal", 0.0)) / _max_sig))
                base_i = 1.0 - (1.0 - q_param * q_param) * s_i
                semantic = _q_cos_sim_base(semantic_cos, base_i)
            else:
                semantic = _q_cosine_similarity(semantic_cos, q_param)
        else:
            semantic = semantic_cos

        # Multi-scale: lexical score at windows 1, 2, 3 chunks wide
        multiscale = _multi_scale_boundary_score(chunks, idx, doc_type)

        # ── Weighted composite score (no double-counting) ─────────────────
        # α=0.25 bleu  β=0.20 syntactic  γ=0.15 token_type
        # δ=0.20 structural  ε=0.20 (semantic+multiscale)/2
        weighted = (
            _ALPHA   * bleu
            + _BETA  * syntactic
            + _GAMMA * token_type
            + _DELTA * structural
            + _EPSILON * ((semantic + multiscale) / 2.0)
        )

        # Apply the global merge_weight scale and clamp to [0, 1]
        decision_score = float(np.clip(weighted * merge_weight, 0.0, 1.0))

        # ── Annotate current chunk ────────────────────────────────────────
        curr["boundary_score"] = round(decision_score, 4)
        curr["icc"]            = _compute_icc(curr["text"])
        curr["boundary_breakdown"] = {
            "bleu":        round(float(bleu),          4),
            "syntactic":   round(float(syntactic),     4),
            "token_type":  round(float(token_type),    4),
            "structural":  round(float(structural),    4),
            "semantic":    round(float(semantic),      4),
            "semantic_cos": round(float(semantic_cos), 4),
            "multiscale":  round(float(multiscale),    4),
            "weighted":    round(float(decision_score), 4),
            "sim_mode":    sim_mode,
        }

        # ── Merge decision ────────────────────────────────────────────────
        # Kernel gate (s4_similarity set): merge when the (q-)cosine similarity
        # of the two chunks exceeds τ_sem — this is the criterion the kernel
        # controls.  Legacy gate (flag absent): the composite lexical+structural
        # score vs τ_sem, exactly as before.  Both require the size guard.
        prev_wc = len(prev["text"].split())
        curr_wc = len(curr["text"].split())
        gate_value = semantic if kernel_gate else decision_score
        if window_mech and qcos_seq is not None:
            # Static neighbour-relative valley score over the input adjacency.
            neigh = [qcos_seq[k] for k in (idx - 1, idx + 1) if 0 < k < len(chunks)]
            nmean = (sum(neigh) / len(neigh)) if neigh else qcos_seq[idx]
            gate_value = (1.0 + qcos_lambda) * qcos_seq[idx] - qcos_lambda * nmean

        if gate_value > tau_sem and (prev_wc + curr_wc) <= n_max * 1.5:
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
    For PROSE: overlaps a unified French + English function word set.

    WHY FRENCH WAS ADDED
    ─────────────────────
    The original used an English-only function word set.  For French documents
    ("le", "la", "de", "dans", "pour", "qui", "que" etc.), it found ZERO
    matching tokens on every comparison → always returned 0.0 → the β=0.20
    syntactic weight contributed nothing to the boundary score for any French
    document.  Adding French function words fixes this.

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

    # Unified French + English function words.
    # French additions cover the most frequent grammatical words in legal text.
    func = {
        # English
        "the", "a", "an", "is", "was", "are", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "of", "in",
        "on", "at", "by", "for", "with", "about", "as", "to",
        # French — high-frequency function words in legal/regulatory prose
        "le", "la", "les", "de", "des", "du", "et", "en", "un", "une",
        "dans", "pour", "que", "qui", "est", "sont", "sur", "par",
        "avec", "au", "aux", "ce", "se", "si", "ne", "pas", "ou",
        "dont", "leur", "leurs", "cette", "son", "ses", "tout", "tous",
        "toute", "toutes", "plus", "bien", "même", "ainsi", "comme",
        "selon", "sous", "entre", "après", "avant", "sans", "lors",
    }

    # Extract function tokens from each text (Unicode-aware tokenization)
    t1 = [w for w in re.findall(r"\b[\wÀ-ÿ]+\b", text1.lower()) if w in func]
    t2 = [w for w in re.findall(r"\b[\wÀ-ÿ]+\b", text2.lower()) if w in func]

    # Clipped count (same logic as BLEU): each function word in text2 can
    # match at most as many times as it appears there
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

    For CODE/TABLE/MIXED: uses brace/markup balance signals (unchanged).
    For PROSE: now detects French and English legal/regulatory structure
               instead of returning a hardcoded neutral 0.5.

    WHY THE PROSE CASE CHANGED
    ───────────────────────────
    The original returned 0.5 unconditionally for all prose documents.
    This meant the δ=0.20 structural weight contributed a fixed 0.10 bias
    to every boundary score regardless of actual structure.

    French legal and regulatory documents have rich structural signals:
      • ARTICLE / CHAPITRE / TITRE headers → strong new-block boundary
      • Numbered paragraphs (1), 2), 3)) → paragraph-level boundaries
      • Lettered sub-items (a-, b-, a), b)) → sub-paragraph boundaries
      • ALL-CAPS section titles → strong boundary

    Scoring logic for prose
    ───────────────────────
    We compute two sub-scores:

    (a) boundary_strength: how strongly does text2 START a new legal unit?
        - New ARTICLE/CHAPITRE/TITRE at start of text2 → 0.0  (strong boundary)
        - New numbered paragraph at start → 0.15
        - New all-caps heading → 0.05
        - No detected structure → 0.5  (neutral)
        Low score = strong boundary = low continuity = texts should NOT be merged.

    (b) internal_consistency: do text1 and text2 share the same structural
        pattern (e.g. both use numbered paragraphs → continuation of same article)?
        - Both use same structure pattern → 0.7 (moderate continuity)
        - Patterns differ → 0.3

    Final = 0.6 × boundary_strength + 0.4 × internal_consistency
    """
    if doc_type in {"code", "mixed", "table"}:
        # ── Original code/mixed/table logic (unchanged) ──────────────────
        brace_delta     = (abs(text1.count("{") - text1.count("}"))
                           + abs(text2.count("{") - text2.count("}")))
        markdown_bridge = 1.0 if re.search(r"^#{1,6}\s", text2, re.MULTILINE) else 0.6
        xml_bridge      = (0.8 if ("<" in text1 and ">" in text1
                                   and "<" in text2 and ">" in text2) else 0.4)
        cont = 1.0 - min(1.0, brace_delta / 6.0)
        return float(np.clip((cont + markdown_bridge + xml_bridge) / 3.0, 0.0, 1.0))

    # ── Prose: legal/regulatory structure detection ───────────────────────

    # Patterns that mark the START of a new major legal unit in text2.
    # Each returns a LOW score because a new unit = strong boundary = low continuity.
    _ARTICLE_RE  = re.compile(
        r"(?m)^\s*(?:ARTICLE|Article|Art\.?|CHAPITRE|TITRE|SECTION|Chapitre|Titre)\s+\w+",
        re.IGNORECASE,
    )
    _PARA_NUM_RE  = re.compile(r"(?m)^\s*\d+\)\s")      # 1)  2)  3)
    _PARA_LET_RE  = re.compile(r"(?m)^\s*[a-z][-\)]\s") # a-  b-  a)  b)
    _ALLCAPS_RE   = re.compile(r"(?m)^[A-ZÉÀÈÙÂÊÎÔÛŒÆ][A-ZÉÀÈÙÂÊÎÔÛŒÆ\s]{4,}$")

    # --- (a) boundary_strength: does text2 open a new legal unit? ---
    first_200 = text2[:200]  # only check the opening of text2

    if _ARTICLE_RE.search(first_200):
        # New article/chapter/title header → very strong boundary
        boundary_strength = 0.05
    elif _ALLCAPS_RE.search(first_200):
        # All-caps heading → strong boundary
        boundary_strength = 0.10
    elif _PARA_NUM_RE.search(first_200):
        # New numbered paragraph → moderate boundary
        boundary_strength = 0.25
    else:
        # No detected legal structure → neutral
        boundary_strength = 0.50

    # --- (b) internal_consistency: do both texts share the same structure? ---
    # If both chunks use numbered paragraphs, they are likely continuation of
    # the same article → higher continuity.  If they differ → lower.
    t1_has_article = bool(_ARTICLE_RE.search(text1))
    t2_has_article = bool(_ARTICLE_RE.search(text2))
    t1_has_para    = bool(_PARA_NUM_RE.search(text1))
    t2_has_para    = bool(_PARA_NUM_RE.search(text2))

    if t1_has_article and t2_has_article:
        # Both contain article headers — likely different articles → low continuity
        internal_consistency = 0.20
    elif t1_has_para and t2_has_para and not t2_has_article:
        # Both use numbered paragraphs and text2 doesn't open a new article
        # → continuation of the same article → moderate-high continuity
        internal_consistency = 0.65
    elif not t1_has_article and not t2_has_article:
        # Neither has article headers → plain prose continuation
        internal_consistency = 0.55
    else:
        # Asymmetric structure → boundary is likely real
        internal_consistency = 0.30

    score = 0.60 * boundary_strength + 0.40 * internal_consistency
    return float(np.clip(score, 0.0, 1.0))


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

    Measures how semantically consistent the natural units WITHIN a chunk are.
    Computed as the mean pairwise Jaccard similarity between consecutive units:

        ICC = mean_{i} Jaccard(tokens(uᵢ), tokens(uᵢ₊₁))

    where uᵢ are the natural units identified by _split_units().

    High ICC (→1) = units share a lot of vocabulary → coherent chunk.
    Low ICC (→0)  = units are topically scattered → incoherent chunk.

    WHY THE SPLITTER CHANGED
    ─────────────────────────
    The original split only on [.!?] — standard English sentence endings.
    French legal text primarily structures paragraphs with:
      • Numbered items: 1)  2)  3)  (most common in conventions/treaties)
      • Lettered items: a-  b-  a)  b)
      • Line breaks between paragraphs
      • Terminal periods (shared with English)

    Using only [.!?] produced 1–3 giant "sentences" for 900-word chunks
    (because legal sentences are very long), giving ICC ≈ 0.5 by default.
    The new splitter recognises French legal paragraph structure and produces
    meaningful units that reflect the actual internal organisation of the chunk.

    Returns 0.5 (neutral) if fewer than 2 units are found.
    """
    units = _split_units(text)
    if len(units) < 2:
        return 0.5

    overlaps: List[float] = []
    for i in range(len(units) - 1):
        # Unicode-aware tokenization to handle French accented characters
        a     = set(re.findall(r"\b[\wÀ-ÿ]+\b", units[i].lower()))
        b     = set(re.findall(r"\b[\wÀ-ÿ]+\b", units[i + 1].lower()))
        union = a | b
        if union:
            overlaps.append(len(a & b) / len(union))

    return float(np.mean(overlaps)) if overlaps else 0.5


def _split_units(text: str) -> List[str]:
    """
    Split text into natural units for ICC computation.

    French-legal-aware splitting strategy (priority order):
    1. Numbered paragraphs: lines starting with N) pattern (most common in treaties)
    2. Lettered sub-items: lines starting with a- or a) pattern
    3. Blank-line paragraph breaks
    4. Standard sentence-ending punctuation [.!?] followed by whitespace

    All units shorter than 4 words are discarded as noise (page numbers,
    short labels, etc.).

    This is intentionally NOT the same as the S3 sentence splitter — it is
    tuned for ICC measurement (structural paragraph units) rather than for
    entropy-rate computation (linguistic sentences).
    """
    # Step 1: try numbered paragraphs (N) at start of line)
    # This is the dominant structure in French legal conventions
    para_split = re.split(r"\n+(?=\s*\d+\)\s)", text)
    if len(para_split) >= 2:
        units = [u.strip() for u in para_split if u.strip()]
        units = [u for u in units if len(u.split()) >= 4]
        if len(units) >= 2:
            return units

    # Step 2: try blank-line paragraph breaks
    para_split2 = re.split(r"\n{2,}", text)
    if len(para_split2) >= 2:
        units = [u.strip() for u in para_split2 if u.strip()]
        units = [u for u in units if len(u.split()) >= 4]
        if len(units) >= 2:
            return units

    # Step 3: fall back to sentence-ending punctuation
    sent_split = re.split(r"(?<=[.!?])\s+", text)
    units = [u.strip() for u in sent_split if u.strip() and len(u.split()) >= 4]
    return units if units else [text.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# q-analog of cosine similarity  —  the Fitouhi–Bouzeffour q-cosine
# ─────────────────────────────────────────────────────────────────────────────
#
# Reference: A. Fitouhi & F. Bouzeffour, "The q-cosine Fourier transform and the
# q-heat equation", Ramanujan J. 28 (2012) 443–461, eqs. (5)–(6).  (This is the
# Koornwinder–Swarttouw q-cosine adopted there; it is NOT Jackson's cos_q / sin_q,
# which the paper explicitly distinguishes.)  Classical cosine similarity of two
# embeddings is cos(θ), the cosine of the angle θ between them.  The q-deformed
# analog replaces the ordinary cosine by the Fitouhi–Bouzeffour q-cosine
#
#     cos(x; q²) = ₁φ₁(0; q; q², (1-q)² x²) = Σ_{n≥0} (-1)ⁿ bₙ(x; q²),      (eq. 5)
#     bₙ(x; q²) = bₙ(1; q²) x^{2n} = q^{n(n-1)} (1-q)^{2n} / (q;q)_{2n} · x^{2n}, (eq. 6)
#
# where (q;q)_m = Π_{k=1}^m (1-q^k) is the q-Pochhammer symbol and 0 < q < 1.
# As q→1 the coefficient q^{n(n-1)}(1-q)^{2n}/(q;q)_{2n} → 1/(2n)! and
# cos(x; q²) → cos(x), so the classical cosine is recovered continuously (this
# matches the paper's bound |bₙ(1;q²)| ≤ 1/(2n)!, Prop. 3.1).
#
# Instructor's convention (the "q is squared" point): the fundamental base in the
# coefficient (eq. 6) is taken as q².  The pipeline's Tsallis parameter lives in
# [-1,1] and may be negative — outside the paper's domain 0<q<1.  Squaring it maps
# [-1,1] → [0,1], so the series is well defined for every pipeline q; we therefore
# evaluate eq. (6) with base = q².

def _q_pochhammer(base: float, m: int) -> float:
    """q-Pochhammer (base; base)_m = Π_{k=1}^m (1 - base^k).  (m=0 → 1.)"""
    prod = 1.0
    p = base  # base^1
    for _ in range(m):
        prod *= (1.0 - p)
        p *= base
    return prod


def _q_cosine(x: float, base: float, terms: int = 24) -> float:
    """
    Fitouhi–Bouzeffour q-cosine cos(x; q²) evaluated from its power series (see header).
    ``base`` is q² ∈ [0,1].  The series is entire in x and converges fast
    because q^{n(n-1)} decays super-geometrically.
    """
    # base → 1 is the classical limit; the series is 0/0 there, so use cos(x).
    if base >= 1.0 - 1e-9:
        return math.cos(x)
    # base → 0: only n=0,1 survive (q^{n(n-1)}=0 for n≥2) → cos(x; 0) = 1 - x².
    if base <= 1e-12:
        return 1.0 - x * x
    total = 0.0
    for n in range(terms):
        poch = _q_pochhammer(base, 2 * n)      # (base;base)_{2n}
        if poch == 0.0:
            break
        bn = (base ** (n * (n - 1))) * ((1.0 - base) ** (2 * n)) / poch
        term = ((-1.0) ** n) * bn * (x ** (2 * n))
        total += term
        if n > 2 and abs(term) < 1e-15:
            break
    return total


def _q_cosine_similarity(cos_val: float, q: float, terms: int = 24) -> float:
    """
    q-analog of cosine similarity: the Fitouhi–Bouzeffour q-cosine evaluated at the
    inter-chunk angle, used in place of the ordinary cosine.

    cos_val : ordinary cosine similarity of the two embeddings, in [-1,1].
    q       : pipeline Tsallis parameter; deformation base = q² ∈ [0,1].

    Returns clip( cos(θ; q²), -1, 1 ) with θ = arccos(cos_val).  Because
    cos(θ; q²) → cos(θ) as the base → 1, at |q|=1 this is EXACTLY the ordinary
    cosine, so the q-cosine kernel reduces to the classical one at the spectrum
    endpoints.  For |q|<1 the kernel is more conservative — a higher ordinary
    cosine is required to reach the same q-similarity (cos(θ;0)=1-θ²) — so
    q-cosine merging keeps finer boundaries.  It is monotonically decreasing in
    the angle for every q, so the "high = similar = merge" ordering is preserved.

    q=-1 is floored to -0.999: base=q^2 would otherwise coincide exactly with
    q=+1's classical-limit anchor, a redundant point the source paper never
    considers (it only treats q>0) -- excluded per instructor guidance. This
    floor is defence-in-depth for callers that reach this function directly
    (e.g. S3's optional q-cosine gap re-ranking); filter_boundaries applies the
    same floor to config["q_entropy_param"] before it ever gets here.
    """
    qf = max(float(q), -0.999)
    return _q_cos_sim_base(cos_val, qf * qf, terms)


def _q_cos_sim_base(cos_val: float, base: float, terms: int = 24) -> float:
    """Core q-cosine similarity for an explicit base ∈ [0,1] (see header).
    base=1 ⇒ ordinary cosine; smaller base ⇒ more conservative (resists merging)."""
    c = max(-1.0, min(1.0, float(cos_val)))
    theta = math.acos(c)
    qc = _q_cosine(theta, max(0.0, min(1.0, float(base))), terms)
    return float(max(-1.0, min(1.0, qc)))


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
    Unicode-aware: handles French accented characters (é, è, à, ù, â, etc.)
    Shared by all functions in this module.
    """
    return re.findall(r"\b[\wÀ-ÿ]+\b", text.lower())