"""
S3 — Entropy Boundary Refinement  (v3 — financial/regulatory edition)
======================================================================
Pipeline position : runs AFTER S2 (chunkers) and BEFORE S4 (boundary quality).
Responsibility    : evaluate every adjacent chunk pair and decide whether to
                    MERGE them (too similar), mark them as HARD (confirmed topic
                    shift), or leave them as SOFT (ambiguous, passed to S4).

Five complementary boundary signals
────────────────────────────────────
1. JSD          Jensen-Shannon Divergence on unigram distributions.
                Symmetric KL-divergence: JSD(P,Q) = ½KL(P‖M) + ½KL(Q‖M)
                where M = (P+Q)/2.  Range [0,1].

2. Hellinger    Hellinger distance between unigram distributions.
                H(P,Q) = ‖√P − √Q‖ / √2.  More sensitive than JSD to
                changes in rare (low-probability) terms.  Range [0,1].

3. PMI-drop     Key concept shift: measures divergence of the TOP-8 content
                terms between adjacent chunks.  High when the dominant concepts
                change (strong boundary), low when they stay the same (merge).

4. Depth-change Structural hierarchy depth at the START of chunk B.
                Article N → score ≈ 0.67; TITRE → 1.0; bullet → ≈ 0.17.
                This is a structural/regex signal, NOT a distributional one.

5. Drift        Cosine distance between lightweight hash embeddings of the two
                chunks.  Detects gradual semantic migration that the LSTM cell
                accumulates across many boundaries.

LSTM memory
───────────
The LSTM receives the 5-dim feature vector at every boundary position in
document order.  Its cell state cₜ accumulates context so that a boundary
at position 15 is judged relative to the entropy history of positions 0–14.
Weights are fixed (Xavier, seed=42): fully deterministic, no training needed.

Decision rules (in priority order)
────────────────────────────────────
  1. protected boundary → HARD  (Article N, TITRE, ALL-CAPS heading, …)
  2. combined_signal < τ_low AND size fits → MERGE
  3. combined_signal > τ_high              → HARD
  4. otherwise                             → SOFT  (passed to S4)

Threshold modes
───────────────
  fixed      : τ_low / τ_high taken directly from the config sliders.
  percentile : τ_low = P(pct_low) and τ_high = P(pct_high) computed from
               THIS document's own signal distribution.  Recommended for
               financial/regulatory text whose entropy range is document-specific.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Stopword set  (English + French)
# ─────────────────────────────────────────────────────────────────────────────
# Used by _content_tokens() to strip function words before computing PMI-drop
# and the drift hash embedding.  Without this, words like "le", "de", "the"
# would dominate the TOP-8 content terms and mask real concept boundaries.

_STOPWORDS = {
    # English function words
    "the","a","an","and","or","but","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","must","shall","can","to","of","in","for","on",
    "with","at","by","from","as","into","through","during","before","after",
    "here","there","when","where","why","how","all","both","each","few",
    "more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","just","this","that","these","those",
    "it","its","he","she","they","we","you","i","my","your","his","her",
    "our","their","what","which","who","whom","about",
    # French function words — essential for Tunisian financial/legal documents
    "le","la","les","de","des","du","et","en","un","une","dans","pour",
    "que","est","sur","par","avec","au","aux","ce","se","si","ne","pas",
    "lui","leur","ils","elles","nous","vous","on","dont","où","car","mais",
    "ni","or","donc","comment","quand","comme","tout","tous","toute",
}


# ─────────────────────────────────────────────────────────────────────────────
# Protected boundary regex
# ─────────────────────────────────────────────────────────────────────────────
# Any chunk whose TEXT begins with a line matching this pattern is ALWAYS a
# hard split, regardless of any entropy or LSTM score.
# Covers:
#   - French structural hierarchy: TITRE I, CHAPITRE 2, SECTION 3, …
#   - Article references: Article 1, Art. 22bis, ARTICLE premier, …
#   - Paragraph symbol: § 3
#   - Markdown headings: # Title, ## Section, …
#   - ALL-CAPS headings ≥5 chars (common in Tunisian regs):
#       "DISPOSITIONS GÉNÉRALES", "CHAMP D'APPLICATION:", …

_PROTECTED_RE = re.compile(
    r"^[ \t]*(?:"
    # French/EN structural keywords followed by a number or ordinal
    r"(?:TITRE|CHAPITRE|SECTION|SOUS-SECTION|PARAGRAPHE|BOOK|PART|CHAPTER)"
    r"\s+(?:[IVXLCDM]+|\d+|PREMIER|PREMIERE|PREMIÈRE|premier|premiere|première)"
    # Article references (multiple French/EN formats)
    r"|(?:Article|Art\.?|ARTICLE)\s+(?:\d+(?:\s*(?:er|eme|ème|e|bis|ter|quater))?|premier|1er|[IVX]+)"
    # Paragraph symbol
    r"|§\s*\d+"
    # Markdown heading (one to six #)
    r"|#{1,6}\s+\S+"
    # ALL-CAPS heading ≥5 chars, optionally ending with ":" or at end of line
    r"|[A-Z][A-Z\s\-]{4,}(?::|$)"
    r")(?:[ \t].*)?$",
    re.MULTILINE | re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Structural depth markers  (Signal 4)
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is (compiled_regex, depth_level).
# depth=1 → strongest boundary (TITRE, CHAPITRE).
# depth=6 → weakest (bullet point).
# Normalization: score = 1 − (depth−1)/6
#   depth 1 → 1.00,  depth 2 → 0.83,  depth 3 → 0.67 (Article),
#   depth 4 → 0.50,  depth 5 → 0.33,  depth 6 → 0.17

_DEPTH_MARKERS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"^\s*(?:TITRE|CHAPITRE|BOOK|PART)\b",             re.I), 1),
    (re.compile(r"^\s*(?:SECTION|SOUS-SECTION)\b",                 re.I), 2),
    (re.compile(r"^\s*(?:Article|Art\.?|ARTICLE|§)\s*\d+",         re.I), 3),
    (re.compile(r"^\s*(?:\d+(?:\.\d+)+)\s+\S",                     re.I), 4),  # 3.2.1 style
    (re.compile(r"^\s*[a-zA-Z]\.\s+\S",                            re.I), 5),  # a. b. c.
    (re.compile(r"^\s*[-–•]\s+\S",                                 re.I), 6),  # bullet
]


# ─────────────────────────────────────────────────────────────────────────────
# Forward LSTM cell  — deterministic, fixed-weight
# ─────────────────────────────────────────────────────────────────────────────

class _ForwardLSTMCell:
    """
    Single-layer forward LSTM.

    Why fixed weights?
    ──────────────────
    We need fully deterministic output (same document → same chunks every run)
    without requiring a training dataset.  Fixed Xavier weights (seed=42) give
    the LSTM its gating structure — forget/input/output gates — so it can
    "remember" entropy history without being trained.

    Architecture
    ────────────
    Input  : 5-dim vector [jsd, hellinger, pmi_drop, depth_change, drift]
    Hidden : 12-dim cell state cₜ and hidden state hₜ
    Output : scalar ∈ [0,1] via  score = σ(Wₚ · hₜ)

    Gate equations (standard LSTM, Hochreiter & Schmidhuber 1997):
      iₜ = σ(Wᵢxₜ + Uᵢhₜ₋₁ + bᵢ)    ← input gate
      fₜ = σ(Wfxₜ + Ufhₜ₋₁ + bf)    ← forget gate  (bias init = 1)
      gₜ = tanh(Wgxₜ + Ughₜ₋₁ + bg) ← candidate cell
      oₜ = σ(Woxₜ + Uohₜ₋₁ + bo)    ← output gate
      cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ     ← cell state update
      hₜ = oₜ ⊙ tanh(cₜ)             ← hidden state
    """

    INPUT_DIM  = 5   # one per entropy signal
    HIDDEN_DIM = 12  # dimensionality of cell and hidden states

    def __init__(self, seed: int = 42):
        rng   = np.random.RandomState(seed)   # fixed seed → reproducible weights
        idim  = self.INPUT_DIM
        hdim  = self.HIDDEN_DIM

        # Xavier scale: sqrt(2 / (fan_in + fan_out)) keeps activations healthy
        scale = np.sqrt(2.0 / (idim + hdim))

        # ── Input gate  iₜ = σ(Wᵢxₜ + Uᵢhₜ₋₁ + bᵢ) ────────────────────
        self.Wi = rng.randn(hdim, idim).astype(np.float32) * scale  # input projection
        self.Ui = rng.randn(hdim, hdim).astype(np.float32) * scale  # recurrent projection
        self.bi = np.zeros(hdim, dtype=np.float32)                  # bias (init 0)

        # ── Forget gate  fₜ = σ(Wfxₜ + Ufhₜ₋₁ + bf) ────────────────────
        # Forget gate bias initialised to 1 (Jozefowicz et al. 2015):
        # at t=0, fₜ ≈ 1 so the cell "remembers everything" by default.
        self.Wf = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Uf = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bf = np.ones(hdim, dtype=np.float32)                   # bias (init 1)

        # ── Cell gate  gₜ = tanh(Wgxₜ + Ughₜ₋₁ + bg) ───────────────────
        self.Wg = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Ug = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bg = np.zeros(hdim, dtype=np.float32)

        # ── Output gate  oₜ = σ(Woxₜ + Uohₜ₋₁ + bo) ────────────────────
        self.Wo = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Uo = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bo = np.zeros(hdim, dtype=np.float32)

        # ── Scalar projection: hdim → 1 scalar score ─────────────────────
        # Scale by sqrt(1/hdim) to keep pre-sigmoid output near 0 initially
        self.Wp = rng.randn(1, hdim).astype(np.float32) * np.sqrt(1.0 / hdim)

        # ── State vectors (reset at start of every document) ─────────────
        self.h = np.zeros(hdim, dtype=np.float32)   # hidden state hₜ
        self.c = np.zeros(hdim, dtype=np.float32)   # cell state   cₜ

    def reset(self) -> None:
        """
        Reset h and c to zero.
        MUST be called before processing each new document — otherwise context
        from a previous document leaks into the current one.
        """
        self.h[:] = 0.0
        self.c[:] = 0.0

    def step(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Execute one LSTM step for one boundary position.

        Parameters
        ──────────
        x : np.ndarray, shape (5,)
            Boundary feature vector [jsd, hellinger, pmi_drop,
            depth_change, drift].  All values ∈ [0,1].

        Returns
        ───────
        score : float ∈ [0,1]
            Boundary confidence.  High → LSTM context says this is a real
            boundary.  Low → context says the chunks are topically continuous.
        cell  : np.ndarray, shape (12,)
            Copy of cₜ after this step.  Exposed in output JSON as "lstm_cell"
            so the Pipeline Inspector can visualise the memory state.
        """
        # Compute gate activations using the current input x and prev hidden h
        i_g = self._sig(self.Wi @ x + self.Ui @ self.h + self.bi)   # input gate
        f_g = self._sig(self.Wf @ x + self.Uf @ self.h + self.bf)   # forget gate
        g_g = np.tanh(  self.Wg @ x + self.Ug @ self.h + self.bg)   # candidate
        o_g = self._sig(self.Wo @ x + self.Uo @ self.h + self.bo)   # output gate

        # Update cell state: forget some old info, add new candidate info
        self.c = f_g * self.c + i_g * g_g

        # Update hidden state: filter cell through output gate
        self.h = o_g * np.tanh(self.c)

        # Project hidden state to scalar boundary confidence score
        score = float(self._sig(self.Wp @ self.h)[0])

        # Return score and a COPY of the cell state (copy prevents aliasing bugs)
        return score, self.c.copy()

    @staticmethod
    def _sig(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid. Clips to [-20, 20] to prevent overflow."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Raw signal selector
# ─────────────────────────────────────────────────────────────────────────────

def _raw_signal_from_features(feat: Dict[str, float], metric: str) -> float:
    """
    Compute the raw boundary score from the 5-dim feature dict.

    This is the bridge between the frontend "Entropy metric" dropdown and the
    actual computation.  The mode determines which signal(s) contribute to the
    raw score BEFORE the LSTM modulates it.

    NOTE: The LSTM always receives all 5 signals regardless of this setting.
    Changing the mode only affects the raw weighted sum — the LSTM's temporal
    context is always computed from the full 5-dim input.

    Parameters
    ──────────
    feat   : dict with keys jsd, hellinger, pmi_drop, depth_change, drift
    metric : string — one of "jsd", "hellinger", "pmi", "depth", "drift",
             or "hybrid" (default)

    Returns
    ───────
    float ∈ [0,1] : the raw boundary score before LSTM modulation
    """
    if metric == "jsd":
        # Pure JSD — classic baseline; good when topic shifts cause clear vocab changes
        return float(feat["jsd"])

    if metric == "hellinger":
        # Hellinger only — better than JSD for detecting rare-term shifts
        # (e.g. a new legal concept introduced with low frequency)
        return float(feat["hellinger"])

    if metric == "pmi":
        # PMI-drop only — best for financial/regulatory text where every article
        # shares the same surface vocabulary ("impôt", "Convention") but each
        # article's KEY concepts differ ("domicile fiscal" vs "établissement stable")
        return float(feat["pmi_drop"])

    if metric == "depth":
        # Structural depth change only — for heavily hierarchical documents where
        # Article N is the primary and sufficient boundary signal
        return float(feat["depth_change"])

    if metric == "drift":
        # Embedding drift only — for long documents with gradual topic migration
        # (the LSTM will accumulate drift across many soft boundaries)
        return float(feat["drift"])

    # ── "hybrid" (default) — weighted combination tuned for financial docs ──
    # Weight rationale:
    #   0.30 JSD:          reliable baseline, always informative
    #   0.20 Hellinger:    sensitive to rare key terms
    #   0.25 PMI-drop:     highest weight — best at detecting legal concept shifts
    #   0.15 depth_change: catches Article N / TITRE boundaries structurally
    #   0.10 drift:        long-range semantic migration detection
    return float(np.clip(
        0.30 * feat["jsd"]
        + 0.20 * feat["hellinger"]
        + 0.25 * feat["pmi_drop"]
        + 0.15 * feat["depth_change"]
        + 0.10 * feat["drift"],
        0.0, 1.0,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def refine_boundaries(chunks: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    """
    Evaluate every adjacent chunk pair and decide: merge, hard, or soft.

    Algorithm (4 passes)
    ────────────────────
    Pass 1: Compute 5-dim feature vector for every N-1 adjacent pair.
    Pass 2: Run the LSTM forward over the feature sequence → lstm_scores[].
    Pass 3: Compute the combined signal (raw + LSTM) and adaptive thresholds.
    Pass 4: Iterate through chunks and apply the merge/hard/soft decision.
            Merges immediately recompute the next boundary's features so the
            decision uses the actual merged content, not stale pre-merged data.

    Output fields added to every chunk
    ───────────────────────────────────
    jsd_score        — raw JSD at this boundary (for chart)
    metric_score     — combined signal value (raw × 0.65 + lstm × 0.35)
    boundary_signal  — alias of metric_score (legacy field kept for frontend)
    hidden_state     — LSTM scalar output ∈ [0,1]
    lstm_cell        — LSTM cell state vector (12-dim list)
    boundary_type    — "hard" | "soft" | "merged" | "end" | "single"
    merge_reason     — human-readable explanation of the decision
    boundary_features — dict of all 5 individual signals + lstm_score + combined

    Last chunk also receives:
    thresholds       — {low, high, mode} actually used
    s3_stats         — {initial_count, final_count, merged_count, …}
    """

    # ── Edge cases ────────────────────────────────────────────────────────
    if not chunks:
        return chunks  # nothing to process

    if len(chunks) == 1:
        # Single chunk: no boundary to evaluate — annotate with neutral values
        c = dict(chunks[0])
        c.update({
            "jsd_score":        0.0,
            "metric_score":     0.0,
            "boundary_signal":  0.0,
            "hidden_state":     0.0,
            "lstm_cell":        [],
            "boundary_type":    "single",
            "boundary_features": {},
            "s3_stats":         _empty_stats(1, 1),
        })
        return [c]

    # ── Read configuration ────────────────────────────────────────────────
    # threshold_mode: "fixed" or "percentile"
    mode         = str(config.get("threshold_mode",       "percentile")).lower()

    # Fixed threshold fallback values (used when mode=="fixed")
    tau_low_cfg  = float(config.get("tau_jsd_low",        0.15))
    tau_high_cfg = float(config.get("tau_jsd_high",       0.45))

    # Maximum allowed tokens per chunk (enforces S2 size constraint on merges)
    n_max        = int(config.get("n_max",                500))

    # Percentile boundaries for adaptive threshold (used when mode=="percentile")
    pct_low      = float(config.get("tau_percentile_low",  25))
    pct_high     = float(config.get("tau_percentile_high", 75))

    # Entropy metric selector — maps to _raw_signal_from_features() logic
    # This reads the frontend dropdown value ("jsd", "hellinger", "pmi",
    # "depth", "drift", or "hybrid")
    metric = str(config.get("entropy_metric", "hybrid")).lower()

    # ── Pass 1: compute 5-dim feature vectors for all N-1 boundaries ──────
    # pairs[i] holds features for the boundary between chunks[i] and chunks[i+1]
    pairs: List[Dict] = []
    for i in range(len(chunks) - 1):
        pairs.append(_boundary_features(chunks[i]["text"], chunks[i + 1]["text"]))

    # ── Pass 2: LSTM forward pass over the boundary sequence ──────────────
    # The LSTM processes all boundaries in document order.
    # After boundary 10, the cell state "remembers" signals at positions 0–9.
    lstm = _ForwardLSTMCell(seed=42)
    lstm.reset()  # reset state — CRITICAL when processing multiple documents

    lstm_scores: List[float]       = []   # scalar LSTM output per boundary
    lstm_cells:  List[List[float]] = []   # cell state per boundary (for JSON)

    for feat in pairs:
        # Build the 5-dim input vector from the feature dict
        x = np.array([
            feat["jsd"],           # Signal 1: JSD
            feat["hellinger"],     # Signal 2: Hellinger
            feat["pmi_drop"],      # Signal 3: PMI-drop
            feat["depth_change"],  # Signal 4: structural depth
            feat["drift"],         # Signal 5: embedding drift
        ], dtype=np.float32)

        # One LSTM step: updates internal h and c, returns score and cell copy
        score, cell = lstm.step(x)
        lstm_scores.append(score)
        lstm_cells.append(cell.tolist())

    # ── Pass 3a: combined signals ─────────────────────────────────────────
    # combined = 0.65 × raw_signal + 0.35 × lstm_score
    # Rationale: raw signal (direct text evidence) dominates at 65%;
    # LSTM modulates with long-range context at 35% without overriding it.
    raw_signals: List[float] = []
    for feat, ls in zip(pairs, lstm_scores):
        # Use the mode-selected raw signal (_raw_signal_from_features respects
        # the frontend "Entropy metric" dropdown)
        raw      = _raw_signal_from_features(feat, metric)
        combined = float(np.clip(0.65 * raw + 0.35 * ls, 0.0, 1.0))
        raw_signals.append(combined)

    # ── Pass 3b: compute adaptive thresholds ──────────────────────────────
    if mode == "percentile" and raw_signals:
        # Percentile mode: τ_low and τ_high derived from THIS document's own
        # signal distribution.  Recommended for financial/regulatory text
        # where the absolute entropy range varies between documents.
        tau_low  = float(np.percentile(raw_signals, pct_low))
        tau_high = float(np.percentile(raw_signals, pct_high))
    else:
        # Fixed mode: use slider values from the frontend directly
        tau_low  = tau_low_cfg
        tau_high = tau_high_cfg

    # Safety guard: ensure τ_low < τ_high with at least a 0.08 gap.
    # Can occur in percentile mode when the document has very uniform signals.
    if tau_low >= tau_high:
        tau_high = min(1.0, tau_low + 0.08)

    # ── Pass 4: iterate and apply merge / hard / soft decisions ───────────
    work = [dict(c) for c in chunks]   # working copy (mutated during merges)
    out:  List[Dict] = []              # accumulates the final chunk list

    # Running counters for s3_stats
    merged_count    = 0
    hard_count      = 0
    soft_count      = 0
    protected_count = 0
    signal_history: List[float] = []  # for mean_signal stat

    i = 0
    while i < len(work):
        curr = dict(work[i])

        if i < len(work) - 1:
            # ── There is a next chunk: evaluate the boundary ───────────────

            # pair_idx: index into pre-computed arrays.
            # After merges, len(work) decreases but raw_signals stays the same
            # length — clamp to avoid out-of-bounds access.
            pair_idx = min(i, len(raw_signals) - 1)

            signal   = raw_signals[pair_idx]    # combined boundary signal
            feat     = pairs[pair_idx] if pair_idx < len(pairs) else {}
            ls       = lstm_scores[pair_idx]    # LSTM scalar output
            lc       = lstm_cells[pair_idx]     # LSTM cell state list

            # Check if the NEXT chunk begins with a protected structural marker
            protected = _is_protected_boundary(work[i + 1].get("text", ""))

            # ── Annotate the current chunk with boundary metadata ──────────
            # These fields appear in the output JSON and the Pipeline Inspector

            # Raw JSD (for the S3 boundary chart in the frontend)
            curr["jsd_score"]       = round(feat.get("jsd", 0.0), 4)

            # Combined signal (the primary boundary decision value)
            curr["metric_score"]    = round(signal, 4)

            # Alias kept for frontend backwards compatibility
            curr["boundary_signal"] = round(signal, 4)

            # True LSTM output scalar (not EMA — genuine gated LSTM)
            curr["hidden_state"]    = round(ls, 4)

            # Full 12-dim cell state vector (visualised in Pipeline Inspector)
            curr["lstm_cell"]       = [round(v, 4) for v in lc]

            # All 5 individual signal values + aggregates (read by S7 _state_vec)
            curr["boundary_features"] = {
                "jsd":          round(feat.get("jsd",          0.0), 4),
                "hellinger":    round(feat.get("hellinger",     0.0), 4),
                "pmi_drop":     round(feat.get("pmi_drop",      0.0), 4),
                "depth_change": round(feat.get("depth_change",  0.0), 4),
                "drift":        round(feat.get("drift",         0.0), 4),
                "lstm_score":   round(ls,     4),   # LSTM output
                "combined":     round(signal, 4),   # final combined signal
            }
            signal_history.append(signal)

            # Size check: only merge if the combined chunk fits within the limit.
            # Allows up to 135% of n_max or n_max+80 tokens (whichever is larger)
            # to handle near-boundary cases without hard truncation.
            curr_words = len(curr.get("text", "").split())
            next_words = len(work[i + 1].get("text", "").split())
            size_ok    = (curr_words + next_words) <= max(n_max * 1.35, n_max + 80)

            # ── Decision tree (priority order) ───────────────────────────
            if protected:
                # CASE 1: Protected boundary — ALWAYS hard split.
                # Article N, TITRE, ALL-CAPS heading etc. override every signal.
                curr["boundary_type"] = "hard"
                curr["merge_reason"]  = "protected_structure_boundary"
                protected_count += 1
                hard_count      += 1
                out.append(curr)
                i += 1

            elif signal < tau_low and size_ok:
                # CASE 2: Low signal AND size fits → MERGE.
                # The chunks are too similar; absorb the next chunk into current.
                nxt = work[i + 1]

                # Concatenate texts with a blank-line separator
                curr["text"]          = curr["text"] + "\n\n" + nxt["text"]
                curr["end"]           = nxt.get("end", curr.get("end", 0))
                curr["boundary_type"] = "merged"
                curr["merge_reason"]  = "low_entropy_boundary"

                # Update the working list: replace work[i] and delete work[i+1]
                work[i] = curr
                work.pop(i + 1)

                # After merging, the left side of the NEXT boundary has changed.
                # Recompute features for the new (merged chunk → next chunk) boundary
                # so the next iteration evaluates the actual merged content.
                if i < len(work) - 1:
                    new_feat = _boundary_features(curr["text"], work[i + 1]["text"])
                    new_x    = np.array([
                        new_feat["jsd"],
                        new_feat["hellinger"],
                        new_feat["pmi_drop"],
                        new_feat["depth_change"],
                        new_feat["drift"],
                    ], dtype=np.float32)

                    # Run one more LSTM step with the new boundary features
                    # (LSTM state is NOT reset — merge is treated as a new observation)
                    new_ls, new_lc = lstm.step(new_x)

                    # Recompute the combined signal for the new boundary
                    new_raw = _raw_signal_from_features(new_feat, metric)
                    new_sig = float(np.clip(0.65 * new_raw + 0.35 * new_ls, 0.0, 1.0))

                    # Patch the pre-computed arrays so the next iteration
                    # uses the updated values
                    if pair_idx < len(raw_signals):
                        raw_signals[pair_idx] = new_sig
                        pairs[pair_idx]        = new_feat
                        lstm_scores[pair_idx]  = new_ls
                        lstm_cells[pair_idx]   = new_lc.tolist()

                merged_count += 1
                # Do NOT increment i — re-evaluate the same index with the
                # now-larger work[i] against its new right neighbour
                continue

            elif signal > tau_high:
                # CASE 3: High signal → confirmed topic shift → HARD split.
                curr["boundary_type"] = "hard"
                curr["merge_reason"]  = "high_entropy_shift"
                hard_count += 1
                out.append(curr)
                i += 1

            else:
                # CASE 4: Signal between thresholds → SOFT boundary.
                # Pass to S4 (boundary quality filter) for further evaluation
                # using CodeBLEU-style scoring and semantic similarity.
                curr["boundary_type"] = "soft"
                curr["merge_reason"]  = "moderate_entropy_shift"
                soft_count += 1
                out.append(curr)
                i += 1

        else:
            # ── Last chunk: no next neighbour ────────────────────────────
            # Assign zero/neutral metadata and mark as "end"
            curr["jsd_score"]        = 0.0
            curr["metric_score"]     = 0.0
            curr["boundary_signal"]  = 0.0
            curr["hidden_state"]     = 0.0
            curr["lstm_cell"]        = []
            curr["boundary_features"] = {}
            curr["boundary_type"]    = "end"
            out.append(curr)
            i += 1

    # ── Attach summary statistics to the last chunk ───────────────────────
    # Visible in the Review Findings tab of the frontend
    if out:
        out[-1]["thresholds"] = {
            "low":  round(tau_low,  4),   # τ_low actually used (may differ from config)
            "high": round(tau_high, 4),   # τ_high actually used
            "mode": mode,                  # "fixed" or "percentile"
        }
        out[-1]["s3_stats"] = {
            "initial_count":   len(chunks),       # chunks entering S3
            "final_count":     len(out),           # chunks leaving S3 (after merges)
            "merged_count":    merged_count,       # number of boundaries merged
            "hard_count":      hard_count,         # hard splits (incl. protected)
            "soft_count":      soft_count,         # soft boundaries passed to S4
            "protected_count": protected_count,    # subset of hard: structure-based
            "mean_signal":     round(float(np.mean(signal_history)), 4) if signal_history else 0.0,
            "merge_ratio":     round(merged_count / max(1, len(chunks) - 1), 4),
        }
    return out


def get_jsd_series(chunks: List[Dict]) -> List[float]:
    """
    Extract the boundary signal series for the Pipeline Inspector chart.
    Returns metric_score (combined signal) for all chunks.
    The last chunk (boundary_type="end") always contributes 0.0.
    """
    return [c.get("metric_score", c.get("jsd_score", 0.0)) for c in chunks]


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation — one function per signal
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_features(text_a: str, text_b: str) -> Dict[str, float]:
    """
    Compute the complete 5-dim feature vector for one boundary.
    text_a = text of the LEFT chunk (the one ending at this boundary).
    text_b = text of the RIGHT chunk (the one starting after the boundary).
    All returned values are guaranteed to be in [0, 1].
    """
    return {
        "jsd":          _compute_jsd(text_a, text_b),
        "hellinger":    _compute_hellinger(text_a, text_b),
        "pmi_drop":     _compute_pmi_drop(text_a, text_b),
        "depth_change": _compute_depth_change(text_b),   # only depends on text_b
        "drift":        _compute_drift(text_a, text_b),
    }


# ── Signal 1 — Jensen-Shannon Divergence ─────────────────────────────────────

def _compute_jsd(t1: str, t2: str) -> float:
    """
    JSD(P, Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),   M = (P+Q)/2

    P and Q are the unigram token distributions of t1 and t2.
    Symmetric and bounded in [0, 1].
    High value = very different token distributions → likely topic boundary.
    Returns 0.5 (neutral) if either text is empty.
    """
    p, q = _distribution_pair(t1, t2)
    if p is None:
        return 0.5
    m = (p + q) / 2.0
    # KL(P‖M) + KL(Q‖M) divided by 2 gives the symmetric JSD
    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


# ── Signal 2 — Hellinger Distance ────────────────────────────────────────────

def _compute_hellinger(t1: str, t2: str) -> float:
    """
    H(P, Q) = ‖√P − √Q‖₂ / √2

    More sensitive than JSD to changes in low-probability (rare) terms.
    For legal documents, new articles often introduce a few specific terms
    with low overall frequency — Hellinger detects this shift better.
    Returns 0.5 (neutral) if either text is empty.
    """
    p, q = _distribution_pair(t1, t2)
    if p is None:
        return 0.5
    # Element-wise sqrt of each distribution, then L2 norm of their difference
    return float(np.clip(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2), 0.0, 1.0))


# ── Signal 3 — PMI-drop ──────────────────────────────────────────────────────

def _compute_pmi_drop(t1: str, t2: str) -> float:
    """
    Key concept shift detector.

    Extracts the TOP-K (K=8) most frequent content words from each chunk
    and measures how much those key concept sets diverge:

        pmi_drop = 1 − |top1 ∩ top2| / |top1 ∪ top2|

    High pmi_drop → the chunks' dominant concepts are completely different
                    → strong evidence of a real topic boundary.

    WHY this matters for financial/regulatory text:
    ─────────────────────────────────────────────────
    Every article of a tax treaty shares surface vocabulary:
      "impôt", "État contractant", "Convention", "résident" appear everywhere.
    But each article has DIFFERENT dominant concepts:
      Article 3 → {"définitions","stable","entreprise","société"}
      Article 4 → {"domicile","foyer","habitation","séjour","intérêts","vitaux"}
    JSD would be LOW (same surface vocab) but pmi_drop is HIGH (different key concepts).
    This is why pmi_drop has the highest weight (0.25) in the hybrid formula.
    """
    toks1 = _content_tokens(t1)
    toks2 = _content_tokens(t2)
    if not toks1 or not toks2:
        return 0.5

    K = 8   # number of top content terms per chunk

    # Count term frequencies in each chunk
    freq1: Dict[str, int] = {}
    for t in toks1:
        freq1[t] = freq1.get(t, 0) + 1
    freq2: Dict[str, int] = {}
    for t in toks2:
        freq2[t] = freq2.get(t, 0) + 1

    # Top-K content terms by frequency
    top1 = {w for w, _ in sorted(freq1.items(), key=lambda x: -x[1])[:K]}
    top2 = {w for w, _ in sorted(freq2.items(), key=lambda x: -x[1])[:K]}

    # Jaccard overlap of the top-K sets
    shared = top1 & top2
    union  = top1 | top2
    if not union:
        return 0.5

    # pmi_drop = 1 - overlap_ratio
    # Low overlap of top terms → high drop → strong boundary signal
    overlap_ratio = len(shared) / len(union)
    return float(np.clip(1.0 - overlap_ratio, 0.0, 1.0))


# ── Signal 4 — Structural depth change ───────────────────────────────────────

def _compute_depth_change(text_b: str) -> float:
    """
    Detect whether chunk B opens at a high level of the document hierarchy.

    Examines the FIRST non-empty line of text_b and matches it against
    _DEPTH_MARKERS.  Returns a normalized depth score:
        depth=1 (TITRE, CHAPITRE) → 1.00
        depth=2 (SECTION)         → 0.83
        depth=3 (Article N)       → 0.67
        depth=4 (3.2.1 style)     → 0.50
        depth=5 (a. b. c.)        → 0.33
        depth=6 (bullet)          → 0.17
        no match                  → 0.00

    Note: this is a structural/regex signal, NOT a distributional metric.
    It is included because for French legal text it is the most reliable
    single boundary indicator when JSD/Hellinger/PMI are all ambiguous.
    """
    first_line = ""
    for line in text_b.splitlines():
        if line.strip():
            first_line = line.strip()
            break
    if not first_line:
        return 0.0

    for pattern, depth in _DEPTH_MARKERS:
        if pattern.match(first_line):
            # Normalize: depth=1 → 1.0, depth=6 → (1 - 5/6) ≈ 0.17
            return float(np.clip(1.0 - (depth - 1) / 6.0, 0.0, 1.0))
    return 0.0  # no structural marker found


# ── Signal 5 — Embedding drift ────────────────────────────────────────────────

def _compute_drift(t1: str, t2: str) -> float:
    """
    Cosine DISTANCE between lightweight hash embeddings of t1 and t2.

        drift = 1 − cos(v1, v2)

    v1 and v2 are 64-dim bag-of-content-words hash vectors.  Unlike S4's
    cosine similarity (which uses a trained sentence-transformer), this is
    computed instantly for ALL boundaries in Pass 1.

    The key value for the LSTM:
    ────────────────────────────
    If drift = [0.3, 0.4, 0.4, 0.5, 0.6] across 5 consecutive boundaries,
    the LSTM's forget gate retains this upward trend in the cell state.
    Boundary 6 will therefore be assessed in the context of an ALREADY drifting
    document — even if its individual drift score is only 0.5, the LSTM may
    output a higher boundary confidence because it "remembers" the trend.
    This is the core advantage of the LSTM over any per-boundary metric.

    Returns 0.5 (neutral) if either embedding is the zero vector.
    """
    v1 = _hash_embed(t1)
    v2 = _hash_embed(t2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.5  # neutral fallback for very short / empty chunks

    # cosine similarity → cosine distance (high distance = semantic drift)
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.clip(1.0 - cosine, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Protected boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_protected_boundary(text: str) -> bool:
    """
    Return True if the first non-empty line of text matches _PROTECTED_RE.
    Called with work[i+1].text in Pass 4 to check if the NEXT chunk starts
    with a structural marker that must always produce a hard split.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            # Only check the FIRST non-empty line
            return bool(_PROTECTED_RE.match(stripped))
    return False  # text was entirely whitespace


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _distribution_pair(t1: str, t2: str):
    """
    Build aligned probability distribution arrays for t1 and t2.

    Returns (p, q) as numpy float64 arrays over the JOINT vocabulary (t1 ∪ t2).
    Both arrays sum to 1.0 (proper probability distributions).
    Returns (None, None) if either text produces zero tokens.

    The joint vocabulary ensures both arrays have the same length,
    which is required for JSD and Hellinger computation.
    """
    tok1 = _tokenize(t1)
    tok2 = _tokenize(t2)
    if not tok1 or not tok2:
        return None, None  # type: ignore[return-value]

    # Joint vocabulary: all unique tokens from either chunk
    vocab = list(set(tok1) | set(tok2))

    # Count term frequencies
    c1: Dict[str, int] = {}
    c2: Dict[str, int] = {}
    for t in tok1:
        c1[t] = c1.get(t, 0) + 1
    for t in tok2:
        c2[t] = c2.get(t, 0) + 1

    # Convert to relative frequency (probability) arrays
    # Tokens absent from a chunk get probability 0
    p = np.array([c1.get(w, 0) / len(tok1) for w in vocab], dtype=np.float64)
    q = np.array([c2.get(w, 0) / len(tok2) for w in vocab], dtype=np.float64)
    return p, q


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """
    KL(P ‖ Q) = Σ_x P(x) log(P(x) / Q(x))

    By convention: 0 · log(0) = 0 (positions where p=0 are skipped).
    q is clipped to 1e-12 to prevent log(0) / division by zero.
    """
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, 1.0))))


def _tokenize(text: str) -> List[str]:
    """
    Extract all word tokens (including stopwords) via a word-boundary regex.
    Lowercased.  Used by _distribution_pair() for JSD and Hellinger
    (which need ALL tokens to build accurate probability distributions).
    """
    return re.findall(r"\b\w+\b", text.lower())


def _content_tokens(text: str) -> List[str]:
    """
    Extract content-bearing tokens only (stopwords removed).
    Matches Unicode word characters including French accented letters (À-ÿ).
    Minimum length 2 to filter single-letter tokens.

    Used by: PMI-drop (top-K key terms) and _hash_embed (drift signal).
    NOT used by JSD/Hellinger — those need full distributions with stopwords.

    Falls back to all tokens if filtering leaves an empty list
    (e.g. a very short chunk consisting entirely of stopwords).
    """
    toks = [
        t for t in re.findall(r"\b[\wÀ-ÿ]{2,}\b", text.lower())
        if t not in _STOPWORDS
    ]
    return toks or _tokenize(text)   # safe fallback


def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """
    Lightweight bag-of-content-words hash embedding.

    Maps each content token to a position in a dim-dimensional vector using
    Python's hash(), then accumulates term frequency at that position.
    Result: a 64-dim sparse vector representing the active vocabulary subspace.

    dim=64: cheap to compute for all boundaries in Pass 1, while still
    providing meaningful cosine distance for the drift signal.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _content_tokens(text):
        vec[hash(tok) % dim] += 1.0   # accumulate TF at hashed index
    return vec


def _empty_stats(initial: int, final: int) -> Dict[str, Any]:
    """Return a neutral s3_stats dict for the single-chunk edge case."""
    return {
        "initial_count":   initial,
        "final_count":     final,
        "merged_count":    0,
        "hard_count":      0,
        "soft_count":      0,
        "protected_count": 0,
        "mean_signal":     0.0,
        "merge_ratio":     0.0,
    }


# The Dict import appears twice due to __future__ annotations; the one below
# is the runtime import needed for type annotations in the helper functions.
from typing import Dict  # noqa: E402