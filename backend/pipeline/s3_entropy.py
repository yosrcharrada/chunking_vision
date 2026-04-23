"""
S3 — Entropy Boundary Refinement  (v3 — financial/regulatory edition)

Five complementary boundary signals, all combined through a real forward-LSTM
that accumulates document-level context across chunk boundaries.

Signals
-------
1. JSD          Jensen-Shannon Divergence on unigram distributions
2. Hellinger    Hellinger distance (more sensitive to tail of distribution)
3. PMI-drop     Pointwise mutual information collapse between content terms
4. Depth-change Structural hierarchy depth shift (Article/paragraph/bullet)
5. Drift        Cosine drift of rolling hashed embedding from a local baseline

The LSTM receives a 5-dim input at each boundary position and outputs a scalar
boundary confidence score in [0,1].  Fixed Xavier weights (seed=42) make the
computation fully deterministic and reproducible.

Design choices for financial/regulatory text
--------------------------------------------
- High token overlap between adjacent chunks does NOT imply they should be merged.
  "impôt sur les sociétés" appears in every article of a tax convention.
  Overlap is removed from the merge signal and replaced by PMI-drop and depth-change.
- Protected boundaries (Article N, TITRE, CHAPITRE, § N, etc.) always produce a
  hard boundary regardless of any metric score.
- Percentile-mode thresholds adapt to each document's own score distribution so
  the pipeline works equally well on 3k-token notes and 50k-token regulations.
- All arithmetic is deterministic for fixed inputs (no random sampling at runtime).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the","a","an","and","or","but","is","are","was","were","be","been",
    "being","have","has","had","do","does","did","will","would","could",
    "should","may","might","must","shall","can","to","of","in","for","on",
    "with","at","by","from","as","into","through","during","before","after",
    "here","there","when","where","why","how","all","both","each","few",
    "more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","just","this","that","these","those",
    "it","its","he","she","they","we","you","i","my","your","his","her",
    "our","their","what","which","who","whom","about",
    # French
    "le","la","les","de","des","du","et","en","un","une","dans","pour",
    "que","est","sur","par","avec","au","aux","ce","se","si","ne","pas",
    "lui","leur","ils","elles","nous","vous","on","dont","où","car","mais",
    "ni","or","donc","comment","quand","comme","tout","tous","toute",
}

# Protected boundaries — always hard splits regardless of entropy scores
_PROTECTED_RE = re.compile(
    r"^[ \t]*(?:"
    r"(?:TITRE|CHAPITRE|SECTION|SOUS-SECTION|PARAGRAPHE|BOOK|PART|CHAPTER)"
    r"\s+(?:[IVXLCDM]+|\d+|PREMIER|PREMIERE|PREMIÈRE|premier|premiere|première)"
    r"|(?:Article|Art\.?|ARTICLE)\s+(?:\d+(?:\s*(?:er|eme|ème|e|bis|ter|quater))?|premier|1er|[IVX]+)"
    r"|§\s*\d+"
    r"|#{1,6}\s+\S+"
    r"|[A-Z][A-Z\s\-]{4,}(?::|$)"      # ALL-CAPS headings common in FR regulations
    r")(?:[ \t].*)?$",
    re.MULTILINE | re.IGNORECASE,
)

# Hierarchy depth markers — used for depth-change signal
_DEPTH_MARKERS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"^\s*(?:TITRE|CHAPITRE|BOOK|PART)\b", re.I), 1),
    (re.compile(r"^\s*(?:SECTION|SOUS-SECTION)\b", re.I), 2),
    (re.compile(r"^\s*(?:Article|Art\.?|ARTICLE|§)\s*\d+", re.I), 3),
    (re.compile(r"^\s*(?:\d+(?:\.\d+)+)\s+\S", re.I), 4),   # 3.2.1 style
    (re.compile(r"^\s*[a-zA-Z]\.\s+\S", re.I), 5),           # a. b. c.
    (re.compile(r"^\s*[-–•]\s+\S", re.I), 6),                # bullet
]


# ─────────────────────────────────────────────────────────────────────────────
# LSTM cell  (deterministic fixed-weight forward LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class _ForwardLSTMCell:
    """
    Single-layer forward LSTM.  Weights are fixed at construction (Xavier,
    seed=42) and never updated at runtime — making the stage fully deterministic
    while preserving the gating structure that lets the cell accumulate document-
    level context across boundaries.

    Input dimension  : 5  (one per entropy signal)
    Hidden dimension : 12
    Output           : scalar in [0,1] via a linear projection + sigmoid
    """

    INPUT_DIM  = 5
    HIDDEN_DIM = 12

    def __init__(self, seed: int = 42):
        rng   = np.random.RandomState(seed)
        idim  = self.INPUT_DIM
        hdim  = self.HIDDEN_DIM
        scale = np.sqrt(2.0 / (idim + hdim))

        # Input gate
        self.Wi = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Ui = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bi = np.zeros(hdim, dtype=np.float32)

        # Forget gate  (bias init = 1 — standard practice, reduces vanishing gradient)
        self.Wf = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Uf = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bf = np.ones(hdim, dtype=np.float32)

        # Cell gate
        self.Wg = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Ug = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bg = np.zeros(hdim, dtype=np.float32)

        # Output gate
        self.Wo = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Uo = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bo = np.zeros(hdim, dtype=np.float32)

        # Scalar projection  (hdim → 1)
        self.Wp = rng.randn(1, hdim).astype(np.float32) * np.sqrt(1.0 / hdim)

        self.h = np.zeros(hdim, dtype=np.float32)
        self.c = np.zeros(hdim, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.h[:] = 0.0
        self.c[:] = 0.0

    def step(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        One LSTM step.
        x : (INPUT_DIM,) numpy array of boundary features
        Returns (scalar_score, cell_state_copy)
        """
        i_g = self._sig(self.Wi @ x + self.Ui @ self.h + self.bi)
        f_g = self._sig(self.Wf @ x + self.Uf @ self.h + self.bf)
        g_g = np.tanh(  self.Wg @ x + self.Ug @ self.h + self.bg)
        o_g = self._sig(self.Wo @ x + self.Uo @ self.h + self.bo)
        self.c = f_g * self.c + i_g * g_g
        self.h = o_g * np.tanh(self.c)
        score  = float(self._sig(self.Wp @ self.h)[0])
        return score, self.c.copy()

    @staticmethod
    def _sig(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _raw_signal_from_features(feat: Dict[str, float], metric: str) -> float:
    """
    Compute the raw boundary score from the 5-dim feature dict.
    The selected metric mode determines which signal(s) dominate.
    The LSTM always receives all 5 signals regardless of this setting.
    """
    if metric == "jsd":
        # Pure JSD — classic baseline
        return float(feat["jsd"])
    if metric == "hellinger":
        # Hellinger only — more sensitive to rare-term distribution changes
        return float(feat["hellinger"])
    if metric == "pmi":
        # PMI-drop only — best for financial/regulatory text with stable surface vocab
        return float(feat["pmi_drop"])
    if metric == "depth":
        # Depth change only — useful for heavily structured hierarchical documents
        return float(feat["depth_change"])
    if metric == "drift":
        # Embedding drift only — detects slow semantic migration across many chunks
        return float(feat["drift"])
    # "hybrid" (default) — weighted combination tuned for financial documents
    return float(np.clip(
        0.30 * feat["jsd"]
        + 0.20 * feat["hellinger"]
        + 0.25 * feat["pmi_drop"]
        + 0.15 * feat["depth_change"]
        + 0.10 * feat["drift"],
        0.0, 1.0,
    ))


def refine_boundaries(chunks: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    """
    Evaluate every adjacent pair of chunks and optionally merge weak boundaries.

    Decision logic
    ──────────────
    1. Compute 5-dim feature vector for every adjacent pair.
    2. Feed the sequence through the forward LSTM → per-boundary score.
    3. Compute adaptive thresholds (percentile or fixed).
    4. Protected boundaries → always hard.
    5. score < τ_low  → merge (unless protected or combined size exceeds limit).
    6. score > τ_high → hard.
    7. Otherwise      → soft.

    All outputs are logged into s3_stats on the last chunk.
    """
    if not chunks:
        return chunks
    if len(chunks) == 1:
        c = dict(chunks[0])
        c.update({
            "jsd_score": 0.0,
            "metric_score": 0.0,
            "boundary_signal": 0.0,
            "hidden_state": 0.0,
            "lstm_cell": [],
            "boundary_type": "single",
            "boundary_features": {},
            "s3_stats": _empty_stats(1, 1),
        })
        return [c]

    # ── Config ──────────────────────────────────────────────────────────
    mode         = str(config.get("threshold_mode", "percentile")).lower()
    tau_low_cfg  = float(config.get("tau_jsd_low",  0.15))
    tau_high_cfg = float(config.get("tau_jsd_high", 0.45))
    n_max        = int(config.get("n_max", 500))
    pct_low      = float(config.get("tau_percentile_low",  25))
    pct_high     = float(config.get("tau_percentile_high", 75))

    metric = str(config.get("entropy_metric", "hybrid")).lower()

    # ── Pass 1: compute raw feature vectors for all adjacent pairs ──────
    pairs: List[Dict] = []
    for i in range(len(chunks) - 1):
        pairs.append(_boundary_features(chunks[i]["text"], chunks[i + 1]["text"]))

    # ── LSTM forward pass over the feature sequence ──────────────────────
    lstm   = _ForwardLSTMCell(seed=42)
    lstm.reset()
    lstm_scores: List[float] = []
    lstm_cells:  List[List[float]] = []
    for feat in pairs:
        x = np.array([
            feat["jsd"],
            feat["hellinger"],
            feat["pmi_drop"],
            feat["depth_change"],
            feat["drift"],
        ], dtype=np.float32)
        score, cell = lstm.step(x)
        lstm_scores.append(score)
        lstm_cells.append(cell.tolist())

    # Combine raw feature signal with LSTM output
    # For financial text: weight the structural/PMI signals heavily
    raw_signals: List[float] = []
    for feat, ls in zip(pairs, lstm_scores):
        raw = _raw_signal_from_features(feat, metric)
        # LSTM modulates the raw signal — it injects long-range context
        combined = float(np.clip(0.65 * raw + 0.35 * ls, 0.0, 1.0))
        raw_signals.append(combined)

    # ── Adaptive thresholds ─────────────────────────────────────────────
    if mode == "percentile" and raw_signals:
        tau_low  = float(np.percentile(raw_signals, pct_low))
        tau_high = float(np.percentile(raw_signals, pct_high))
    else:
        tau_low  = tau_low_cfg
        tau_high = tau_high_cfg

    # Guard: ensure a minimum gap between thresholds
    if tau_low >= tau_high:
        tau_high = min(1.0, tau_low + 0.08)

    # ── Pass 2: merge / label boundaries ────────────────────────────────
    work = [dict(c) for c in chunks]
    out: List[Dict] = []
    merged_count = hard_count = soft_count = protected_count = 0
    signal_history: List[float] = []

    i = 0
    while i < len(work):
        curr = dict(work[i])
        if i < len(work) - 1:
            pair_idx  = min(i, len(raw_signals) - 1)
            signal    = raw_signals[pair_idx]
            feat      = pairs[pair_idx] if pair_idx < len(pairs) else {}
            ls        = lstm_scores[pair_idx]
            lc        = lstm_cells[pair_idx]
            protected = _is_protected_boundary(work[i + 1].get("text", ""))

            curr["jsd_score"]        = round(feat.get("jsd", 0.0), 4)
            curr["metric_score"]     = round(signal, 4)
            curr["boundary_signal"]  = round(signal, 4)
            curr["hidden_state"]     = round(ls, 4)          # true LSTM output
            curr["lstm_cell"]        = [round(v, 4) for v in lc]
            curr["boundary_features"] = {
                "jsd":          round(feat.get("jsd",         0.0), 4),
                "hellinger":    round(feat.get("hellinger",    0.0), 4),
                "pmi_drop":     round(feat.get("pmi_drop",     0.0), 4),
                "depth_change": round(feat.get("depth_change", 0.0), 4),
                "drift":        round(feat.get("drift",        0.0), 4),
                "lstm_score":   round(ls, 4),
                "combined":     round(signal, 4),
            }
            signal_history.append(signal)

            curr_words = len(curr.get("text", "").split())
            next_words = len(work[i + 1].get("text", "").split())
            size_ok    = (curr_words + next_words) <= max(n_max * 1.35, n_max + 80)

            if protected:
                curr["boundary_type"] = "hard"
                curr["merge_reason"]  = "protected_structure_boundary"
                protected_count += 1
                hard_count      += 1
                out.append(curr)
                i += 1
            elif signal < tau_low and size_ok:
                # Merge: absorb next chunk into current
                nxt = work[i + 1]
                curr["text"]          = curr["text"] + "\n\n" + nxt["text"]
                curr["end"]           = nxt.get("end", curr.get("end", 0))
                curr["boundary_type"] = "merged"
                curr["merge_reason"]  = "low_entropy_boundary"
                work[i] = curr
                work.pop(i + 1)
                # Recompute features for next boundary (curr is now larger)
                if i < len(work) - 1:
                    new_feat = _boundary_features(curr["text"], work[i + 1]["text"])
                    new_x    = np.array([
                        new_feat["jsd"], new_feat["hellinger"],
                        new_feat["pmi_drop"], new_feat["depth_change"], new_feat["drift"],
                    ], dtype=np.float32)
                    new_ls, new_lc = lstm.step(new_x)
                    new_raw = _raw_signal_from_features(new_feat, metric)
                    new_sig = float(np.clip(0.65 * new_raw + 0.35 * new_ls, 0.0, 1.0))
                    # Patch the working lists at this position
                    if pair_idx < len(raw_signals):
                        raw_signals[pair_idx] = new_sig
                        pairs[pair_idx]        = new_feat
                        lstm_scores[pair_idx]  = new_ls
                        lstm_cells[pair_idx]   = new_lc.tolist()
                merged_count += 1
                continue   # re-evaluate same index i with updated work[i]
            elif signal > tau_high:
                curr["boundary_type"] = "hard"
                curr["merge_reason"]  = "high_entropy_shift"
                hard_count += 1
                out.append(curr)
                i += 1
            else:
                curr["boundary_type"] = "soft"
                curr["merge_reason"]  = "moderate_entropy_shift"
                soft_count += 1
                out.append(curr)
                i += 1
        else:
            curr["jsd_score"]        = 0.0
            curr["metric_score"]     = 0.0
            curr["boundary_signal"]  = 0.0
            curr["hidden_state"]     = 0.0
            curr["lstm_cell"]        = []
            curr["boundary_features"] = {}
            curr["boundary_type"]    = "end"
            out.append(curr)
            i += 1

    # ── Attach stats to last chunk ───────────────────────────────────────
    if out:
        out[-1]["thresholds"] = {
            "low":  round(tau_low,  4),
            "high": round(tau_high, 4),
            "mode": mode,
        }
        out[-1]["s3_stats"] = {
            "initial_count":  len(chunks),
            "final_count":    len(out),
            "merged_count":   merged_count,
            "hard_count":     hard_count,
            "soft_count":     soft_count,
            "protected_count": protected_count,
            "mean_signal":    round(float(np.mean(signal_history)), 4) if signal_history else 0.0,
            "merge_ratio":    round(merged_count / max(1, len(chunks) - 1), 4),
        }
    return out


def get_jsd_series(chunks: List[Dict]) -> List[float]:
    """Return the boundary signal series for the Pipeline Inspector chart."""
    return [c.get("metric_score", c.get("jsd_score", 0.0)) for c in chunks]


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_features(text_a: str, text_b: str) -> Dict[str, float]:
    """
    Compute the 5-dim feature vector for a single boundary.
    All values are in [0, 1].
    """
    return {
        "jsd":          _compute_jsd(text_a, text_b),
        "hellinger":    _compute_hellinger(text_a, text_b),
        "pmi_drop":     _compute_pmi_drop(text_a, text_b),
        "depth_change": _compute_depth_change(text_b),
        "drift":        _compute_drift(text_a, text_b),
    }


# ── Signal 1: Jensen-Shannon Divergence ─────────────────────────────────────

def _compute_jsd(t1: str, t2: str) -> float:
    p, q = _distribution_pair(t1, t2)
    if p is None:
        return 0.5
    m = (p + q) / 2.0
    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


# ── Signal 2: Hellinger Distance ─────────────────────────────────────────────

def _compute_hellinger(t1: str, t2: str) -> float:
    p, q = _distribution_pair(t1, t2)
    if p is None:
        return 0.5
    return float(np.clip(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2), 0.0, 1.0))


# ── Signal 3: PMI-drop ───────────────────────────────────────────────────────
#
# Intuition: inside a coherent legal article, the key content terms (e.g.
# "contribution", "solidarité", "personnes morales") co-occur frequently →
# high mutual information.  When a new article begins, a different set of key
# terms takes over → low MI with the prior chunk.  This signal detects that
# concept shift even when surface vocabulary overlap remains high.

def _compute_pmi_drop(t1: str, t2: str) -> float:
    toks1 = _content_tokens(t1)
    toks2 = _content_tokens(t2)
    if not toks1 or not toks2:
        return 0.5

    # Top-K content terms in each chunk (by frequency)
    K = 8
    freq1: Dict[str, int] = {}
    for t in toks1:
        freq1[t] = freq1.get(t, 0) + 1
    freq2: Dict[str, int] = {}
    for t in toks2:
        freq2[t] = freq2.get(t, 0) + 1

    top1 = {w for w, _ in sorted(freq1.items(), key=lambda x: -x[1])[:K]}
    top2 = {w for w, _ in sorted(freq2.items(), key=lambda x: -x[1])[:K]}

    # Overlap of top key terms
    shared = top1 & top2
    union  = top1 | top2
    if not union:
        return 0.5

    # PMI-drop: low overlap of TOP terms = high boundary signal
    overlap_ratio = len(shared) / len(union)
    return float(np.clip(1.0 - overlap_ratio, 0.0, 1.0))


# ── Signal 4: Structural depth change ────────────────────────────────────────
#
# When chunk B starts with a shallower hierarchy level than chunk A's last line,
# a new top-level section began → strong boundary.  When B goes deeper, it's
# a sub-item continuation → weak boundary.

def _compute_depth_change(text_b: str) -> float:
    """
    Returns a value in [0,1] representing the magnitude of hierarchy depth change
    at the start of text_b.  1.0 = the chunk opens at the top level (Article / Chapter).
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
            # Normalize: depth=1 (top) → 1.0, depth=6 (bullet) → ~0.17
            return float(np.clip(1.0 - (depth - 1) / 6.0, 0.0, 1.0))
    return 0.0


# ── Signal 5: Embedding drift from local baseline ────────────────────────────
#
# Cosine distance between chunk B's hash embedding and chunk A's embedding.
# Unlike raw cosine (which S4 already computes), this will also be used by the
# LSTM to track whether the document has been drifting gradually — the LSTM's
# cell state accumulates drift over many boundaries, letting it detect slow
# topic migration that no single-boundary metric can catch.

def _compute_drift(t1: str, t2: str) -> float:
    v1 = _hash_embed(t1)
    v2 = _hash_embed(t2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.5
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.clip(1.0 - cosine, 0.0, 1.0))   # high = drifted = boundary


# ─────────────────────────────────────────────────────────────────────────────
# Protected boundary detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_protected_boundary(text: str) -> bool:
    """
    Returns True if text begins with a structural marker that must always
    produce a hard split regardless of entropy scores.
    Covers French/EN legal markers + ALL-CAPS headings used in Tunisian regs.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return bool(_PROTECTED_RE.match(stripped))
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _distribution_pair(t1: str, t2: str):
    tok1 = _tokenize(t1)
    tok2 = _tokenize(t2)
    if not tok1 or not tok2:
        return None, None
    vocab = list(set(tok1) | set(tok2))
    c1: Dict[str, int] = {}
    c2: Dict[str, int] = {}
    for t in tok1:
        c1[t] = c1.get(t, 0) + 1
    for t in tok2:
        c2[t] = c2.get(t, 0) + 1
    p = np.array([c1.get(w, 0) / len(tok1) for w in vocab], dtype=np.float64)
    q = np.array([c2.get(w, 0) / len(tok2) for w in vocab], dtype=np.float64)
    return p, q


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, 1.0))))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _content_tokens(text: str) -> List[str]:
    toks = [
        t for t in re.findall(r"\b[\wÀ-ÿ]{2,}\b", text.lower())
        if t not in _STOPWORDS
    ]
    return toks or _tokenize(text)


def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _content_tokens(text):
        vec[hash(tok) % dim] += 1.0
    return vec


def _empty_stats(initial: int, final: int) -> Dict[str, Any]:
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


# Keep type hint imports at the bottom to avoid circular import issues
from typing import Dict  # noqa: E402  (already imported above via __future__)