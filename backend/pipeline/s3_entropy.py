"""
S3 — qentropy Boundary Refinement  (Tsallis / Hill diversity-number driven)
===========================================================================
Pipeline position : runs AFTER S2 (chunkers) and BEFORE S4 (boundary quality).
Responsibility    : decide, using the **qentropy engine** (engine/entropy.py +
                    engine/tree_entropy.py), how many of S2's candidate
                    boundaries are *real* semantic shifts, then keep that many
                    and merge across the rest.

WHY THIS REWRITE
────────────────
This project merges two codebases.  The old S3 carried a bespoke palette of
boundary signals (JSD / Hellinger / PMI / depth / drift) plus a GPT-2 perplexity
validator and its own Tsallis math.  All of that is gone.  S3 now works *only*
with the second project's qentropy formulation:

  1. Each S2 chunk is treated as a *unit*.  We embed the units (shared
     EmbeddingService) and form the semantic-shift signal at every candidate
     boundary  d_i = 1 - cos(u_i, u_{i+1})            (engine.chunking._gap_distances style)

  2. The *significant* shifts (above the document's own median coherence) become
     a probability distribution p over boundaries (engine.entropy.normalize).

  3. The generalised entropy DRIVES the granularity:
         H   = Shannon entropy of p            (bits)
         S_q = Tsallis q-entropy of p          (bits)
         D_q = Hill diversity number of p      ("effective # of real boundaries")
     target_boundaries = round(D_q).  q<1 → finer, q>1 → coarser, q=1 → Shannon
     perplexity exp(H).  q is supplied by config (tuned by the S7 GA).

  4. P1's forward LSTM is KEPT and RE-WIRED: instead of the old 9-dim classical
     feature vector it now runs over a 5-dim feature sequence DERIVED FROM THE
     QENTROPY SIGNAL (normalised gap, probability mass, excess-over-baseline,
     pointwise Tsallis weight p^q, structural-prior flag).  Its context-aware
     score is blended with the raw shift to RANK boundaries before selection.

  5. Structural headings (Article / CHAPITRE / SECTION …) are protected priors,
     and a token-feasibility floor keeps every merged chunk above min_chunk_tokens.

  6. A paper-faithful tree entropy (engine.tree_entropy: Σ log Z_K) is computed
     over the resulting segmentation and exposed as a document fingerprint.

PUBLIC CONTRACT (unchanged so S4/S5/S6/S7 keep working)
───────────────
  refine_boundaries(chunks, config) -> List[Dict]
      each output chunk keeps S2's {text, start, end, method} and gains:
        boundary_type      "single" | "start" | "hard" | "protected_structure_boundary"
        jsd_score / metric_score   the (LSTM-refined) shift at the chunk's lead boundary
                                   (names kept for the frontend chart + get_jsd_series)
        boundary_signal    raw semantic-shift d at the lead boundary
        q, diversity_number, shannon_bits, tsallis_bits, entropy_rate
        s3_stats           {initial, final, merged_count, hard_count, soft_count,
                            protected_count, mean_signal, tree_entropy}
        thresholds         {q, target_boundaries, baseline, ...}
  get_jsd_series(chunks)  -> the per-chunk shift series (for charts)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── qentropy engine (the single source of entropy truth) ─────────────────────
from engine import entropy as ent
from engine import tree_entropy as te
from engine.chunking import TreeNode


# ═════════════════════════════════════════════════════════════════════════════
#  TWO ENTROPY FAMILIES USED TO DRIVE THE S3 BOUNDARY COUNT
# ═════════════════════════════════════════════════════════════════════════════
#  S3 turns a probability distribution p = {p_i} over candidate boundary gaps
#  into a target number of boundaries, via an "effective number of gaps" D.
#  Two generalized-entropy families are supported, selected by config
#  "entropy_mode" ∈ {"tsallis", "qlog"}; both reduce to Shannon at q = 1.
#
#  (A) TSALLIS q-ENTROPY  (the default; implemented in engine/entropy.py):
#         S_q^{Tsallis}(p) = ( 1 - Σ_i p_i^q ) / (q - 1)
#      with effective number = Hill number  D_q = ( Σ_i p_i^q )^{1/(1-q)}.
#
#  (B) q-LOGARITHM ENTROPY  (the new formula benchmarked here), defined via a
#      q-deformed logarithm expressed as a series (q-bracket [n]_q = (1-q^n)/(1-q)):
#
#          S_q^{K}(p) = (1 - q) · Σ_i p_i · Σ_{n=1}^{∞}  (1 - p_i)^n / (1 - q^n)          (†)
#
#      As q → 1, (1-q)/(1-q^n) → 1/n, so the inner sum → -ln(p_i) (Mercator
#      series) and S_q^{K} → -Σ_i p_i ln p_i = Shannon entropy.
#
#      The infinite series in (†) converges only slowly when some p_i is small
#      (it needs O(1/p_i) terms).  We therefore evaluate the mathematically
#      IDENTICAL Lambert-type rearrangement, using 1/(1-q^n) = Σ_{k≥0} q^{nk}
#      and closing the inner geometric sum over n:
#
#          S_q^{K}(p) = (1 - q) · Σ_i p_i · Σ_{k=0}^{∞}  (1-p_i) q^k / ( 1 - (1-p_i) q^k )  (‡)
#
#      which converges geometrically at rate |q| — a fixed ~O(100) terms for any
#      p_i, so the truncation depth is an internal numerical constant (NOT a
#      tunable/GA parameter: it only controls approximation accuracy of a fixed
#      quantity, not the model).  Forms (†) and (‡) are verified to agree to 1e-3.
#      q = -1 is excluded (even-n terms 1/(1-q^n) → 1/0), matching the S4 kernel.
# ═════════════════════════════════════════════════════════════════════════════

_QLOG_MIN_Q = -0.999          # q=-1 makes 1/(1-q^n) singular for even n
_QLOG_MAX_TERMS = 600         # safety cap on the Lambert sum (adaptive early-exit below)


def _qlog_L(p_i: float, q: float, tol: float = 1e-9) -> float:
    """Inner q-deformed (-log) series for a single probability p_i, via the
    geometrically-convergent Lambert form of (‡):
        L(p_i) = Σ_{k≥0} (1-p_i) q^k / ( 1 - (1-p_i) q^k ).
    """
    a = 1.0 - p_i                       # (1 - p_i)
    total = 0.0
    qk = 1.0                            # q^0
    for _ in range(_QLOG_MAX_TERMS):
        x = a * qk                      # (1-p_i) q^k
        term = x / (1.0 - x)            # denom finite for |q|<1 since |x|<1
        total += term
        qk *= q
        if abs(term) < tol and abs(qk) < tol:
            break
    return total


def qlog_entropy(p, q: float) -> float:
    """q-logarithm entropy S_q^{K} of Eq.(†)/(‡) above, in NATS.
    Reduces to Shannon (nats) at q≈1; q floored away from -1."""
    p = ent.normalize(p)
    if abs(q - 1.0) < 1e-6:                        # removable singularity → Shannon
        return float(-np.sum(p * np.log(p)))
    qf = max(float(q), _QLOG_MIN_Q)
    s = 0.0
    for pi in p:
        s += float(pi) * _qlog_L(float(pi), qf)
    return float((1.0 - qf) * s)


def qlog_diversity_number(p, q: float) -> float:
    """Effective number of gaps implied by the q-log entropy: the size D of a
    UNIFORM distribution having the same entropy (the "numbers equivalent",
    the general analogue of the Hill number).  Solved by bisection on D∈[1,N]
    since S_q^{K}(uniform_D) is monotonically increasing in D.  Returns D for a
    uniform-N input by construction, and D<N for a skewed input."""
    p = ent.normalize(p)
    N = len(p)
    if N <= 1:
        return 1.0
    if abs(q - 1.0) < 1e-6:                        # Shannon → perplexity exp(H)
        return float(math.exp(-np.sum(p * np.log(p))))
    qf = max(float(q), _QLOG_MIN_Q)
    s_obs = qlog_entropy(p, qf)
    # entropy of a uniform distribution over D categories (continuous in D):
    #   S(uniform_D) = (1-q) · L(1/D)   (each of the D masses 1/D contributes equally)
    def s_uniform(D: float) -> float:
        return (1.0 - qf) * _qlog_L(1.0 / D, qf)
    lo, hi = 1.0, float(N)
    if s_obs <= s_uniform(lo):
        return 1.0
    if s_obs >= s_uniform(hi):
        return float(N)
    for _ in range(60):                            # bisection to ~1e-15 relative
        mid = 0.5 * (lo + hi)
        if s_uniform(mid) < s_obs:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

# Embedding authority (shared with S6 / metrics / GA).  Imported lazily inside
# the functions that need vectors so importing this module never triggers a
# heavyweight model load.


# ─────────────────────────────────────────────────────────────────────────────
# Protected-boundary regex (structural priors — legal / technical headings)
# ─────────────────────────────────────────────────────────────────────────────
_PROTECTED_RE = re.compile(
    r"^[ \t]*(?:"
    r"(?:TITRE|CHAPITRE|SECTION|SOUS-SECTION|PARAGRAPHE|BOOK|PART|CHAPTER)"
    r"\s+(?:[IVXLCDM]+|\d+|PREMIER|PREMIERE|PREMIÈRE|premier|premiere|première)"
    r"|(?:Article|Art\.?|ARTICLE)\s+(?:\d+(?:\s*(?:er|eme|ème|e|bis|ter|quater))?|premier|1er|[IVX]+)"
    r"|§\s*\d+"
    r"|#{1,6}\s+\S+"
    r"|[A-Z][A-Z\s\-]{4,}(?::|$)"
    r")(?:[ \t].*)?$",
    re.MULTILINE | re.IGNORECASE,
)


def _is_protected_boundary(text: str) -> bool:
    """True if a chunk STARTS with a structural heading that must open a chunk."""
    head = (text or "").lstrip()[:200]
    return bool(_PROTECTED_RE.match(head))


def _clip_q(q: float) -> float:
    """Tsallis q restricted to the platform spec range [-1, 1]."""
    try:
        return float(np.clip(float(q), -1.0, 1.0))
    except (TypeError, ValueError):
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic forward LSTM (kept from P1, re-wired for the qentropy features)
# ─────────────────────────────────────────────────────────────────────────────
class _ForwardLSTMCell:
    """Single-layer forward LSTM with fixed Xavier weights (seed → determinism).

    The LSTM is NOT trained; its value is the *gated memory* — a boundary at
    position t is judged in the context of the entropy history of positions
    0…t-1.  We feed it the qentropy-derived feature sequence so that the
    selection of which shifts become boundaries is context-aware rather than
    purely pointwise.
    """

    HIDDEN_MULT = 2  # hidden width = 2 × input width

    def __init__(self, input_dim: int = 5, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.idim = int(input_dim)
        self.hdim = int(input_dim * self.HIDDEN_MULT)
        idim, hdim = self.idim, self.hdim
        scale = np.sqrt(2.0 / (idim + hdim))

        self.Wi = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Ui = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bi = np.zeros(hdim, dtype=np.float32)

        self.Wf = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Uf = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bf = np.ones(hdim, dtype=np.float32)   # forget-gate bias = 1

        self.Wg = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Ug = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bg = np.zeros(hdim, dtype=np.float32)

        self.Wo = rng.randn(hdim, idim).astype(np.float32) * scale
        self.Uo = rng.randn(hdim, hdim).astype(np.float32) * scale
        self.bo = np.zeros(hdim, dtype=np.float32)

        self.Wp = rng.randn(1, hdim).astype(np.float32) * np.sqrt(1.0 / hdim)

        self.h = np.zeros(hdim, dtype=np.float32)
        self.c = np.zeros(hdim, dtype=np.float32)

    def reset(self) -> None:
        self.h[:] = 0.0
        self.c[:] = 0.0

    def step(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        i_g = self._sig(self.Wi @ x + self.Ui @ self.h + self.bi)
        f_g = self._sig(self.Wf @ x + self.Uf @ self.h + self.bf)
        g_g = np.tanh(  self.Wg @ x + self.Ug @ self.h + self.bg)
        o_g = self._sig(self.Wo @ x + self.Uo @ self.h + self.bo)
        self.c = f_g * self.c + i_g * g_g
        self.h = o_g * np.tanh(self.c)
        score = float(self._sig(self.Wp @ self.h)[0])
        return score, self.c.copy()

    @staticmethod
    def _sig(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Unit embedding (shared EmbeddingService, with a deterministic offline fallback)
# ─────────────────────────────────────────────────────────────────────────────
_EMB_CACHE: Dict[str, np.ndarray] = {}


def _embed_units(texts: List[str], backend: Optional[str]) -> np.ndarray:
    """L2-normalised embeddings for the S2 chunk units.

    Uses the shared engine.embeddings.EmbeddingService so S3 lives in the same
    vector space as S6/metrics/GA.  Results are memoised per text so the S7 GA —
    which re-runs S3 many times with the SAME units but different q — never
    re-embeds the same text.  If the embedder fails entirely (no model, offline)
    we fall back to a deterministic hash embedding so the stage still runs.
    """
    miss = [t for t in texts if t not in _EMB_CACHE]
    if miss:
        vecs = None
        try:
            from engine.embeddings import get_embedder
            vecs = get_embedder().embed(miss, backend)
        except Exception:
            vecs = None
        if vecs is None or getattr(vecs, "size", 0) == 0 or len(vecs) != len(miss):
            vecs = np.array([_hash_embed(t) for t in miss], dtype=np.float32)
        for t, v in zip(miss, vecs):
            _EMB_CACHE[t] = np.asarray(v, dtype=np.float32)
    return np.array([_EMB_CACHE[t] for t in texts], dtype=np.float32)


def _hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic bag-of-hashed-tokens embedding, L2-normalised."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"\w+", (text or "").lower()):
        v[hash(tok) % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────
def refine_boundaries(chunks: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    """Keep round(D_q) of S2's candidate boundaries (the qentropy-strongest,
    LSTM-refined, structurally-protected) and merge across the rest."""
    if not chunks:
        return chunks

    q = _clip_q(config.get("q_entropy_param", 1.0))
    K = int(config.get("K", 4))
    min_chunk_tokens = int(config.get("min_chunk_tokens", config.get("n_min", 20)) or 20)
    window = int(config.get("window", 1))
    backend = config.get("embedding_backend")
    use_dq = bool(config.get("use_dq", True))
    use_lstm = bool(config.get("use_lstm", True))            # ablation switch
    use_structure = bool(config.get("use_structure", True))  # ablation switch
    use_qcos_rank = bool(config.get("s3_qcos_rank", False))  # q-cosine gap re-ranking
    # Which generalized entropy drives the boundary count (see the block above):
    #   "tsallis" (default) -> Hill number D_q ;  "qlog" -> q-log numbers-equivalent.
    entropy_mode = str(config.get("entropy_mode", "tsallis")).lower()

    # ── Single-unit document ────────────────────────────────────────────────
    if len(chunks) == 1:
        c = dict(chunks[0])
        c.update({
            "boundary_type": "single",
            "jsd_score": 0.0, "metric_score": 0.0, "boundary_signal": 0.0,
            "q": q, "diversity_number": 1.0,
            "shannon_bits": 0.0, "tsallis_bits": 0.0, "entropy_rate": 0.0,
            "s3_stats": _empty_stats(1, 1),
            "thresholds": {"q": q, "target_boundaries": 0},
        })
        return [c]

    texts = [str(ch.get("text", "")) for ch in chunks]
    N = len(texts)

    # ── 1. Semantic-shift signal between adjacent units ─────────────────────
    emb = _embed_units(texts, backend)
    emb = _windowed(emb, window)
    gap_d = np.array(
        [1.0 - float(np.dot(emb[i], emb[i + 1])) for i in range(N - 1)],
        dtype=np.float64,
    )
    gap_d = np.clip(gap_d, 0.0, None)

    # ── 2. Significant-shift distribution (P2 logic) ────────────────────────
    baseline = float(np.median(gap_d)) if gap_d.size else 0.0
    excess = np.clip(gap_d - baseline, 0.0, None)
    sig_idx = np.where(excess > 1e-9)[0]
    if sig_idx.size >= 1:
        p_cand = ent.normalize(excess[sig_idx])
    else:                                  # no clear structure → ~1 chunk
        sig_idx = np.arange(gap_d.size)
        p_cand = ent.normalize(gap_d + 1e-9)

    # ── 3. Generalised entropy drives the boundary count ────────────────────
    shannon_bits = ent.shannon_entropy(p_cand)
    tsallis_bits = ent.tsallis_entropy(p_cand, q)
    # Effective number of gaps D → target boundary count.  The Tsallis Hill
    # number is the default; entropy_mode="qlog" swaps in the q-log entropy's
    # numbers-equivalent (both defined in the header block of this file).
    if entropy_mode == "qlog":
        d_q = qlog_diversity_number(p_cand, q)
    else:
        d_q = ent.diversity_number(p_cand, q)
    target = int(round(d_q)) if use_dq else int(sig_idx.size)

    # token-feasibility guard: never request more boundaries than min_chunk_tokens
    # can sustain, nor more than the available gaps.
    unit_tokens = [_count_tokens(t) for t in texts]
    prefix = np.concatenate([[0], np.cumsum(unit_tokens)]).astype(np.int64)
    total_tokens = int(prefix[-1])
    hi = max(0, int(np.floor(total_tokens / max(min_chunk_tokens, 1))) - 1)
    target = max(0, min(target, hi, gap_d.size))

    # ── 4. LSTM refinement over the qentropy feature sequence ───────────────
    p_full = ent.normalize(gap_d + 1e-12)                  # mass over ALL gaps
    gmax = float(gap_d.max()) if gap_d.size and gap_d.max() > 0 else 1.0
    emax = float(excess.max()) if excess.size and excess.max() > 0 else 1.0
    pw = np.power(p_full, q)                                # pointwise Tsallis weight
    pwmax = float(pw.max()) if pw.size and pw.max() > 0 else 1.0
    struct_flag = np.array(
        [1.0 if _is_protected_boundary(texts[i + 1]) else 0.0 for i in range(N - 1)],
        dtype=np.float64,
    )
    if not use_structure:                    # ablation: ignore structural priors
        struct_flag[:] = 0.0

    lstm = _ForwardLSTMCell(input_dim=5, seed=42)
    lstm.reset()
    lstm_scores = np.zeros(gap_d.size, dtype=np.float64)
    for i in range(gap_d.size):
        x = np.array([
            gap_d[i] / gmax,            # normalised shift
            p_full[i] / (pwmax if pwmax else 1.0),  # mass (scaled)
            excess[i] / emax,           # excess over baseline coherence
            pw[i] / pwmax,              # pointwise Tsallis weight p_i^q
            struct_flag[i],             # structural prior
        ], dtype=np.float32)
        s, _ = lstm.step(x)
        lstm_scores[i] = s

    # Boundary-strength term used to RANK gaps.  Optionally reshape the raw shift
    # through the Fitouhi–Bouzeffour q-cosine (base q²).  Unlike S4's single-
    # threshold gate (a monotone reparam of cosine), here the deformed shift is
    # BLENDED with the LSTM score, so the nonlinear reshape can reorder which gaps
    # land in the top round(D_q) — a genuine re-ranking.  q=±1 ⇒ raw shift (sanity).
    if use_qcos_rank:
        from .s4_boundary import _q_cosine_similarity  # shared q-cosine kernel
        cos_seq = np.clip(1.0 - gap_d, -1.0, 1.0)               # adjacent-unit cosine
        q_shift = np.array(
            [1.0 - _q_cosine_similarity(float(c), q) for c in cos_seq], dtype=np.float64)
        smax = float(q_shift.max()) if q_shift.size and q_shift.max() > 0 else 1.0
        shift_term = q_shift / smax
    else:
        shift_term = gap_d / gmax

    # blended, context-aware salience used to RANK boundaries.  Ablating the LSTM
    # falls back to the (possibly q-reshaped) qentropy shift signal alone.
    if use_lstm:
        combined = np.clip(0.65 * shift_term + 0.35 * lstm_scores, 0.0, 1.0)
    else:
        combined = np.clip(shift_term, 0.0, 1.0)

    # ── 5. Select the `target` strongest gaps (structural priors forced) ────
    priority = {int(i) for i in np.where(struct_flag > 0.5)[0]} if use_structure else set()
    boundaries = _select_boundaries(
        np.arange(gap_d.size), combined, target, N, prefix, min_chunk_tokens, priority
    )
    boundary_set = set(boundaries)

    # ── 6. Tree entropy fingerprint over the chosen segmentation ────────────
    tree = _build_unit_tree(unit_tokens, combined, K)
    tree_ent = te.tree_entropy(tree, K, total_tokens=total_tokens, q=q)
    entropy_rate = float(tree_ent.get("entropy_rate", 0.0))

    # ── 7. Merge across non-kept gaps → final chunk dicts ───────────────────
    out: List[Dict] = []
    merged_count = 0
    hard_count = 0
    protected_count = 0

    seg_start = 0
    for g in range(N):                      # g indexes units; cut AFTER unit g
        is_cut = (g in boundary_set) or (g == N - 1)
        if not is_cut:
            continue
        units = list(range(seg_start, g + 1))
        merged_count += len(units) - 1
        text = "\n\n".join(texts[u] for u in units).strip()
        base = dict(chunks[seg_start])
        # lead-boundary type / signal (boundary that OPENS this segment)
        lead = seg_start - 1
        if seg_start == 0:
            btype = "start"
            sig = 0.0
        elif lead in priority:
            btype = "protected_structure_boundary"
            protected_count += 1
            sig = float(combined[lead])
        else:
            btype = "hard"
            hard_count += 1
            sig = float(combined[lead])
        base.update({
            "text": text,
            "start": chunks[seg_start].get("start", 0),
            "end": chunks[g].get("end", 0),
            "method": chunks[seg_start].get("method", "qentropy"),
            "boundary_type": btype,
            "jsd_score": round(sig, 4),
            "metric_score": round(sig, 4),
            "boundary_signal": round(float(gap_d[lead]) if lead >= 0 else 0.0, 4),
            "q": q,
            "diversity_number": round(float(d_q), 3),
            "shannon_bits": round(float(shannon_bits), 4),
            "tsallis_bits": round(float(tsallis_bits), 4),
            "entropy_rate": round(entropy_rate, 4),
        })
        out.append(base)
        seg_start = g + 1

    if not out:                              # degenerate guard → single chunk
        c = dict(chunks[0])
        c["text"] = "\n\n".join(texts).strip()
        c["end"] = chunks[-1].get("end", 0)
        c.update({"boundary_type": "single", "jsd_score": 0.0, "metric_score": 0.0,
                  "boundary_signal": 0.0, "q": q, "diversity_number": float(d_q),
                  "shannon_bits": float(shannon_bits), "tsallis_bits": float(tsallis_bits),
                  "entropy_rate": entropy_rate})
        out = [c]

    if len(out) == 1:
        out[0]["boundary_type"] = "single"

    stats = {
        "initial_chunks": N,
        "final_chunks": len(out),
        "merged_count": merged_count,
        "hard_count": hard_count,
        "soft_count": 0,                     # S3 emits hard/merge only; S4 may merge further
        "protected_count": protected_count,
        "mean_signal": round(float(np.mean(combined)) if combined.size else 0.0, 4),
        "target_boundaries": int(target),
        "diversity_number": round(float(d_q), 3),
        "shannon_bits": round(float(shannon_bits), 4),
        "tsallis_bits": round(float(tsallis_bits), 4),
        "tree_entropy": tree_ent,
    }
    thresholds = {
        "q": q, "K": K, "target_boundaries": int(target),
        "baseline": round(baseline, 4), "n_significant": int(sig_idx.size),
    }
    for c in out:
        c["s3_stats"] = stats
        c["thresholds"] = thresholds
    return out


def get_jsd_series(chunks: List[Dict]) -> List[float]:
    """Per-chunk boundary-shift series for the frontend chart."""
    return [float(c.get("metric_score", c.get("jsd_score", 0.0))) for c in chunks]


# Back-compat alias (the new name reflects that this is no longer JSD-specific).
get_boundary_signal = get_jsd_series


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _count_tokens(text: str) -> int:
    """Token count via the shared tokenizer, with a whitespace fallback."""
    try:
        from engine.embeddings import get_embedder
        return max(1, get_embedder().count_tokens(text))
    except Exception:
        return max(1, len((text or "").split()))


def _windowed(emb: np.ndarray, window: int) -> np.ndarray:
    """Neighbour-averaged, re-normalised unit embeddings (smoother shift signal)."""
    a = np.asarray(emb, dtype=np.float64)
    n = len(a)
    if window <= 0 or n <= 2:
        return _l2_rows(a)
    out = np.zeros_like(a)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        out[i] = a[lo:hi].mean(axis=0)
    return _l2_rows(out)


def _l2_rows(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


def _select_boundaries(
    cand: np.ndarray,
    signal: np.ndarray,
    max_boundaries: int,
    n: int,
    prefix: np.ndarray,
    min_tok: int,
    priority: Optional[set] = None,
) -> List[int]:
    """Greedily accept boundaries by descending salience while keeping every
    resulting segment >= min_tok tokens.  Structural `priority` gaps are placed
    first with a relaxed floor.  Gap g means: split AFTER unit g."""
    def seg_tokens(s_idx: int, e_idx: int) -> int:
        return int(prefix[e_idx + 1] - prefix[s_idx])

    chosen: List[int] = []

    def ok(g: int, floor: int) -> bool:
        left_b = max([b for b in chosen if b < g], default=-1)
        right_b = min([b for b in chosen if b > g], default=n - 1)
        return seg_tokens(left_b + 1, g) >= floor and seg_tokens(g + 1, right_b) >= floor

    relaxed = max(8, min_tok // 2)
    for g in sorted(priority or set()):
        g = int(g)
        if 0 <= g < n - 1 and g not in chosen and ok(g, relaxed):
            chosen.append(g)
            chosen.sort()

    if max_boundaries > 0 and cand.size:
        for g in cand[np.argsort(-signal[cand], kind="stable")]:
            if len(chosen) >= max_boundaries:
                break
            g = int(g)
            if g in chosen:
                continue
            if ok(g, min_tok):
                chosen.append(g)
                chosen.sort()
    return sorted(chosen)


def _build_unit_tree(unit_tokens: List[int], signal: np.ndarray, K: int) -> TreeNode:
    """Divisive K-ary tree over the S2 units, splitting strongest-gap-first, with
    each node carrying its token count so engine.tree_entropy can sum log Z_K."""
    n = len(unit_tokens)
    cum = np.concatenate([[0], np.cumsum(unit_tokens)]).astype(np.int64)

    def span_tokens(s: int, e: int) -> int:
        return int(cum[e + 1] - cum[s])

    def split(s: int, e: int, depth: int) -> TreeNode:
        node = TreeNode(s, e, "", tokens=span_tokens(s, e))
        if e <= s or depth > 6:
            return node
        gaps = list(range(s, e))
        if not gaps:
            return node
        gaps_sorted = sorted(gaps, key=lambda g: signal[g], reverse=True)
        top = float(signal[gaps_sorted[0]])
        chosen = [gaps_sorted[0]]
        for g in gaps_sorted[1:max(1, K - 1)]:
            if signal[g] >= 0.45 * top and signal[g] > 0.12:
                chosen.append(g)
        chosen = sorted(chosen)
        node.strength = top
        prev = s
        for g in chosen:
            node.children.append(split(prev, g, depth + 1))
            prev = g + 1
        node.children.append(split(prev, e, depth + 1))
        return node

    return split(0, n - 1, 0)


def _empty_stats(initial: int, final: int) -> Dict[str, Any]:
    return {
        "initial_chunks": initial,
        "final_chunks": final,
        "merged_count": 0,
        "hard_count": 0,
        "soft_count": 0,
        "protected_count": 0,
        "mean_signal": 0.0,
        "target_boundaries": 0,
        "diversity_number": 1.0,
        "shannon_bits": 0.0,
        "tsallis_bits": 0.0,
        "tree_entropy": {"shannon_nats": 0.0, "entropy_rate": 0.0,
                         "tsallis_nats": 0.0, "n_internal": 0, "K": 0},
    }
