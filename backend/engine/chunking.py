"""Entropy-driven semantic chunking.

Faithful-but-practical extension of Zhong et al. (arXiv:2602.13194):

  * In the paper, an LLM produces a semantic tree and entropy only *measures*
    the tree ensemble -- so Shannon vs Tsallis would yield identical chunks.
  * Here we let the generalized entropy *drive* the chunking granularity, so
    Shannon (q=1) and Tsallis (q != 1) produce genuinely different chunk sets
    that can be benchmarked against each other.

Pipeline (per Fig. 1 of the paper, adapted to be embedding-driven & free):

  1. Segment the document into sentences.
  2. Embed each sentence.
  3. Semantic-shift signal d_i at each gap = 1 - cos(s_i, s_{i+1}).
  4. Normalise the d_i into a boundary distribution p.
  5. Diversity number D_q = (sum p_i^q)^(1/(1-q)) sets the target number of
     chunks.  q<1 amplifies weak/rare boundaries (finer chunks); q>1 suppresses
     them (coarser); q=1 == Shannon perplexity (the paper's baseline).
  6. Open the strongest (target-1) gaps as boundaries, respecting min/max
     chunk token budgets, then build the flat chunk list.
  7. Build a divisive dendrogram (boundaries opened strongest-first) as the
     "semantic tree T" for visualisation, capped at branching factor K.

Returns rich per-chunk info (text, tokens, internal coherence) plus the
document-level entropy fingerprint (Shannon H, Tsallis S_q, D_q).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import entropy as ent
from . import tree_entropy as te
from .embeddings import EmbeddingService, cosine_matrix

_LN2 = float(np.log(2.0))


def _span_preview(sentences: List[str], s: int, e: int, limit: int = 63) -> str:
    """Short label for a span — reads only the first few sentences (NOT the whole
    span), so building a tree over a large document stays O(n), not O(n²)."""
    out: List[str] = []
    ln = 0
    for i in range(s, min(e, s + 40) + 1):
        out.append(sentences[i])
        ln += len(sentences[i]) + 1
        if ln > limit + 8:
            break
    txt = " ".join(out)
    return (txt[:60] + "...") if len(txt) > limit else txt

# Structure-aware segmentation lives in engine/structure.py. These names are
# re-exported here so existing imports (and call-sites below) keep working.
from .structure import (  # noqa: E402
    detect_language,
    heading_gaps as structural_boundary_gaps,
    section_paths,
    segment,
    split_sentences,
)


def chunk_id(text: str) -> str:
    """Stable content-hash id (sha256 prefix) — safe for vector-store upserts:
    the same text always maps to the same id, so re-ingesting a document
    overwrites instead of duplicating."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    index: int
    text: str
    tokens: int
    sent_start: int
    sent_end: int  # inclusive
    coherence: float  # mean adjacent cosine within the chunk (1.0 if single sentence)
    section_path: str = ""  # heading breadcrumb, e.g. "ITEM 7 > Liquidity"

    @property
    def id(self) -> str:
        return chunk_id(self.text)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "index": self.index,
            "text": self.text,
            "tokens": self.tokens,
            "sent_start": self.sent_start,
            "sent_end": self.sent_end,
            "coherence": round(float(self.coherence), 4),
            "n_sentences": self.sent_end - self.sent_start + 1,
            "section_path": self.section_path,
        }

    def to_export_dict(self) -> dict:
        """Minimal record for JSONL export / vector-store ingestion."""
        return {
            "id": self.id,
            "index": self.index,
            "text": self.text,
            "tokens": self.tokens,
            "section_path": self.section_path,
            "sent_start": self.sent_start,
            "sent_end": self.sent_end,
        }


@dataclass
class TreeNode:
    sent_start: int
    sent_end: int
    label: str
    strength: float = 0.0  # dissimilarity of the gap that created this split
    tokens: int = 0        # token count of this span (for paper-faithful entropy)
    children: List["TreeNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sent_start": self.sent_start,
            "sent_end": self.sent_end,
            "label": self.label,
            "strength": round(float(self.strength), 4),
            "tokens": int(self.tokens),
            "n_sentences": self.sent_end - self.sent_start + 1,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class ChunkingResult:
    method: str
    q: float
    chunks: List[Chunk]
    tree: Optional[TreeNode]
    boundary_signal: List[float]  # d_i per gap
    boundary_distribution: List[float]  # normalised p_i per gap
    boundaries: List[int]  # gap indices opened as chunk boundaries
    shannon_bits: float
    tsallis_bits: float
    diversity_number: float
    chunking_time_ms: float
    n_sentences: int
    tree_entropy: Optional[dict] = None  # paper-faithful tree entropy (recursive method)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "q": self.q,
            "chunks": [c.to_dict() for c in self.chunks],
            "tree": self.tree.to_dict() if self.tree else None,
            "boundary_signal": [round(float(x), 4) for x in self.boundary_signal],
            "boundary_distribution": [round(float(x), 5) for x in self.boundary_distribution],
            "boundaries": list(self.boundaries),
            "shannon_bits": round(self.shannon_bits, 4),
            "tsallis_bits": round(self.tsallis_bits, 4),
            "diversity_number": round(self.diversity_number, 3),
            "chunking_time_ms": round(self.chunking_time_ms, 2),
            "n_sentences": self.n_sentences,
            "n_chunks": len(self.chunks),
            "avg_chunk_tokens": round(
                float(np.mean([c.tokens for c in self.chunks])) if self.chunks else 0.0, 1
            ),
            "tree_entropy": self.tree_entropy,
        }


@dataclass
class ChunkParams:
    q: float = 1.0
    K: int = 4               # max branching factor (paper's single free parameter)
    min_chunk_tokens: int = 20
    max_chunk_tokens: int = 320
    window: int = 1          # neighbour-averaging window for the distance signal
    overlap: int = 0         # sentences each chunk borrows from its neighbours (RAG safety)
    # --- ablation switches (default = full method; used by scripts/ablation.py) ---
    use_dq: bool = True        # False: open ALL gaps above tau (plain thresholding, no D_q)
    use_structure: bool = True  # False: ignore heading priors / hard structural boundaries


def _apply_overlap(
    chunks: List["Chunk"],
    sentences: List[str],
    overlap: int,
    embedder: "EmbeddingService",
) -> List["Chunk"]:
    """Extend each chunk's TEXT to include `overlap` sentences from the previous
    and next chunk, so an answer sitting on a boundary isn't lost at retrieval.

    Sentence spans (sent_start/sent_end) are left untouched — they still mark the
    chunk's "home" range for the tree/visualisation; only the retrievable text
    and token count grow.
    """
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    n = len(sentences)
    out: List[Chunk] = []
    for c in chunks:
        s = max(0, c.sent_start - overlap)
        e = min(n - 1, c.sent_end + overlap)
        text = " ".join(sentences[s : e + 1]).strip()
        out.append(Chunk(c.index, text, embedder.count_tokens(text),
                         c.sent_start, c.sent_end, c.coherence, c.section_path))
    return out


def _attach_section_paths(chunks: List["Chunk"], sentences: List[str]) -> List["Chunk"]:
    """Tag each chunk with the heading breadcrumb of its first sentence
    (e.g. "ITEM 7 > Liquidity") — high-impact retrieval metadata that the
    structure detector already knows."""
    if not chunks or not sentences:
        return chunks
    paths = section_paths(sentences)
    for c in chunks:
        if 0 <= c.sent_start < len(paths):
            c.section_path = paths[c.sent_start]
    return chunks


def _build_chunks(
    sentences: List[str],
    boundaries: List[int],
    sims: np.ndarray,
    embedder: EmbeddingService,
) -> List[Chunk]:
    """boundaries: sorted gap indices i meaning a split between sent i and i+1."""
    cut_points = sorted(set(boundaries))
    spans: List[tuple] = []
    start = 0
    for b in cut_points:
        spans.append((start, b))  # inclusive end = b
        start = b + 1
    spans.append((start, len(sentences) - 1))

    chunks: List[Chunk] = []
    for (s, e) in spans:
        if s > e:
            continue
        text = " ".join(sentences[s : e + 1]).strip()
        if not text:
            continue
        if e > s:
            coh = float(np.mean([sims[i] for i in range(s, e)]))
        else:
            coh = 1.0
        chunks.append(
            Chunk(
                index=len(chunks),
                text=text,
                tokens=embedder.count_tokens(text),
                sent_start=s,
                sent_end=e,
                coherence=coh,
            )
        )
    return chunks


def _select_boundaries(
    cand: np.ndarray,
    gap_d: np.ndarray,
    max_boundaries: int,
    n: int,
    prefix: np.ndarray,
    min_tok: int,
    priority: "set | None" = None,
) -> List[int]:
    """Greedily accept boundaries keeping every chunk >= min_tok tokens.

    Structural `priority` gaps (article/section headings) are placed first with a
    relaxed floor so legal articles stay intact; the q-driven count then fills the
    rest with the strongest semantic-shift gaps. Boundary g => split after sent g.
    """
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
        # kind="stable": equal gaps tie-break by position on every numpy version,
        # part of the determinism contract.
        for g in cand[np.argsort(-gap_d[cand], kind="stable")]:
            if len(chosen) >= max_boundaries:
                break
            g = int(g)
            if g in chosen:
                continue
            if ok(g, min_tok):
                chosen.append(g)
                chosen.sort()
    return sorted(chosen)


def _split_oversized(
    chunks: List[Chunk],
    sentences: List[str],
    sims: np.ndarray,
    embedder: EmbeddingService,
    max_tok: int,
) -> List[Chunk]:
    guard = 0
    i = 0
    while i < len(chunks) and guard < 500:
        guard += 1
        c = chunks[i]
        if c.tokens <= max_tok or c.sent_end == c.sent_start:
            i += 1
            continue
        gap_range = range(c.sent_start, c.sent_end)
        best = max(gap_range, key=lambda g: 1.0 - sims[g])
        left = _make_chunk(c.sent_start, best, sentences, sims, embedder)
        right = _make_chunk(best + 1, c.sent_end, sentences, sims, embedder)
        chunks[i : i + 1] = [left, right]
    for idx, c in enumerate(chunks):
        c.index = idx
    return chunks


def _make_chunk(s, e, sentences, sims, embedder) -> Chunk:
    text = " ".join(sentences[s : e + 1]).strip()
    coh = float(np.mean([sims[i] for i in range(s, e)])) if e > s else 1.0
    return Chunk(0, text, embedder.count_tokens(text), s, e, coh)


def _build_tree(
    sentences: List[str],
    gap_d: np.ndarray,
    K: int,
) -> Optional[TreeNode]:
    """Divisive dendrogram: recursively split a span at its strongest gaps,
    opening up to K-1 splits per level (branching factor <= K)."""
    n = len(sentences)
    if n == 0:
        return None

    def preview(s, e):
        return _span_preview(sentences, s, e)

    def split(s: int, e: int, depth: int) -> TreeNode:
        node = TreeNode(s, e, preview(s, e))
        if e <= s or depth > 6:
            return node
        gaps = list(range(s, e))  # candidate gap indices within [s, e-1]
        if not gaps:
            return node
        gaps_sorted = sorted(gaps, key=lambda g: gap_d[g], reverse=True)
        # Number of splits at this level: stop early if the strongest remaining
        # gap is weak relative to the top gap (coherent region).
        top = gap_d[gaps_sorted[0]]
        chosen = [gaps_sorted[0]]
        for g in gaps_sorted[1 : K - 1]:
            if gap_d[g] >= 0.45 * top and gap_d[g] > 0.12:
                chosen.append(g)
        chosen = sorted(chosen)
        node.strength = float(top)
        prev = s
        for g in chosen:
            child = split(prev, g, depth + 1)
            node.children.append(child)
            prev = g + 1
        node.children.append(split(prev, e, depth + 1))
        return node

    return split(0, n - 1, 0)


# --------------------------------------------------------------- shared signal
def _l2_rows(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


def _windowed_embeddings(sent_emb: np.ndarray, window: int) -> np.ndarray:
    """Neighbour-averaged sentence embeddings (Kamradt's buffer trick).

    Single-sentence embeddings are noisy; averaging each sentence with its
    `window` neighbours on each side gives a smoother, more reliable
    semantic-shift signal for boundary detection. window=0 disables it.
    """
    emb = np.asarray(sent_emb, dtype=np.float64)
    n = len(emb)
    if window <= 0 or n <= 2:
        return emb
    out = np.zeros_like(emb)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        out[i] = emb[lo:hi].mean(axis=0)
    return _l2_rows(out)


def _gap_distances(sent_emb: np.ndarray, window: int = 1) -> np.ndarray:
    """Semantic-shift signal d_i = 1 - cos(s_i, s_{i+1}) on windowed embeddings.

    This is the single signal shared by BOTH the classic semantic-percentile
    chunker and the entropy-enhanced chunker, so the only thing that differs
    between the two methods is how boundaries are *selected* from it.
    """
    emb = _windowed_embeddings(sent_emb, window)
    n = len(emb)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    sims = np.array([float(np.dot(emb[i], emb[i + 1])) for i in range(n - 1)],
                    dtype=np.float64)
    sims = np.clip(sims, -1.0, 1.0)
    return np.clip(1.0 - sims, 0.0, None)


def chunk_document(
    text: str,
    params: ChunkParams,
    embedder: "EmbeddingService",
    sentences: Optional[List[str]] = None,
    sent_emb: Optional[np.ndarray] = None,
    build_tree: bool = True,
    backend: Optional[str] = None,
) -> ChunkingResult:
    t0 = time.perf_counter()
    if sentences is None:
        sentences = split_sentences(text)
    n = len(sentences)

    if sent_emb is None:
        sent_emb = embedder.embed(sentences, backend)

    if n <= 1:
        chunks = [
            Chunk(0, sentences[0] if sentences else text,
                  embedder.count_tokens(sentences[0] if sentences else text),
                  0, max(0, n - 1), 1.0)
        ]
        dt = (time.perf_counter() - t0) * 1000.0
        return ChunkingResult(
            method=("Shannon" if abs(params.q - 1) < 1e-9 else "Tsallis"),
            q=params.q, chunks=chunks, tree=None, boundary_signal=[],
            boundary_distribution=[], boundaries=[], shannon_bits=0.0,
            tsallis_bits=0.0, diversity_number=1.0, chunking_time_ms=dt,
            n_sentences=n,
        )

    # Windowed semantic-shift signal (shared with the classic baseline).
    gap_d = _gap_distances(sent_emb, params.window)
    sims = 1.0 - gap_d  # similarity proxy used for coherence / oversize splits

    # SIGNIFICANT boundaries = gaps whose semantic shift exceeds the document's
    # own baseline coherence (its median gap).  Restricting the distribution to
    # these peaks is what makes the diversity number meaningful: a near-uniform
    # signal (no real structure) has few significant peaks, while a document
    # with several sharp topic shifts has exactly that many.
    baseline = float(np.median(gap_d))
    excess = np.clip(gap_d - baseline, 0.0, None)
    sig_idx = np.where(excess > 1e-9)[0]
    if sig_idx.size >= 1:
        p_cand = ent.normalize(excess[sig_idx])
    else:
        # No clear structure: keep entropy well-defined but expect ~1 chunk.
        sig_idx = np.arange(len(gap_d))
        p_cand = ent.normalize(gap_d + 1e-9)

    shannon_bits = ent.shannon_entropy(p_cand)
    tsallis_bits = ent.tsallis_entropy(p_cand, params.q)
    d_q = ent.diversity_number(p_cand, params.q)  # effective # of real boundaries

    # The diversity number IS the boundary count: the effective number of
    # distinct semantic shifts at granularity q.  D_q is non-increasing in q,
    # so q<1 counts weak/rare boundaries too (finer chunks), q>1 keeps only the
    # dominant ones (coarser), and q=1 == Shannon perplexity exp(H).  No remap
    # onto token budgets -- those act only as guards below.
    target_boundaries = int(round(d_q)) if params.use_dq else int(sig_idx.size)

    sent_tokens = [embedder.count_tokens(s) for s in sentences]
    prefix = np.concatenate([[0], np.cumsum(sent_tokens)]).astype(np.int64)
    total_tokens = int(prefix[-1])
    # Token-feasibility guard: never request more boundaries than min_chunk_tokens
    # can sustain (a chunk must hold >= min tokens), nor more than the gaps allow.
    hi = max(0, int(np.floor(total_tokens / max(params.min_chunk_tokens, 1))) - 1)
    target_boundaries = max(0, min(target_boundaries, hi, len(gap_d)))

    # Structure-aware priors: align boundaries to Article/Section/Chapitre
    # headings so legal/technical articles stay intact (these are opened first).
    struct_gaps = structural_boundary_gaps(sentences) if params.use_structure else set()

    # Open the `target_boundaries` strongest gaps (drawn from ALL gaps, not just
    # the above-baseline ones, so a fine setting can keep subdividing), never
    # creating a chunk below min_chunk_tokens.  Boundaries always land on the
    # genuine semantic-shift peaks; q only decides HOW MANY of them to open.
    all_gaps = np.arange(len(gap_d))
    boundaries = _select_boundaries(
        all_gaps, gap_d, target_boundaries, n, prefix, params.min_chunk_tokens,
        priority=struct_gaps,
    )

    chunks = _build_chunks(sentences, boundaries, sims, embedder)
    # Only split pathologically oversized chunks; separation already enforces min.
    chunks = _split_oversized(
        chunks, sentences, sims, embedder, params.max_chunk_tokens
    )
    chunks = _attach_section_paths(chunks, sentences)
    chunks = _apply_overlap(chunks, sentences, params.overlap, embedder)

    # For the tree view, boost structural gaps so the divisive tree splits there.
    gap_tree = gap_d.copy()
    if struct_gaps:
        boost = max(float(np.percentile(gap_d, 95)), 1.0)
        for g in struct_gaps:
            gap_tree[g] = max(gap_tree[g], boost)
    tree = _build_tree(sentences, gap_tree, params.K) if build_tree else None

    dt = (time.perf_counter() - t0) * 1000.0
    return ChunkingResult(
        method=("Shannon" if abs(params.q - 1) < 1e-9 else "Tsallis"),
        q=params.q,
        chunks=chunks,
        tree=tree,
        boundary_signal=list(gap_d),
        boundary_distribution=list(ent.normalize(gap_d)),
        boundaries=boundaries,
        shannon_bits=shannon_bits,
        tsallis_bits=tsallis_bits,
        diversity_number=d_q,
        chunking_time_ms=dt,
        n_sentences=n,
    )


def fixed_size_chunks(
    text: str,
    embedder: EmbeddingService,
    target_tokens: int = 200,
) -> ChunkingResult:
    """Naive fixed-size baseline (sentence-packing to ~target_tokens)."""
    t0 = time.perf_counter()
    sentences = split_sentences(text)
    chunks: List[Chunk] = []
    buf: List[str] = []
    start = 0
    cur = 0
    for i, s in enumerate(sentences):
        st = embedder.count_tokens(s)
        if buf and cur + st > target_tokens:
            txt = " ".join(buf)
            chunks.append(Chunk(len(chunks), txt, embedder.count_tokens(txt),
                                start, i - 1, 1.0))
            buf, cur, start = [], 0, i
        buf.append(s)
        cur += st
    if buf:
        txt = " ".join(buf)
        chunks.append(Chunk(len(chunks), txt, embedder.count_tokens(txt),
                            start, len(sentences) - 1, 1.0))
    dt = (time.perf_counter() - t0) * 1000.0
    return ChunkingResult(
        method="Fixed-size", q=None, chunks=chunks, tree=None,
        boundary_signal=[], boundary_distribution=[], boundaries=[],
        shannon_bits=0.0, tsallis_bits=0.0, diversity_number=float(len(chunks)),
        chunking_time_ms=dt, n_sentences=len(sentences),
    )


def rcts_chunks(
    text: str,
    embedder: EmbeddingService,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
) -> ChunkingResult:
    """RecursiveCharacterTextSplitter baseline (dependency-free reimplementation
    of the LangChain default, measured in tokens).

    This is what real-world RAG pipelines actually use by default: split on
    paragraph -> line -> sentence -> word boundaries until pieces fit, then pack
    them into ~chunk_tokens windows with overlap_tokens of carryover.  Any
    'smarter' chunker has to beat THIS, not just the fixed-size strawman.
    """
    t0 = time.perf_counter()
    seps = ["\n\n", "\n", ". ", " "]
    tok = embedder.count_tokens

    def split(txt: str, si: int) -> List[str]:
        if tok(txt) <= chunk_tokens or si >= len(seps):
            return [txt]
        sep = seps[si]
        parts = txt.split(sep)
        if len(parts) == 1:
            return split(txt, si + 1)
        pieces: List[str] = []
        for j, p in enumerate(parts):
            seg = p + (sep if j < len(parts) - 1 else "")
            if not seg:
                continue
            if tok(seg) > chunk_tokens:
                pieces.extend(split(seg, si + 1))
            else:
                pieces.append(seg)
        return pieces or [txt]

    pieces = [p for p in split(text, 0) if p.strip()]

    texts: List[str] = []
    buf: List[str] = []
    cur = 0
    for p in pieces:
        pt = tok(p)
        if buf and cur + pt > chunk_tokens:
            texts.append("".join(buf).strip())
            keep: List[str] = []
            kept = 0
            for prev in reversed(buf):       # carry trailing pieces as overlap
                qt = tok(prev)
                if kept + qt > overlap_tokens:
                    break
                keep.insert(0, prev)
                kept += qt
            buf, cur = keep, kept
        buf.append(p)
        cur += pt
    if buf:
        tail = "".join(buf).strip()
        if tail:
            texts.append(tail)

    chunks = [Chunk(i, t, tok(t), 0, 0, 1.0) for i, t in enumerate(texts)]
    dt = (time.perf_counter() - t0) * 1000.0
    return ChunkingResult(
        method=f"RCTS {chunk_tokens}/{overlap_tokens}", q=None, chunks=chunks,
        tree=None, boundary_signal=[], boundary_distribution=[], boundaries=[],
        shannon_bits=0.0, tsallis_bits=0.0, diversity_number=float(len(chunks)),
        chunking_time_ms=dt, n_sentences=len(split_sentences(text)),
    )


def semantic_percentile_chunks(
    text: str,
    params: ChunkParams,
    embedder: "EmbeddingService",
    sentences: Optional[List[str]] = None,
    sent_emb: Optional[np.ndarray] = None,
    percentile: float = 95.0,
    backend: Optional[str] = None,
) -> ChunkingResult:
    """Classic semantic chunking (Kamradt / LangChain SemanticChunker).

    The de-facto baseline: compute the windowed semantic-shift signal, then open
    a boundary at every gap whose distance is at or above a FIXED percentile
    threshold (default 95th).  Its only knob is that arbitrary percentile --
    which is exactly what the entropy-enhanced chunker replaces with a
    principled, document-adaptive boundary count.

    Uses the SAME distance signal and the SAME min/max token guards as
    `chunk_document`, so the head-to-head benchmark isolates the effect of how
    boundaries are chosen (fixed percentile vs. entropy-derived count).
    """
    t0 = time.perf_counter()
    if sentences is None:
        sentences = split_sentences(text)
    n = len(sentences)
    if sent_emb is None:
        sent_emb = embedder.embed(sentences, backend)

    label = f"Semantic P{percentile:g} (classic)"
    if n <= 1:
        only = sentences[0] if sentences else text
        chunks = [Chunk(0, only, embedder.count_tokens(only), 0, max(0, n - 1), 1.0)]
        dt = (time.perf_counter() - t0) * 1000.0
        return ChunkingResult(
            method=label, q=None, chunks=chunks, tree=None, boundary_signal=[],
            boundary_distribution=[], boundaries=[], shannon_bits=0.0,
            tsallis_bits=0.0, diversity_number=1.0, chunking_time_ms=dt,
            n_sentences=n,
        )

    gap_d = _gap_distances(sent_emb, params.window)
    sims = 1.0 - gap_d

    # Classic breakpoint rule: cut wherever the shift is at/above the percentile.
    thr = float(np.percentile(gap_d, percentile))
    cand = np.where(gap_d >= thr)[0]

    sent_tokens = [embedder.count_tokens(s) for s in sentences]
    prefix = np.concatenate([[0], np.cumsum(sent_tokens)]).astype(np.int64)

    # Open all above-threshold gaps (strongest first), respecting min tokens.
    # Heading breaks are opened too (structure-aware), so the head-to-head with
    # the entropy method isolates the percentile-vs-entropy choice, not structure.
    boundaries = _select_boundaries(
        cand, gap_d, int(cand.size), n, prefix, params.min_chunk_tokens,
        priority=structural_boundary_gaps(sentences) if params.use_structure else set(),
    )
    chunks = _build_chunks(sentences, boundaries, sims, embedder)
    chunks = _split_oversized(chunks, sentences, sims, embedder, params.max_chunk_tokens)
    chunks = _attach_section_paths(chunks, sentences)
    chunks = _apply_overlap(chunks, sentences, params.overlap, embedder)

    # Entropy fingerprint (Shannon) purely for display/comparison.
    p_all = ent.normalize(gap_d + 1e-9)
    tree = _build_tree(sentences, gap_d, params.K)
    dt = (time.perf_counter() - t0) * 1000.0
    return ChunkingResult(
        method=label, q=None, chunks=chunks, tree=tree,
        boundary_signal=list(gap_d),
        boundary_distribution=list(p_all),
        boundaries=boundaries,
        shannon_bits=ent.shannon_entropy(p_all),
        tsallis_bits=ent.shannon_entropy(p_all),
        diversity_number=float(len(chunks)),
        chunking_time_ms=dt, n_sentences=n,
    )


# ============================================================================
# Entropy-Guided Recursive Semantic Chunking (EG-RSC)  -- the primary method.
#
# Faithful to Zhong et al. (arXiv:2602.13194): recursively segment each span
# into AT MOST K semantically coherent children (K = max branching factor, the
# paper's single parameter, empirically 2-6), building a semantic tree.  Our
# extension: the number of cuts opened at each node is the *local* Tsallis
# q-diversity number of that node's significant gaps, capped at K-1.  This makes
# the granularity decision LOCAL + HIERARCHICAL (adapts per region) rather than
# a single GLOBAL percentile threshold -- the mechanism by which entropy beats
# the classic baseline.  The leaves, in reading order, are the final chunks.
# ============================================================================
def recursive_chunk_document(
    text: str,
    params: ChunkParams,
    embedder: "EmbeddingService",
    sentences: Optional[List[str]] = None,
    sent_emb: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> ChunkingResult:
    t0 = time.perf_counter()
    if sentences is None:
        sentences = split_sentences(text)
    n = len(sentences)
    if sent_emb is None:
        sent_emb = embedder.embed(sentences, backend)

    is_shannon = abs(params.q - 1.0) < 1e-9
    label = "Recursive Shannon (q=1)" if is_shannon else f"Recursive Tsallis q={params.q:g}"
    K = max(2, int(params.K))

    if n <= 1:
        only = sentences[0] if sentences else text
        tok = embedder.count_tokens(only)
        chunks = [Chunk(0, only, tok, 0, max(0, n - 1), 1.0)]
        dt = (time.perf_counter() - t0) * 1000.0
        return ChunkingResult(
            method=label, q=params.q, chunks=chunks, tree=None, boundary_signal=[],
            boundary_distribution=[], boundaries=[], shannon_bits=0.0,
            tsallis_bits=0.0, diversity_number=1.0, chunking_time_ms=dt,
            n_sentences=n, tree_entropy=None,
        )

    gap_d = _gap_distances(sent_emb, params.window)        # len n-1
    sims = 1.0 - gap_d

    # PROMINENCE BAR -- SELF-ADJUSTING (no magic percentile-of-index constants).
    # We use a robust outlier rule on the document's OWN gap distribution: a gap
    # is "prominent" if it sits k(q) robust-deviations above the median.  median
    # + MAD adapt to each document's scale and spread, so the same q behaves
    # sensibly whether the text has sharp topic jumps or is uniformly dense.
    #   k(q): q=1 -> ~2.6 dev (only dominant shifts split: coarse, == Shannon)
    #         q=0 -> ~1.5 dev (natural)
    #         q=-1-> ~0.4 dev (weak shifts also split: fine)
    baseline = float(np.median(gap_d))                     # self-similar scale
    mad = float(np.median(np.abs(gap_d - baseline))) * 1.4826  # robust std
    spread = max(mad, 1e-6)
    k_q = float(np.clip(1.5 + 1.1 * params.q, 0.0, 3.0))
    tau = baseline + k_q * spread

    sent_tokens = [embedder.count_tokens(s) for s in sentences]
    prefix = np.concatenate([[0], np.cumsum(sent_tokens)]).astype(np.int64)

    def span_tokens(s: int, e: int) -> int:
        return int(prefix[e + 1] - prefix[s])

    # HARD boundaries: the gap before every heading (section/table break) is
    # ALWAYS opened so a chunk never straddles a structural unit.
    struct = structural_boundary_gaps(sentences) if params.use_structure else set()
    opened: set = set()
    DEPTH_CAP = 14

    def preview(s: int, e: int) -> str:
        return _span_preview(sentences, s, e)

    def pick(cands: np.ndarray, k: int, s: int, e: int, forced: List[int]) -> List[int]:
        """Open `forced` (hard heading) gaps first with a relaxed floor, then add
        the strongest semantic `cands` up to a total of k, always keeping every
        resulting sub-span >= min_chunk_tokens."""
        chosen: List[int] = []

        def ok(g: int, floor: int) -> bool:
            left = max([b for b in chosen if b < g], default=s - 1)
            right = min([b for b in chosen if b > g], default=e)
            return (span_tokens(left + 1, g) >= floor
                    and span_tokens(g + 1, right) >= floor)

        relaxed = max(8, params.min_chunk_tokens // 2)
        for g in sorted(forced):
            if g not in chosen and ok(int(g), relaxed):
                chosen.append(int(g)); chosen.sort()
        target = max(k, len(chosen))                 # never drop a hard boundary
        for g in sorted(cands, key=lambda gg: -gap_d[gg]):
            if len(chosen) >= target:
                break
            g = int(g)
            if g in chosen:
                continue
            if ok(g, params.min_chunk_tokens):
                chosen.append(g); chosen.sort()
        return sorted(chosen)

    def build(s: int, e: int, depth: int) -> TreeNode:
        node = TreeNode(s, e, preview(s, e), tokens=span_tokens(s, e))
        if e <= s or depth > DEPTH_CAP:
            return node
        tok = span_tokens(s, e)
        inner = np.arange(s, e)                      # gap g => split after sent g
        prom = inner[gap_d[inner] >= tau]            # gaps that clear the q-bar
        hard = [int(g) for g in inner if g in struct]  # heading breaks in this span
        oversized = tok > params.max_chunk_tokens

        # Stop: at the token floor, or a coherent span (no prominent shift, no
        # heading break) that already fits the size budget.
        if tok <= params.min_chunk_tokens:
            return node
        if prom.size == 0 and not hard and not oversized:
            return node

        # Local q-diversity of the prominent peaks sets how many SEMANTIC cuts to
        # open at this node, capped at the paper's branching factor K-1.  Hard
        # heading breaks are always opened on top of that.  With use_dq=False
        # (ablation) every gap above tau is opened directly — no D_q, no K cap —
        # which isolates what the entropy machinery adds over plain thresholding.
        if not params.use_dq:
            n_cuts = max(1, int(prom.size))
        elif prom.size > 0:
            excess = np.clip(gap_d[prom] - baseline, 1e-9, None)
            d_q = ent.diversity_number(ent.normalize(excess), params.q)
            n_cuts = max(1, min(int(round(d_q)), K - 1))
        else:
            n_cuts = 1                               # oversized/heading: at least bisect

        cand = prom if prom.size > 0 else inner
        chosen = pick(cand, n_cuts, s, e, forced=hard)
        if not chosen:
            return node                              # can't split under min tokens

        node.strength = float(max(gap_d[g] for g in chosen))
        prev = s
        for g in chosen:
            opened.add(g)
            node.children.append(build(prev, g, depth + 1))
            prev = g + 1
        node.children.append(build(prev, e, depth + 1))
        return node

    root = build(0, n - 1, 0)

    # Leaves, in reading order, are the chunks.
    leaf_spans: List[tuple] = []

    def collect(nd: TreeNode) -> None:
        if nd.children:
            for c in nd.children:
                collect(c)
        else:
            leaf_spans.append((nd.sent_start, nd.sent_end))

    collect(root)
    chunks: List[Chunk] = []
    for (s, e) in leaf_spans:
        chunks.append(_make_chunk(s, e, sentences, sims, embedder))
    for idx, c in enumerate(chunks):
        c.index = idx
    chunks = _attach_section_paths(chunks, sentences)
    chunks = _apply_overlap(chunks, sentences, params.overlap, embedder)

    # Paper-faithful tree entropy (Shannon H, rate h_K) + Tsallis q-extension.
    tent = te.tree_entropy(root, K, total_tokens=int(prefix[-1]), q=params.q)
    p_all = ent.normalize(gap_d + 1e-9)
    dt = (time.perf_counter() - t0) * 1000.0
    return ChunkingResult(
        method=label, q=params.q, chunks=chunks, tree=root,
        boundary_signal=list(gap_d),
        boundary_distribution=list(p_all),
        boundaries=sorted(opened),
        shannon_bits=tent["shannon_nats"] / _LN2,
        tsallis_bits=tent["tsallis_nats"] / _LN2,
        diversity_number=float(len(chunks)),
        chunking_time_ms=dt, n_sentences=n,
        tree_entropy={
            "shannon_nats": round(tent["shannon_nats"], 4),
            "tsallis_nats": round(tent["tsallis_nats"], 4),
            "entropy_rate": round(tent["entropy_rate"], 4),
            "n_internal": tent["n_internal"],
            "K": tent["K"],
        },
    )
