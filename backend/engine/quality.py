"""Label-free chunk-quality score + automatic q selection.

The retrieval metrics in metrics.py need (query, answer) pairs.  But on a brand
new, hard document you often have NO questions yet — so how do you pick a good
granularity?  This module scores a chunking using only the embeddings, so it
works on ANY document and lets the tool tune itself.

A good chunking has:
  * HIGH coherence   — sentences inside a chunk are similar (belong together);
  * HIGH separation  — neighbouring chunks are about different things;
  * BALANCED sizes   — no mix of huge and tiny chunks.

`chunking_quality` combines these into one number (higher = better).  Because
splitting too finely makes neighbours look similar (low separation) and too
coarsely lowers coherence, the score has a natural sweet spot — which
`choose_best_q` finds by trying several q values and keeping the best.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def chunking_quality(chunks, sent_emb: np.ndarray) -> Tuple[float, dict]:
    """Return (score, details) for a list of chunks given sentence embeddings.

    score = 0.5·coherence + 0.4·separation + 0.1·size_balance, all in [0, 1].
    """
    if not chunks:
        return 0.0, {"coherence": 0.0, "separation": 0.0, "balance": 0.0, "n_chunks": 0}
    if len(chunks) == 1:
        # One chunk: maximally coherent, but no separation signal.
        return 0.0, {"coherence": 1.0, "separation": 0.0, "balance": 1.0, "n_chunks": 1}

    means: List[np.ndarray] = []
    intra: List[float] = []
    for c in chunks:
        seg = sent_emb[c.sent_start : c.sent_end + 1]
        if len(seg) == 0:
            means.append(np.zeros(sent_emb.shape[1]))
            intra.append(1.0)
            continue
        means.append(_unit(seg.mean(axis=0)))
        if len(seg) > 1:
            adj = [float(seg[i] @ seg[i + 1]) for i in range(len(seg) - 1)]
            intra.append(float(np.mean(adj)))
        else:
            intra.append(1.0)

    inter = [float(means[i] @ means[i + 1]) for i in range(len(means) - 1)]
    coherence = float(np.clip(np.mean(intra), 0.0, 1.0))
    separation = float(np.clip(1.0 - np.mean(inter), 0.0, 1.0))

    toks = np.array([max(1, c.tokens) for c in chunks], dtype=np.float64)
    cv = float(np.std(toks) / max(np.mean(toks), 1e-9))   # coeff. of variation
    balance = float(1.0 / (1.0 + cv))

    score = 0.5 * coherence + 0.4 * separation + 0.1 * balance
    return score, {
        "coherence": round(coherence, 4),
        "separation": round(separation, 4),
        "balance": round(balance, 4),
        "n_chunks": len(chunks),
        "score": round(score, 4),
    }


def choose_best_q(
    text: str,
    params,
    embedder,
    sentences: Optional[List[str]] = None,
    sent_emb: Optional[np.ndarray] = None,
    candidates: Optional[List[float]] = None,
    backend: Optional[str] = None,
):
    """Run the recursive chunker for several q values and keep the one with the
    best label-free quality score.  Returns (best_q, best_result, scores)."""
    from .chunking import recursive_chunk_document, split_sentences

    if candidates is None:
        candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    if sentences is None:
        sentences = split_sentences(text)
    if sent_emb is None:
        sent_emb = embedder.embed(sentences, backend)

    best_q, best_res, best_score = candidates[0], None, -1e9
    scores = []
    for q in candidates:
        p = replace(params, q=q)
        res = recursive_chunk_document(
            text, p, embedder, sentences=sentences, sent_emb=sent_emb, backend=backend,
        )
        s, _ = chunking_quality(res.chunks, sent_emb)
        scores.append({"q": q, "score": round(s, 4), "n_chunks": len(res.chunks)})
        if s > best_score:
            best_q, best_res, best_score = q, res, s
    return best_q, best_res, scores
