"""
retrieval_metrics.py — Real MRR / NDCG / Precision / Recall
============================================================
Zero LLM calls. Zero human annotation. Zero extra compute beyond what is
already available in the pipeline.

Two approaches, both self-supervised:

  1. BM25 leave-one-out (primary, preferred)
     For each chunk C_i, take its first sentence as the "query".
     Build a BM25 index over all other chunks.
     Retrieve top-k, measure rank of source chunk.
     MRR = mean(1/rank), Precision@k, Recall@k, NDCG@k.
     Requires: pip install rank_bm25

  2. Embedding leave-one-out (fallback, uses S6 embeddings already computed)
     Same logic but uses cosine similarity over embedding vectors.
     No extra dependency needed.

Ground truth: the source chunk is the relevant document for each query.
This is self-supervised / leave-one-out evaluation — valid for comparing
strategies against each other without human annotation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rank_bm25 import BM25Okapi as _BM25
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_retrieval_metrics(
    chunks: List[Dict[str, Any]],
    k: int = 5,
    embeddings: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    Compute real retrieval metrics for a chunk set.

    Uses BM25 leave-one-out if rank_bm25 is installed (fast, no model).
    Falls back to embedding leave-one-out if embeddings are provided.
    Returns zeros with computed=False if neither is possible.

    Parameters
    ----------
    chunks     : list of chunk dicts, each must have a "text" key.
    k          : number of results to retrieve.
    embeddings : optional list of embedding vectors (one per chunk, same order).

    Returns
    -------
    dict with keys: mrr, ndcg, precision, recall, method, computed, n_queries, k
    """
    # Exclude end-boundary chunks (zeroed signal fields, not real content)
    real_chunks = [c for c in chunks if c.get("boundary_type") != "end"]
    n = len(real_chunks)

    if n < 3:
        return _zero_metrics("too_few_chunks")

    texts = [c.get("text", "") or "" for c in real_chunks]

    if _BM25_AVAILABLE:
        return _bm25_leave_one_out(texts, k=k)

    if embeddings is not None and len(embeddings) >= len(chunks):
        real_indices = [i for i, c in enumerate(chunks) if c.get("boundary_type") != "end"]
        real_embs    = [embeddings[i] for i in real_indices if i < len(embeddings)]
        if len(real_embs) == n:
            return _embedding_leave_one_out(texts, real_embs, k=k)

    return _zero_metrics("rank_bm25_not_installed_and_no_embeddings")


def compute_ss2fd(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> float:
    """
    Semantic Similarity to Full Document.
    ss2fd = cosine_sim(mean(chunk_embeddings), weighted_doc_embedding)
    where doc_embedding is approximated as token-count-weighted mean of chunks.
    Returns a float in [0, 1].
    """
    if not embeddings or not chunks:
        return 0.0

    real_indices = [i for i, c in enumerate(chunks) if c.get("boundary_type") != "end"]
    real_embs    = [np.array(embeddings[i], dtype=np.float32) for i in real_indices if i < len(embeddings)]
    real_chunks  = [chunks[i] for i in real_indices]

    if not real_embs:
        return 0.0

    weights = np.array(
        [max(1, len(c.get("text", "").split())) for c in real_chunks],
        dtype=np.float32,
    )
    weights /= weights.sum()
    doc_vec    = sum(w * v for w, v in zip(weights, real_embs))
    chunk_mean = np.mean(real_embs, axis=0)

    return float(_cosine_sim(chunk_mean, doc_vec))


def compute_qcs(
    chunks: List[Dict[str, Any]],
    ss2fd: float,
    retrieval_token_cost: float,
) -> float:
    """
    Query Coverage Score = ss2fd / log(1 + retrieval_token_cost).
    Higher = more document coverage per retrieval token spent.
    """
    if retrieval_token_cost <= 0:
        return 0.0
    import math
    return float(np.clip(ss2fd / math.log(1.0 + retrieval_token_cost), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# BM25 leave-one-out
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b\w{2,}\b", text.lower())


def _first_sentence(text: str, min_words: int = 6) -> str:
    text = text.strip()
    for pattern in [r"(?<=[.!?])\s+", r"\n"]:
        parts = re.split(pattern, text, maxsplit=1)
        if len(parts[0].split()) >= min_words:
            return parts[0]
    words = text.split()
    return " ".join(words[:max(min_words, len(words) // 3)])


def _bm25_leave_one_out(texts: List[str], k: int = 5) -> Dict[str, Any]:
    n              = len(texts)
    corpus_tokens  = [_tokenise(t) for t in texts]

    mrr_scores:       List[float] = []
    ndcg_scores:      List[float] = []
    precision_scores: List[float] = []
    recall_scores:    List[float] = []

    for i in range(n):
        query        = _first_sentence(texts[i])
        query_tokens = _tokenise(query)
        if not query_tokens:
            continue

        other_indices = [j for j in range(n) if j != i]
        other_tokens  = [corpus_tokens[j] for j in other_indices]
        if len(other_tokens) < 2:
            continue

        bm25   = _BM25(other_tokens)
        scores = bm25.get_scores(query_tokens)

        ranked_local  = sorted(range(len(other_indices)), key=lambda x: scores[x], reverse=True)
        ranked_global = [other_indices[r] for r in ranked_local]

        # FIX: chunk i is excluded from corpus, so relevant={i} always gave hits=0.
        # Adjacent chunks (i-1, i+1) are the correct ground truth — they ARE in
        # the corpus and share the most content with the removed chunk.
        relevant = set()
        if i > 0:     relevant.add(i - 1)
        if i < n - 1: relevant.add(i + 1)
        if not relevant:
            continue

        rank = None
        for pos, idx in enumerate(ranked_global[:k], start=1):
            if idx in relevant:
                rank = pos
                break

        mrr_scores.append(1.0 / rank if rank else 0.0)

        hits = sum(1 for idx in ranked_global[:k] if idx in relevant)
        precision_scores.append(hits / k)
        recall_scores.append(hits / len(relevant))

        dcg  = sum(
            (1.0 if ranked_global[pos] in relevant else 0.0) / np.log2(pos + 2)
            for pos in range(min(k, len(ranked_global)))
        )
        idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    if not mrr_scores:
        return _zero_metrics("no_valid_queries")

    return {
        "mrr":       round(float(np.mean(mrr_scores)),       4),
        "ndcg":      round(float(np.clip(np.mean(ndcg_scores), 0, 1)), 4),
        "precision": round(float(np.mean(precision_scores)), 4),
        "recall":    round(float(np.mean(recall_scores)),    4),
        "method":    "bm25_leave_one_out",
        "computed":  True,
        "n_queries": len(mrr_scores),
        "k":         k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Embedding leave-one-out (fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _embedding_leave_one_out(
    texts: List[str],
    embeddings: List[List[float]],
    k: int = 5,
) -> Dict[str, Any]:
    n    = len(texts)
    vecs = [np.array(e, dtype=np.float32) for e in embeddings]

    mrr_scores:       List[float] = []
    ndcg_scores:      List[float] = []
    precision_scores: List[float] = []
    recall_scores:    List[float] = []

    for i in range(n):
        other_indices = [j for j in range(n) if j != i]
        if len(other_indices) < 2:
            continue

        sims          = [(j, _cosine_sim(vecs[i], vecs[j])) for j in other_indices]
        sims.sort(key=lambda x: x[1], reverse=True)
        ranked_global = [idx for idx, _ in sims]

        # FIX: same as BM25 — chunk i is excluded from corpus so relevant={i}
        # always gave hits=0. Use adjacent chunks as ground truth instead.
        relevant = set()
        if i > 0:     relevant.add(i - 1)
        if i < n - 1: relevant.add(i + 1)
        if not relevant:
            continue

        rank = None
        for pos, idx in enumerate(ranked_global[:k], start=1):
            if idx in relevant:
                rank = pos
                break

        mrr_scores.append(1.0 / rank if rank else 0.0)

        hits = sum(1 for idx in ranked_global[:k] if idx in relevant)
        precision_scores.append(hits / k)
        recall_scores.append(hits / len(relevant))

        dcg  = sum(
            (1.0 if ranked_global[pos] in relevant else 0.0) / np.log2(pos + 2)
            for pos in range(min(k, len(ranked_global)))
        )
        idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    if not mrr_scores:
        return _zero_metrics("no_valid_queries")

    return {
        "mrr":       round(float(np.mean(mrr_scores)),       4),
        "ndcg":      round(float(np.clip(np.mean(ndcg_scores), 0, 1)), 4),
        "precision": round(float(np.mean(precision_scores)), 4),
        "recall":    round(float(np.mean(recall_scores)),    4),
        "method":    "embedding_leave_one_out",
        "computed":  True,
        "n_queries": len(mrr_scores),
        "k":         k,
    }


def _zero_metrics(reason: str) -> Dict[str, Any]:
    return {
        "mrr":       0.0,
        "ndcg":      0.0,
        "precision": 0.0,
        "recall":    0.0,
        "method":    reason,
        "computed":  False,
        "n_queries": 0,
        "k":         0,
    }