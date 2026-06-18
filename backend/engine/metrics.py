"""Evaluation metrics from Table I of the chunking-comparison paper.

Given a chunking (list of chunk texts) and a set of queries each paired with a
ground-truth answer, we embed everything once and compute:

  Precision & Recall  cosine >= 0.7.  A chunk is "true" if cos(chunk, GT) >= 0.7,
                      "predicted" if cos(chunk, query) >= 0.7.
                      P = TP/(TP+FP), R = TP/(TP+FN).
  MRR                 mean reciprocal rank of the first truly-relevant chunk,
                      ranking chunks by cos(query, chunk).
  NDCG@5              graded relevance = max(0, cos(chunk, GT)); top-5 by query sim.
  ss2fd               mean cos(chunk, full-document embedding).
  SRGT                mean cos(retrieved top-5 chunk, GT).
  Retrieval Token     mean total tokens across the top-5 retrieved chunks.
  Cost
  QCS                 mean cos(query, top-5 retrieved chunks).
  Chunking Time       wall-clock ms to produce the chunking (passed in).

All cosine values use L2-normalised embeddings.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .embeddings import EmbeddingService, cosine_matrix

DEFAULT_REL_THRESHOLD = 0.7
TOP_K = 5


def _dcg(rels: Sequence[float]) -> float:
    rels = np.asarray(rels, dtype=np.float64)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def retrieved_contexts(
    chunk_texts: List[str], chunk_emb, q_emb, k: int = 3
) -> List[str]:
    """For each query, return the concatenated text of its top-`k` retrieved
    chunks (ranked by query↔chunk cosine).  Used by the answerability judge."""
    if not chunk_texts or chunk_emb is None or q_emb is None or q_emb.size == 0:
        return []
    sims = cosine_matrix(chunk_emb, q_emb)  # (n_chunks, n_queries)
    out: List[str] = []
    kk = min(k, len(chunk_texts))
    for qi in range(q_emb.shape[0]):
        order = np.argsort(-sims[:, qi], kind="stable")[:kk]
        out.append("\n\n".join(chunk_texts[int(i)] for i in order))
    return out


def auto_relevance_threshold(
    ref_emb: np.ndarray, q_emb: np.ndarray, gt_emb: np.ndarray
) -> float:
    """Pick a relevance cutoff from the actual similarity distribution so the
    metric adapts to the embedding model's scale (OpenAI/multilingual cosines
    run lower than the 0.7 the paper assumed).  We take the high tail of
    reference (sentence) similarities to the queries + ground truths.

    CAVEAT: the threshold is calibrated per benchmark run, so precision/recall
    are comparable BETWEEN methods within one run, but NOT across documents.
    Treat them as within-run diagnostics; the winner is decided by the
    answerability judge + token cost (see services/benchmark.summarize)."""
    sims = []
    for other in (q_emb, gt_emb):
        if ref_emb.size and other.size:
            sims.append(cosine_matrix(ref_emb, other).flatten())
    if not sims:
        return DEFAULT_REL_THRESHOLD
    allsim = np.concatenate(sims)
    if allsim.size == 0:
        return DEFAULT_REL_THRESHOLD
    thr = float(np.percentile(allsim, 88))
    return float(np.clip(thr, 0.30, 0.70))


def evaluate_chunking(
    chunk_texts: List[str],
    chunk_tokens: List[int],
    queries: List[str],
    ground_truths: List[str],
    full_doc_text: str,
    chunking_time_ms: float,
    embedder: EmbeddingService,
    backend: Optional[str] = None,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
    q_emb=None,
    gt_emb=None,
    doc_emb=None,
    chunk_emb=None,
) -> dict:
    if not chunk_texts:
        return _empty_metrics(chunking_time_ms, rel_threshold)

    REL = rel_threshold
    if chunk_emb is None:                       # allow callers to batch-embed once
        chunk_emb = embedder.embed(chunk_texts, backend)
    if doc_emb is None:
        doc_emb = embedder.embed([full_doc_text], backend)  # (1, d)

    # ss2fd: independent of queries.
    ss2fd = float(np.mean(cosine_matrix(chunk_emb, doc_emb)[:, 0]))

    if not queries:
        m = _empty_metrics(chunking_time_ms, REL)
        m["ss2fd"] = round(ss2fd, 4)
        m["n_chunks"] = len(chunk_texts)
        m["avg_chunk_tokens"] = round(float(np.mean(chunk_tokens)), 1)
        return m

    if q_emb is None:
        q_emb = embedder.embed(queries, backend)
    if gt_emb is None:
        gt_emb = embedder.embed(ground_truths, backend)

    sim_cq = cosine_matrix(chunk_emb, q_emb)    # (n_chunks, n_queries)
    sim_cg = cosine_matrix(chunk_emb, gt_emb)   # (n_chunks, n_queries)

    n_chunks = len(chunk_texts)
    tokens_arr = np.asarray(chunk_tokens, dtype=np.float64)

    precisions, recalls = [], []
    mrrs, ndcgs, srgts, qcss, token_costs = [], [], [], [], []

    for qi in range(len(queries)):
        cq = sim_cq[:, qi]
        cg = sim_cg[:, qi]

        true_mask = cg >= REL
        pred_mask = cq >= REL
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(pred_mask & ~true_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

        order = np.argsort(-cq, kind="stable")  # rank chunks by query similarity
        # MRR: first truly-relevant chunk in the ranking.
        rr = 0.0
        for rank, idx in enumerate(order, start=1):
            if true_mask[idx]:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        top = order[: min(TOP_K, n_chunks)]
        # NDCG@5 with graded relevance = max(0, cos(chunk, GT)).
        gains = np.clip(cg, 0.0, None)
        dcg = _dcg(gains[top])
        ideal = _dcg(np.sort(gains)[::-1][: min(TOP_K, n_chunks)])
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

        srgts.append(float(np.mean(cg[top])))
        qcss.append(float(np.mean(cq[top])))
        token_costs.append(float(np.sum(tokens_arr[top])))

    return {
        "precision": round(float(np.mean(precisions)), 4),
        "recall": round(float(np.mean(recalls)), 4),
        "f1": round(_f1(np.mean(precisions), np.mean(recalls)), 4),
        "mrr": round(float(np.mean(mrrs)), 4),
        "ndcg": round(float(np.mean(ndcgs)), 4),
        "ss2fd": round(ss2fd, 4),
        "srgt": round(float(np.mean(srgts)), 4),
        "retrieval_token_cost": round(float(np.mean(token_costs)), 1),
        "qcs": round(float(np.mean(qcss)), 4),
        "chunking_time_ms": round(float(chunking_time_ms), 2),
        "n_chunks": n_chunks,
        "avg_chunk_tokens": round(float(np.mean(chunk_tokens)), 1),
        "rel_threshold": round(float(REL), 3),
        # Per-query values: the bootstrap in summarize() resamples these to put
        # confidence intervals on the ranking instead of trusting point means.
        "per_query": {
            "mrr": [round(float(x), 4) for x in mrrs],
            "ndcg": [round(float(x), 4) for x in ndcgs],
            "srgt": [round(float(x), 4) for x in srgts],
            "qcs": [round(float(x), 4) for x in qcss],
            "token_cost": [round(float(x), 1) for x in token_costs],
        },
    }


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _empty_metrics(chunking_time_ms: float, rel_threshold: float = DEFAULT_REL_THRESHOLD) -> dict:
    return {
        "rel_threshold": round(float(rel_threshold), 3),
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "mrr": 0.0, "ndcg": 0.0,
        "ss2fd": 0.0, "srgt": 0.0, "retrieval_token_cost": 0.0, "qcs": 0.0,
        "chunking_time_ms": round(float(chunking_time_ms), 2),
        "n_chunks": 0, "avg_chunk_tokens": 0.0,
    }


# Metric metadata for the frontend: label, whether higher is better, group.
# NOTE: precision/recall/f1 compare two cosine rules against each other (both
# the "true" and "predicted" sides are embedding thresholds), so they are
# diagnostics, not retrieval correctness — they are shown but no longer drive
# the winner ranking.
METRIC_INFO = [
    {"key": "precision", "label": "Precision", "higher": True,
     "desc": "TP/(TP+FP), cosine-threshold diagnostic (not used for ranking)."},
    {"key": "recall", "label": "Recall", "higher": True,
     "desc": "TP/(TP+FN), cosine-threshold diagnostic (not used for ranking)."},
    {"key": "f1", "label": "F1", "higher": True,
     "desc": "Harmonic mean of precision and recall (diagnostic)."},
    {"key": "mrr", "label": "MRR", "higher": True,
     "desc": "Mean reciprocal rank of first relevant chunk."},
    {"key": "ndcg", "label": "NDCG@5", "higher": True,
     "desc": "Normalised discounted cumulative gain at top-5."},
    {"key": "ss2fd", "label": "ss2fd", "higher": None,
     "desc": "Avg cosine(chunk, full document). High=broad, low=focused."},
    {"key": "srgt", "label": "SRGT", "higher": True,
     "desc": "Avg cosine(retrieved chunk, ground-truth answer)."},
    {"key": "qcs", "label": "QCS", "higher": True,
     "desc": "Avg cosine(query, top-5 retrieved chunks)."},
    {"key": "retrieval_token_cost", "label": "Token Cost", "higher": False,
     "desc": "Avg tokens across top-5 retrieved chunks (lower=cheaper)."},
    {"key": "chunking_time_ms", "label": "Chunk Time (ms)", "higher": False,
     "desc": "Wall-clock time to chunk the document."},
]
