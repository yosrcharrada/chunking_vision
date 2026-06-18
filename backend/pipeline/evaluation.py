"""
Evaluation — the SINGLE source of metric truth (Project-2 / Table-I).
=====================================================================
Every chunking — each S2 strategy AND the post-GA winner — is scored here with
the *same* retrieval metrics as the qentropy project (engine/metrics.py), and
the overall winner is chosen with the *same* honest policy as
engine services/benchmark.summarize:

  PRIMARY  : LLM answer-correctness when the judge ran, else the mean of the
             cosine rank metrics (mrr/ndcg/srgt/qcs).  precision/recall/f1 are
             deliberately excluded from ranking (they re-use the cosine rule).
  TIES     : a paired bootstrap over queries puts a 95% CI on every pairwise
             difference with the leader; CI ∋ 0 ⇒ statistical tie.
  TIE-BREAK: among ties the cheapest retrieval (lowest token cost) wins.

This module is used in two places:
  * as the S7 GA fitness signal (``ga_fitness`` — fast, embedding-only),
  * as the final S8 evaluation + ranking (``score_run`` + ``rank_runs``).

It standardises on the shared EmbeddingService so all cosine numbers live in one
space, and it degrades gracefully offline (no API key → no judge → cosine-rank
winner; no queries → label-free quality fitness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from engine import answerability as A
from engine import metrics as M
from engine.embeddings import get_embedder, cosine_matrix
from engine.structure import split_sentences

COSINE_RANK_KEYS = ("mrr", "ndcg", "srgt", "qcs")
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 0

# Module-level chunk-text embedding cache: the GA scores many chunkings that
# share chunk texts, so we never re-embed the same text inside a tuning run.
_CE_CACHE: Dict[str, np.ndarray] = {}


def _embed_cached(texts: Sequence[str], embedder, backend) -> np.ndarray:
    miss = [t for t in texts if t not in _CE_CACHE]
    if miss:
        vecs = embedder.embed(miss, backend)
        for t, v in zip(miss, vecs):
            _CE_CACHE[t] = np.asarray(v, dtype=np.float32)
    return np.array([_CE_CACHE[t] for t in texts], dtype=np.float32) if texts else None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation context — embed queries / ground-truths / document ONCE per run
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalContext:
    embedder: object
    backend: Optional[str]
    full_text: str
    queries: List[str]
    ground_truths: List[str]
    q_emb: object
    gt_emb: object
    doc_emb: object
    rel_threshold: float
    judge: bool = False

    @property
    def have_queries(self) -> bool:
        return bool(self.queries)


def build_context(
    text: str,
    qa_pairs: Optional[List[dict]] = None,
    backend: Optional[str] = None,
    rel_threshold="auto",
    judge: bool = False,
) -> EvalContext:
    """Embed the evaluation set once.  ``qa_pairs`` is a list of
    {"query","ground_truth"} dicts (e.g. from engine.qagen.generate_qa)."""
    embedder = get_embedder()
    resolved = embedder.resolve(backend)

    qa_pairs = qa_pairs or []
    queries = [p.get("query", "").strip() for p in qa_pairs if p.get("query", "").strip()]
    gts = [p.get("ground_truth", "") for p in qa_pairs if p.get("query", "").strip()]

    q_emb = embedder.embed(queries, resolved) if queries else np.zeros((0, 8), dtype="float32")
    gt_emb = embedder.embed(gts, resolved) if gts else np.zeros((0, 8), dtype="float32")
    doc_emb = embedder.embed([text], resolved)

    if rel_threshold in (None, "", "auto"):
        # Calibrate the cutoff to the model's cosine scale using sentence sims.
        sents = split_sentences(text)[:400]
        sent_emb = embedder.embed(sents, resolved) if sents else doc_emb
        thr = M.auto_relevance_threshold(sent_emb, q_emb, gt_emb)
    else:
        try:
            thr = float(max(0.0, min(1.0, float(rel_threshold))))
        except (TypeError, ValueError):
            thr = M.DEFAULT_REL_THRESHOLD

    judge = bool(judge and queries and A.available())
    return EvalContext(embedder, resolved, text, queries, gts,
                       q_emb, gt_emb, doc_emb, float(thr), judge)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
def score_run(chunks: List[Dict], ctx: EvalContext, judge: Optional[bool] = None) -> dict:
    """Full Table-I metric set for one chunking (list of chunk dicts)."""
    chunk_texts = [str(c.get("text", "")) for c in chunks]
    chunk_tokens = [int(c.get("tokens", _wc(c.get("text", "")))) for c in chunks]
    chunk_emb = _embed_cached(chunk_texts, ctx.embedder, ctx.backend)
    chunking_time = float(sum(float(c.get("chunking_time_ms", 0.0)) for c in chunks))

    met = M.evaluate_chunking(
        chunk_texts, chunk_tokens, ctx.queries, ctx.ground_truths, ctx.full_text,
        chunking_time, ctx.embedder, backend=ctx.backend, rel_threshold=ctx.rel_threshold,
        q_emb=ctx.q_emb, gt_emb=ctx.gt_emb, doc_emb=ctx.doc_emb, chunk_emb=chunk_emb,
    )

    do_judge = ctx.judge if judge is None else (judge and ctx.judge)
    if do_judge and chunk_texts and ctx.queries:
        ctxs = M.retrieved_contexts(chunk_texts, chunk_emb, ctx.q_emb, k=3)
        tasks = [(ctx.queries[i], ctx.ground_truths[i] if i < len(ctx.ground_truths) else "",
                  ctxs[i] if i < len(ctxs) else "") for i in range(len(ctx.queries))]
        verdicts = A.judge_many(tasks)
        n = max(len(verdicts), 1)
        met["answerable"] = round(sum(1 for v in verdicts if v["answerable"]) / n, 4)
        met["answer_correct"] = round(sum(1 for v in verdicts if v["correct"]) / n, 4)
        met.setdefault("per_query", {})["answer_correct"] = [
            1.0 if v["correct"] else 0.0 for v in verdicts]
        met["per_query"]["answerable"] = [1.0 if v["answerable"] else 0.0 for v in verdicts]
    return met


def ga_fitness(chunks: List[Dict], ctx: EvalContext) -> float:
    """Fast scalar fitness for the S7 GA, in [0, 1].

    With queries: the primary P2 signal — LLM answer-correctness when judged,
    else the cosine rank mean (mrr/ndcg/srgt/qcs).  Without queries: a label-free
    coherence/separation/balance quality (so the GA still tunes q offline)."""
    if not chunks:
        return 0.0
    if ctx.have_queries:
        met = score_run(chunks, ctx, judge=ctx.judge)
        if ctx.judge and "answer_correct" in met:
            return float(met["answer_correct"])
        return float(np.mean([met.get(k, 0.0) for k in COSINE_RANK_KEYS]))
    return _label_free_quality(chunks, ctx)


def _label_free_quality(chunks: List[Dict], ctx: EvalContext) -> float:
    """Separation between neighbouring chunks + size balance (no labels needed)."""
    texts = [str(c.get("text", "")) for c in chunks]
    if len(texts) < 2:
        return 0.0
    emb = _embed_cached(texts, ctx.embedder, ctx.backend)
    inter = [float(np.dot(emb[i], emb[i + 1])) for i in range(len(emb) - 1)]
    separation = float(np.clip(1.0 - np.mean(inter), 0.0, 1.0))
    toks = np.array([max(1, int(c.get("tokens", _wc(c.get("text", ""))))) for c in chunks],
                    dtype=np.float64)
    cv = float(np.std(toks) / max(np.mean(toks), 1e-9))
    balance = float(1.0 / (1.0 + cv))
    return float(np.clip(0.6 * separation + 0.4 * balance, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Winner policy (ported from engine services/benchmark.summarize)
# ─────────────────────────────────────────────────────────────────────────────
def _per_query_primary(metrics: dict, judged: bool) -> Optional[np.ndarray]:
    pq = metrics.get("per_query") or {}
    if judged:
        vals = pq.get("answer_correct")
        return np.asarray(vals, dtype=np.float64) if vals else None
    cols = [pq.get(k) for k in COSINE_RANK_KEYS]
    if any(c is None for c in cols) or not cols[0]:
        return None
    return np.mean(np.asarray(cols, dtype=np.float64), axis=0)


def _bootstrap_diff_ci(a: np.ndarray, b: np.ndarray,
                       iters: int = BOOTSTRAP_ITERS, seed: int = BOOTSTRAP_SEED) -> tuple:
    n = len(a)
    if n == 0:
        return (0.0, 0.0)
    if n == 1 or np.allclose(a, b):
        d = float(np.mean(a) - np.mean(b))
        return (d, d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    diffs = np.mean(a[idx], axis=1) - np.mean(b[idx], axis=1)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def rank_runs(runs: List[dict], judged: bool = False,
              metric_info: Optional[List[dict]] = None) -> dict:
    """runs: [{id,label,metrics}]. Returns {best_per_metric, overall} exactly like
    the qentropy benchmark summary."""
    info = metric_info or M.METRIC_INFO
    best_per_metric = {}
    for m in info:
        key, higher = m["key"], m["higher"]
        if higher is None:
            continue
        vals = [(r["id"], r["metrics"].get(key, 0.0)) for r in runs]
        if not vals:
            continue
        best = max(vals, key=lambda kv: kv[1]) if higher else min(vals, key=lambda kv: kv[1])
        best_per_metric[key] = best[0]

    overall = None
    have_queries = any((r["metrics"].get("per_query") for r in runs))
    if have_queries:
        scored = []
        for r in runs:
            pq = _per_query_primary(r["metrics"], judged)
            if pq is None:
                continue
            cosine_mean = float(np.mean([r["metrics"].get(k, 0.0) for k in COSINE_RANK_KEYS]))
            scored.append({
                "id": r["id"], "label": r.get("label", r["id"]),
                "pq": pq, "score": float(np.mean(pq)),
                "tokens": float(r["metrics"].get("retrieval_token_cost", 0.0)),
                "cosine_mean": cosine_mean,
            })
        if scored:
            scored.sort(key=lambda s: (-s["score"], s["tokens"], -s["cosine_mean"]))
            leader = scored[0]
            tie_ids = []
            for s in scored[1:]:
                lo, hi = _bootstrap_diff_ci(leader["pq"], s["pq"])
                s["diff_ci"] = [round(lo, 4), round(hi, 4)]
                if lo <= 0.0 <= hi:
                    tie_ids.append(s["id"])
            decided_by, winner = "primary", leader
            if tie_ids:
                tied = [leader] + [s for s in scored if s["id"] in tie_ids]
                cheapest = min(tied, key=lambda s: (s["tokens"], -s["cosine_mean"]))
                if cheapest["id"] != leader["id"]:
                    winner = cheapest
                decided_by = "token_cost_among_ties"
            overall = {
                "winner_id": winner["id"], "winner_label": winner["label"],
                "ranked_by": "answer_correct" if judged else "cosine_rank_mean",
                "decided_by": decided_by, "tie_ids": tie_ids,
                "note": (None if judged else
                         "cosine-only ranking — enable the LLM answerability judge for the honest signal"),
                "ranking": [{
                    "id": s["id"], "label": s["label"], "score": round(s["score"], 4),
                    "tokens": round(s["tokens"], 1),
                    "diff_ci_vs_leader": s.get("diff_ci"),
                    "tied_with_leader": s["id"] in tie_ids or s["id"] == leader["id"],
                } for s in scored],
            }
    return {"best_per_metric": best_per_metric, "overall": overall}


def _wc(text: str) -> int:
    return max(1, len((text or "").split()))
