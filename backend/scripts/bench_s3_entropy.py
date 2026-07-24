"""
bench_s3_entropy.py — controlled A/B of the S3 entropy formula.
===============================================================
Compares the two generalized-entropy families that drive S3's boundary count,
per document and per Tsallis q (both defined in pipeline/s3_entropy.py header):

    tsallis : S_q = (1 - Σ p_i^q)/(q-1)          -> Hill number D_q  (the default)
    qlog    : S_q^K = (1-q) Σ_i p_i Σ_n (1-p_i)^n/(1-q^n)  (the new q-log formula)
              -> numbers-equivalent effective count

Everything upstream (S1 profile, S2 strategy) and downstream (S4 default gate,
S5/S6) is held IDENTICAL between the two arms, so any metric delta is
attributable to the S3 entropy choice alone.  q=-1 is excluded (the q-log form
is singular there; see the S3 header).  Judged (answer-correctness on) so the
AC column is populated.

Requires a working OPENAI_API_KEY.  Resumable: saves after every document.
Run from backend/:  python scripts/bench_s3_entropy.py
Emits:  scripts/s3_entropy_results.json   (raw per-doc numbers)
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from engine import qagen  # noqa: E402
from engine.embeddings import get_embedder  # noqa: E402
from pipeline import evaluation as EV  # noqa: E402
from pipeline.s1_profiler import profile_document  # noqa: E402
from pipeline.s2_chunkers import run_all_chunkers  # noqa: E402
from pipeline.s3_entropy import refine_boundaries  # noqa: E402
from pipeline.s4_boundary import filter_boundaries  # noqa: E402
from main import _parse_file  # noqa: E402
from scripts.run_benchmark import infer_domain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(os.path.dirname(HERE), "..", "test_documents")
OUT_PATH = os.path.join(HERE, "s3_entropy_results.json")

EXCLUDE_DOCS = {"code_irpp_s_2023.pdf"}          # historical OOM trigger; excluded as elsewhere
BACKEND = os.environ.get("QCOS_GA_BACKEND", "openai")
JUDGE = os.environ.get("QCOS_GA_JUDGE", "1") != "0"
Q_SWEEP = [-0.5, 0.0, 0.5, 1.0]                  # q=-1 excluded (q-log singular there)
QA_COUNT = 10
MAX_DOC_TOKENS = 12000
MODES = ["tsallis", "qlog"]
KPIS = ["precision", "recall", "f1", "mrr", "ndcg", "qcs",
        "retrieval_token_cost", "answer_correct"]


def _pick(m: dict) -> dict:
    out = {k: m.get(k, 0.0) for k in KPIS}
    out["n_chunks"] = m.get("n_chunks", 0)
    return out


def _load() -> list:
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return []
    return []


def _save(R: list) -> None:
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, OUT_PATH)


def bench_doc(fn: str, domain: str) -> dict:
    import pipeline.s3_entropy as _S3
    _S3._EMB_CACHE.clear()
    EV._CE_CACHE.clear()

    text = _parse_file(fn, open(os.path.join(DOC_DIR, fn), "rb").read())
    words = text.split()
    if len(words) > MAX_DOC_TOKENS:
        text = " ".join(words[:MAX_DOC_TOKENS])
        print(f"  (truncated to {MAX_DOC_TOKENS} tokens)", flush=True)
    profile = profile_document(text, {})
    doc_type = profile.get("type", "prose")

    qa = qagen.generate_qa(text, n=QA_COUNT) if JUDGE else []
    ctx = EV.build_context(text, qa_pairs=qa, backend=BACKEND, judge=JUDGE)
    embedder = get_embedder()
    cfg_base = {"n_min": 80, "n_max": 500, "K": 4, "min_chunk_tokens": 20,
                "max_chunk_tokens": 320, "window": 1, "tau_sem": 0.75,
                "embedding_backend": BACKEND}
    print(f"  QA={len(qa)} judge={ctx.judge} rel_thr={ctx.rel_threshold:.2f}", flush=True)

    # Representative S2 strategy = best pre-entropy cosine-rank (judge off).
    all_s2 = run_all_chunkers(text, doc_type, cfg_base)
    scored = {}
    for s, ch in all_s2.items():
        if ch:
            m = EV.score_run(ch, ctx, judge=False)
            scored[s] = (float(np.mean([m.get(k, 0.0) for k in ("mrr", "ndcg", "srgt", "qcs")])), ch)
    raw = max(scored.values(), key=lambda t: t[0])[1] if scored else []
    strat = max(scored, key=lambda s: scored[s][0]) if scored else None
    print(f"  strategy={strat} ({len(raw)} raw chunks)", flush=True)

    res = {"doc": fn, "domain": domain, "n_tokens": len(text.split()),
           "n_queries": len(ctx.queries), "strategy": strat, "by_q": {}}
    for q in Q_SWEEP:
        row = {}
        for mode in MODES:
            # S3 with the chosen entropy formula; S4 held at the default gate.
            ref = refine_boundaries([dict(c) for c in raw],
                                    {**cfg_base, "q_entropy_param": q, "entropy_mode": mode})
            emb = embedder.embed([c["text"] for c in ref], BACKEND).tolist()
            filt = filter_boundaries([dict(c) for c in ref], doc_type, emb,
                                     {**cfg_base, "q_entropy_param": q})
            m = EV.score_run(filt, ctx)
            row[mode] = {**_pick(m), "n_chunks": len(filt),
                         "d_q": (ref[0].get("diversity_number") if ref else None)}
        res["by_q"][f"{q:g}"] = row
        print(f"  q={q:+.1f}  tsallis: F1={row['tsallis']['f1']:.3f} n={row['tsallis']['n_chunks']} "
              f"| qlog: F1={row['qlog']['f1']:.3f} n={row['qlog']['n_chunks']}", flush=True)
    return res


def main() -> None:
    docs = sorted(f for f in os.listdir(DOC_DIR)
                  if os.path.isfile(os.path.join(DOC_DIR, f)) and f not in EXCLUDE_DOCS)
    R = _load()
    done = {r["doc"] for r in R if "by_q" in r}
    print(f"Backend={BACKEND} judge={JUDGE} -- {len(done)}/{len(docs)} done (resuming). "
          f"Excluded: {sorted(EXCLUDE_DOCS)}", flush=True)
    for fn in docs:
        if fn in done:
            continue
        print(f"\n=== {fn} ({infer_domain(fn)}) ===", flush=True)
        try:
            r = bench_doc(fn, infer_domain(fn))
            R = [x for x in R if x["doc"] != fn] + [r]
            _save(R)
        except Exception as exc:
            print(f"  !! {fn} failed: {exc}", flush=True)
    if R:
        from statistics import mean

        def col(mode, key):
            return [r["by_q"][q][mode][key] for r in R for q in r["by_q"]
                    if isinstance(r["by_q"][q][mode].get(key), (int, float))]
        print("\n" + "=" * 60)
        for key in ("f1", "answer_correct"):
            t, ql = col("tsallis", key), col("qlog", key)
            if t and ql:
                print(f"Mean {key:15s} -- tsallis: {mean(t):.4f}   qlog: {mean(ql):.4f}"
                      f"   delta: {mean(ql) - mean(t):+.4f}")
    print(f"\nWrote {os.path.relpath(OUT_PATH)} ({len(R)} documents).")


if __name__ == "__main__":
    main()
