"""
bench_s4_similarity.py — controlled A/B of the S4 similarity kernel.
====================================================================
Compares the two S4 semantic-merge kernels on every document in test_documents/:

    cosine   : classical cosine similarity of adjacent chunks (the old way)
    qcosine  : the Fitouhi–Bouzeffour q-cosine of the inter-chunk angle, base = q²
               (Ramanujan J. 28 (2012) 443–461, eqs. 5–6; "q is squared")

Everything upstream (S1 profile, S2 strategy, S3 q-entropy boundaries) and the
embeddings fed to S4 are held IDENTICAL between the two arms, so any metric
delta is attributable to the similarity kernel alone.  Uses the same fixed-dim
embedding backend as the main benchmark so the cosine numbers live in one space.

Run from backend/:  python scripts/bench_s4_similarity.py
Emits:  scripts/s4_similarity_results.json   (raw per-doc numbers)
        scripts/s4_similarity_table.tex       (paper-ready KPI table)
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from collections import defaultdict
from statistics import mean

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
BACKEND = "openai"         # fixed-dim (1536), reliable cross-set cosine; same as the main benchmark
# q=-1 excluded per instructor guidance: base=q^2 makes it a redundant duplicate
# of the q=+1 classical-limit anchor for the q-cosine kernel specifically (the
# source paper only treats q>0). Kept for the plain-cosine-only qentropy sweeps
# elsewhere (e.g. run_benchmark.py's Table VII), just not for these kernel-swap
# comparisons.
Q_SWEEP = [-0.5, 0.0, 0.5, 1.0]
QA_COUNT = 10
MAX_DOC_TOKENS = 6000      # keep the offline sweep fast across 9 docs
KPIS = ["f1", "mrr", "ndcg", "qcs", "retrieval_token_cost"]


def _pick(m: dict) -> dict:
    out = {k: m.get(k, 0.0) for k in KPIS}
    out["n_chunks"] = m.get("n_chunks", 0)
    return out


def bench_doc(fn: str, domain: str) -> dict:
    import pipeline.s3_entropy as _S3
    _S3._EMB_CACHE.clear()
    EV._CE_CACHE.clear()

    text = _parse_file(fn, open(os.path.join(DOC_DIR, fn), "rb").read())
    words = text.split()
    if len(words) > MAX_DOC_TOKENS:
        text = " ".join(words[:MAX_DOC_TOKENS])
    profile = profile_document(text, {})
    doc_type = profile.get("type", "prose")

    qa = qagen.generate_qa(text, n=QA_COUNT)
    ctx = EV.build_context(text, qa_pairs=qa, backend=BACKEND, judge=False)
    embedder = get_embedder()
    cfg_base = {"n_min": 80, "n_max": 500, "K": 4, "min_chunk_tokens": 20,
                "max_chunk_tokens": 320, "window": 1, "tau_sem": 0.75,
                "embedding_backend": BACKEND}

    # Representative S2 strategy = best pre-entropy cosine-rank (judge off).
    all_s2 = run_all_chunkers(text, doc_type, cfg_base)
    scored = {}
    for s, ch in all_s2.items():
        if ch:
            m = EV.score_run(ch, ctx, judge=False)
            scored[s] = (float(np.mean([m.get(k, 0.0) for k in ("mrr", "ndcg", "srgt", "qcs")])), ch)
    raw = max(scored.values(), key=lambda t: t[0])[1] if scored else []

    res = {"doc": fn, "domain": domain, "n_tokens": len(text.split()),
           "n_queries": len(ctx.queries), "by_q": {}}
    print(f"=== {fn} ({domain})  {len(text.split())} tok, {len(raw)} S2 chunks ===", flush=True)

    for q in Q_SWEEP:
        # S3 boundaries (identical for both arms), then embed those chunks ONCE
        # with the same backend, then run S4 twice — cosine vs q-cosine.
        refined = refine_boundaries([dict(c) for c in raw], {**cfg_base, "q_entropy_param": q})
        emb = embedder.embed([c["text"] for c in refined], BACKEND).tolist()
        row = {}
        for mode in ("cosine", "qcosine"):
            cfg = {**cfg_base, "q_entropy_param": q, "s4_similarity": mode}
            filt = filter_boundaries([dict(c) for c in refined], doc_type, emb, cfg)
            m = EV.score_run(filt, ctx, judge=False)
            row[mode] = _pick(m)
        res["by_q"][f"{q:g}"] = row
        print(f"  q={q:+.1f}  cos: F1={row['cosine']['f1']:.3f} n={row['cosine']['n_chunks']}"
              f"  | qcos: F1={row['qcosine']['f1']:.3f} n={row['qcosine']['n_chunks']}", flush=True)
    return res


def _avg(rows, mode, q, k):
    vals = [r["by_q"][q][mode][k] for r in rows
            if q in r["by_q"] and isinstance(r["by_q"][q][mode].get(k), (int, float))]
    return mean(vals) if vals else 0.0


def emit_table(R: list) -> str:
    def f(x, d=3):
        return ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "--"
    L = ["% ===== S4 similarity-kernel KPI table (avg across corpora) =====",
         "\\begin{table}[t]\\centering",
         "\\caption{Effect of the S4 similarity kernel: classical cosine vs.\\ the "
         "Fitouhi--Bouzeffour $q$-cosine (base $q^{2}$), averaged across all corpora. Upstream "
         "S1--S3 and the embeddings fed to S4 are identical between the two arms, "
         "so each row isolates the kernel. At $q=\\pm1$ the $q$-cosine reduces to "
         "the classical angular cosine.}",
         "\\label{tab:s4kernel}",
         "\\begin{tabular}{@{}llcccccr@{}}\\toprule",
         "$q$ & Kernel & F1$\\uparrow$ & MRR$\\uparrow$ & NDCG$\\uparrow$ & QCS$\\uparrow$ "
         "& \\#chk & Tok.C.$\\downarrow$ \\\\\\midrule"]
    for q in Q_SWEEP:
        qk = f"{q:g}"
        for mode, name in (("cosine", "Cosine"), ("qcosine", "$q$-Cosine")):
            cells = [f(_avg(R, mode, qk, "f1")), f(_avg(R, mode, qk, "mrr")),
                     f(_avg(R, mode, qk, "ndcg")), f(_avg(R, mode, qk, "qcs")),
                     f(_avg(R, mode, qk, "n_chunks"), 1),
                     f(_avg(R, mode, qk, "retrieval_token_cost"), 1)]
            qcol = f"${q:g}$" if mode == "cosine" else ""
            L.append(f"{qcol} & {name} & " + " & ".join(cells) + " \\\\")
        if q != Q_SWEEP[-1]:
            L.append("\\midrule")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L)


def main():
    docs = sorted(os.listdir(DOC_DIR))
    R = []
    for fn in docs:
        path = os.path.join(DOC_DIR, fn)
        if not os.path.isfile(path):
            continue
        try:
            R.append(bench_doc(fn, infer_domain(fn)))
        except Exception as exc:  # keep going; note the failure
            print(f"  !! {fn} failed: {exc}", flush=True)
    with open(os.path.join(HERE, "s4_similarity_results.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    tex = emit_table(R)
    with open(os.path.join(HERE, "s4_similarity_table.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print("\n" + tex)
    print(f"\nWrote s4_similarity_results.json and s4_similarity_table.tex "
          f"({len(R)} documents).")


if __name__ == "__main__":
    main()
