"""
exp_s4_couple.py — does the entropy-gated (boundary-local) q-cosine beat the
plain global q-cosine and plain cosine?  Three arms, identical upstream:
  cos      : classical cosine
  qcos     : global Fitouhi–Bouzeffour q-cosine (s4_qcos_couple=False)
  qcos+S3  : entropy-gated q-cosine, base coupled to S3 boundary_signal (default)
Prints per-q means over ACTIVE documents and head-to-head F1 win/tie/loss.
Run from backend/:  python scripts/exp_s4_couple.py
"""
from __future__ import annotations
import os, sys, warnings
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

DOC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "test_documents")
# q=-1 excluded per instructor guidance (redundant duplicate of q=+1 for the
# q-cosine kernel's base=q^2 convention — see s4_boundary._q_cosine_similarity).
BACKEND, Q_SWEEP, QA_COUNT, MAXTOK = "openai", [-0.5, 0.0, 0.5, 1.0], 10, 6000
ARMS = {"cos":      {"s4_similarity": "cosine"},
        "qcos":     {"s4_similarity": "qcosine"},
        "qwin.5":   {"s4_similarity": "qcosine", "s4_qcos_window": True, "s4_qcos_lambda": 0.5},
        "qwin1":    {"s4_similarity": "qcosine", "s4_qcos_window": True, "s4_qcos_lambda": 1.0}}


def run_doc(fn):
    import pipeline.s3_entropy as _S3
    _S3._EMB_CACHE.clear(); EV._CE_CACHE.clear()
    text = _parse_file(fn, open(os.path.join(DOC_DIR, fn), "rb").read())
    w = text.split()
    if len(w) > MAXTOK:
        text = " ".join(w[:MAXTOK])
    prof = profile_document(text, {}); dt = prof.get("type", "prose")
    ctx = EV.build_context(text, qa_pairs=qagen.generate_qa(text, n=QA_COUNT), backend=BACKEND, judge=False)
    emb = get_embedder()
    base = {"n_min": 80, "n_max": 500, "K": 4, "min_chunk_tokens": 20,
            "max_chunk_tokens": 320, "window": 1, "tau_sem": 0.75, "embedding_backend": BACKEND}
    s2 = run_all_chunkers(text, dt, base)
    sc = {s: (float(np.mean([EV.score_run(ch, ctx, judge=False).get(k, 0) for k in ("mrr", "ndcg", "srgt", "qcs")])), ch)
          for s, ch in s2.items() if ch}
    raw = max(sc.values(), key=lambda t: t[0])[1] if sc else []
    out = {}
    for q in Q_SWEEP:
        ref = refine_boundaries([dict(c) for c in raw], {**base, "q_entropy_param": q})
        ve = emb.embed([c["text"] for c in ref], BACKEND).tolist()
        out[q] = {}
        for arm, ov in ARMS.items():
            filt = filter_boundaries([dict(c) for c in ref], dt, ve, {**base, "q_entropy_param": q, **ov})
            m = EV.score_run(filt, ctx, judge=False)
            out[q][arm] = (m.get("f1", 0.0), len(filt), m.get("retrieval_token_cost", 0.0))
    return out


def main():
    docs = sorted(f for f in os.listdir(DOC_DIR) if os.path.isfile(os.path.join(DOC_DIR, f)))
    R = {}
    for fn in docs:
        try:
            R[fn] = run_doc(fn); print(f"done {fn}", flush=True)
        except Exception as e:
            print(f"!! {fn}: {e}", flush=True)
    arms = list(ARMS)
    print("\n q    | meanF1 over ACTIVE docs (" + "  ".join(arms) + ")"
          " | win arms vs cos (W/T/L)")
    print("-" * 96)
    for q in Q_SWEEP:
        f = {a: [] for a in arms}
        wl = {a: [0, 0, 0] for a in arms if a != "cos"}
        for fn in R:
            vals = {a: R[fn][q][a] for a in arms}
            ncs = {vals[a][1] for a in arms}
            active = not (len(ncs) == 1 and all(abs(vals[a][0] - vals["cos"][0]) < 1e-4 for a in arms))
            if not active:
                continue
            for a in arms:
                f[a].append(vals[a][0])
            for a in wl:
                d = vals[a][0] - vals["cos"][0]
                wl[a][0 if d > 1e-4 else 2 if d < -1e-4 else 1] += 1
        mc = lambda a: f"{mean(f[a]):.3f}" if f[a] else " --  "
        means = "  ".join(f"{mc(a):>5}" for a in arms)
        wls = "   ".join(f"{a}:{wl[a][0]}/{wl[a][1]}/{wl[a][2]}" for a in wl)
        print(f"{q:+.1f} | {means}  | {wls}  (n={len(f['cos'])})")


if __name__ == "__main__":
    main()
