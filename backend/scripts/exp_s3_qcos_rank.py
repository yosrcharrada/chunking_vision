"""
exp_s3_qcos_rank.py — does q-cosine gap RE-RANKING in S3 beat the raw shift?
===========================================================================
Two arms, identical everywhere except the S3 ranking term; S4 held at plain
cosine so the only moving part is which boundaries S3 selects:
  base    : rank gaps by 0.65*(raw shift) + 0.35*LSTM           (current S3)
  s3qcos  : rank gaps by 0.65*(q-cosine-reshaped shift) + 0.35*LSTM
At q=±1 the q-cosine == cosine, so the two arms are identical (sanity check).
Prints per-q mean F1 over ACTIVE docs + head-to-head F1 win/tie/loss.
Run from backend/:  python scripts/exp_s3_qcos_rank.py
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
# both arms run S4 in plain cosine mode so only the S3 ranking differs
ARMS = {"base":   {"s3_qcos_rank": False, "s4_similarity": "cosine"},
        "s3qcos": {"s3_qcos_rank": True,  "s4_similarity": "cosine"}}


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
        out[q] = {}
        for arm, ov in ARMS.items():
            cfg = {**base, "q_entropy_param": q, **ov}
            ref = refine_boundaries([dict(c) for c in raw], cfg)
            ve = emb.embed([c["text"] for c in ref], BACKEND).tolist()
            filt = filter_boundaries([dict(c) for c in ref], dt, ve, cfg)
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
    print("\n q    | meanF1 ACTIVE (base  s3qcos) | s3qcos vs base (W/T/L)  | mean dF1*  mean dTok*")
    print("-" * 88)
    for q in Q_SWEEP:
        fb, fq, df1, dtok, wtl = [], [], [], [], [0, 0, 0]
        for fn in R:
            b, s = R[fn][q]["base"], R[fn][q]["s3qcos"]
            if b[1] == s[1] and abs(b[0] - s[0]) < 1e-4:
                continue  # inactive (identical selection)
            fb.append(b[0]); fq.append(s[0])
            d = s[0] - b[0]; df1.append(d); dtok.append(s[2] - b[2])
            wtl[0 if d > 1e-4 else 2 if d < -1e-4 else 1] += 1
        mb = f"{mean(fb):.3f}" if fb else " --  "
        mq = f"{mean(fq):.3f}" if fq else " --  "
        md = f"{mean(df1):+.3f}" if df1 else "  -- "
        mt = f"{mean(dtok):+.0f}" if dtok else "  -- "
        print(f"{q:+.1f} | {mb:>6} {mq:>6}            | {wtl[0]:>2}/{wtl[1]:>2}/{wtl[2]:<2} (n={sum(wtl)})       | {md:>6}    {mt:>6}")


if __name__ == "__main__":
    main()
