"""
run_benchmark.py — real experimental benchmark for the qEntropy pipeline.

Produces, per document and averaged across documents:
  * baselines            : Fixed-size, RCTS 512/64, Semantic-P95
  * BEFORE entropy (S2)  : raw S2 chunks, scored with the Table-I metrics
  * AFTER entropy  (S3)  : qEntropy at q in the sweep (records D_q, #chunks)
  * AFTER GA       (S7)  : the GA-tuned winner

All scoring uses engine.metrics (the same Table-I metrics as the app), against
auto-generated QA pairs, with the LLM answerability judge ON for reporting.  The
GA's inner fitness runs with the judge OFF (cosine-rank mean) so calibration
stays fast and cheap; only the final winner is judged.

Run from backend/:   python scripts/run_benchmark.py
Outputs:  scripts/benchmark_results.json  and  scripts/benchmark_tables.tex
Requires OPENAI_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from statistics import mean

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from engine import qagen  # noqa: E402
from engine.embeddings import get_embedder  # noqa: E402
from engine.chunking import (  # noqa: E402
    ChunkParams, fixed_size_chunks, rcts_chunks, semantic_percentile_chunks,
    split_sentences, detect_language,
)
from pipeline import evaluation as EV  # noqa: E402
from pipeline.s1_profiler import profile_document  # noqa: E402
from pipeline.s2_chunkers import run_all_chunkers  # noqa: E402
from pipeline.s3_entropy import refine_boundaries  # noqa: E402
from pipeline.s7_rl import run_rl_loop  # noqa: E402
from main import _parse_file  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(os.path.dirname(HERE), "..", "test_documents")
DOCS = [
    ("legalbench.pdf", "Legal"),
    ("PubMedQA.pdf",   "Biomedical"),
    ("general.pdf",    "General"),
]
Q_SWEEP = [-1.0, -0.5, 0.0, 0.5, 1.0]   # 1.0 == Shannon baseline
QA_COUNT = 10
GA_POP, GA_GEN = 4, 2
BACKEND = "openai"
REPORT_KEYS = ["precision", "recall", "f1", "mrr", "ndcg", "ss2fd", "srgt",
               "qcs", "retrieval_token_cost", "answer_correct"]


def _to_dicts(result) -> list:
    """engine.chunking ChunkingResult -> list of chunk dicts for score_run."""
    out = []
    for c in result.chunks:
        out.append({"text": c.text, "tokens": c.tokens})
    if out:
        out[0]["chunking_time_ms"] = getattr(result, "chunking_time_ms", 0.0)
    return out


def _pick(m: dict) -> dict:
    return {k: m.get(k, 0.0) for k in REPORT_KEYS}


def benchmark_doc(path: str, domain: str) -> dict:
    name = os.path.basename(path)
    print(f"\n=== {name}  ({domain}) ===", flush=True)
    text = _parse_file(name, open(path, "rb").read())
    profile = profile_document(text, {})
    doc_type = profile.get("type", "prose")
    print(f"  parsed: {len(text.split())} tokens, type={doc_type}", flush=True)

    print("  generating QA …", flush=True)
    qa = qagen.generate_qa(text, n=QA_COUNT)
    ctx = EV.build_context(text, qa_pairs=qa, backend=BACKEND, judge=True)
    print(f"  QA pairs: {len(qa)} · judge={ctx.judge} · rel_thr={ctx.rel_threshold:.2f}",
          flush=True)

    embedder = get_embedder()
    params = ChunkParams(q=1.0, K=4, min_chunk_tokens=20, max_chunk_tokens=320)
    res = {"doc": name, "domain": domain, "n_tokens": len(text.split()),
           "n_queries": len(ctx.queries), "baselines": {}, "before_entropy": {},
           "q_sweep": {}, "after_ga": {}}

    # ── Baselines ────────────────────────────────────────────────────────────
    print("  baselines …", flush=True)
    for label, chunks in [
        ("Fixed-size",   _to_dicts(fixed_size_chunks(text, embedder, target_tokens=320))),
        ("RCTS 512/64",  _to_dicts(rcts_chunks(text, embedder, 512, 64))),
        ("Semantic-P95", _to_dicts(semantic_percentile_chunks(text, params, embedder, percentile=95))),
    ]:
        m = EV.score_run(chunks, ctx)
        res["baselines"][label] = {**_pick(m), "n_chunks": m.get("n_chunks", len(chunks))}

    # ── Our method: pick the best S2 strategy by pre-entropy cosine-rank ──────
    cfg_base = {"n_min": 80, "n_max": 500, "K": 4, "min_chunk_tokens": 20,
                "max_chunk_tokens": 320, "window": 1, "tau_sem": 0.75,
                "embedding_backend": BACKEND}
    all_s2 = run_all_chunkers(text, doc_type, cfg_base)
    strat_scores = {}
    for s, ch in all_s2.items():
        if ch:
            m = EV.score_run(ch, ctx, judge=False)  # quick pick, no judge
            strat_scores[s] = (float(np.mean([m.get(k, 0.0) for k in ("mrr", "ndcg", "srgt", "qcs")])), ch, m)
    best_strat = max(strat_scores, key=lambda s: strat_scores[s][0])
    raw_chunks = strat_scores[best_strat][1]
    res["strategy"] = best_strat
    print(f"  representative strategy: {best_strat} ({len(raw_chunks)} raw chunks)", flush=True)

    # BEFORE entropy (raw S2), judged
    m = EV.score_run(raw_chunks, ctx)
    res["before_entropy"] = {**_pick(m), "n_chunks": m.get("n_chunks", len(raw_chunks))}

    # AFTER entropy — q sweep
    print("  q-sweep …", flush=True)
    for q in Q_SWEEP:
        cfg = {**cfg_base, "q_entropy_param": q}
        refined = refine_boundaries([dict(c) for c in raw_chunks], cfg)
        m = EV.score_run(refined, ctx)
        dq = refined[0].get("diversity_number") if refined else None
        res["q_sweep"][f"{q:g}"] = {**_pick(m), "n_chunks": len(refined),
                                    "diversity_number": dq}
        print(f"    q={q:>4}: D_q={dq} chunks={len(refined)} "
              f"f1={m.get('f1')} mrr={m.get('mrr')} ac={m.get('answer_correct')}",
              flush=True)

    # AFTER GA (judge OFF inside GA; winner judged here)
    print("  S7 GA …", flush=True)
    ga_cfg = {**cfg_base, "ga_population": GA_POP, "ga_generations": GA_GEN,
              "ga_workers": 1, "judge_answerability": False, "qa_pairs": qa,
              "q_entropy_param": 1.0}
    best_chunks, _hist, final_cfg = run_rl_loop(text, profile, raw_chunks, ga_cfg)
    m = EV.score_run(best_chunks, ctx)
    res["after_ga"] = {**_pick(m), "n_chunks": len(best_chunks),
                       "tuned_q": final_cfg.get("q_entropy_param"),
                       "winner_strategy": final_cfg.get("overall_winner_strategy")}
    print(f"    GA winner: {res['after_ga'].get('winner_strategy')} "
          f"tuned_q={res['after_ga'].get('tuned_q')} f1={m.get('f1')} "
          f"ac={m.get('answer_correct')}", flush=True)
    return res


def main():
    if not qagen.openai_configured():
        print("OPENAI_API_KEY not set — aborting (this benchmark needs the key).")
        sys.exit(1)
    results = []
    for fn, domain in DOCS:
        path = os.path.join(DOC_DIR, fn)
        if not os.path.exists(path):
            print(f"  !! missing {path}; skipping")
            continue
        try:
            results.append(benchmark_doc(path, domain))
        except Exception as exc:
            import traceback
            print(f"  !! {fn} failed: {exc}\n{traceback.format_exc()}")
        # persist incrementally so a late failure never loses earlier work
        with open(os.path.join(HERE, "benchmark_results.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    _emit_latex(results)
    print("\nDONE → scripts/benchmark_results.json, scripts/benchmark_tables.tex")


def _avg(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return mean(vals) if vals else 0.0


def _emit_latex(results):
    """Emit E1 (main comparison), E2 (q-sweep), E3 (before/after entropy/GA)."""
    if not results:
        return
    L = []
    f = lambda x, d=3: ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "--"

    # E1 — averaged main comparison
    L.append("% ── E1: Main comparison (averaged across documents) ──")
    L.append("\\begin{table}[h]\\centering\\caption{Benchmark (avg.\\ across corpora)}")
    L.append("\\begin{tabular}{@{}lcccccc@{}}\\toprule")
    L.append("Method & F1 & MRR & NDCG & SRGT & QCS & Ans.\\,Corr. \\\\\\midrule")
    methods = {}
    for label in ("Fixed-size", "RCTS 512/64", "Semantic-P95"):
        rows = [r["baselines"][label] for r in results if label in r["baselines"]]
        methods[label] = rows
    methods["Ours (Shannon, q=1)"] = [r["q_sweep"].get("1") for r in results if r.get("q_sweep")]
    methods["Ours (Tsallis, GA)"] = [r["after_ga"] for r in results if r.get("after_ga")]
    for label, rows in methods.items():
        rows = [x for x in rows if x]
        nm = f"\\textbf{{{label}}}" if "GA" in label else label
        L.append(f"{nm} & " + " & ".join(
            f(_avg(rows, k)) for k in ("f1", "mrr", "ndcg", "srgt", "qcs", "answer_correct")
        ) + " \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}\n")

    # E2 — q sweep (averaged)
    L.append("% ── E2: q-sweep (averaged across documents) ──")
    L.append("\\begin{table}[h]\\centering\\caption{Effect of Tsallis $q$ (avg.)}")
    L.append("\\begin{tabular}{@{}lcccccc@{}}\\toprule")
    L.append("$q$ & $D_q$ & \\#chunks & F1 & MRR & NDCG & Ans.\\,Corr. \\\\\\midrule")
    for q in Q_SWEEP:
        rows = [r["q_sweep"].get(f"{q:g}") for r in results if r.get("q_sweep")]
        rows = [x for x in rows if x]
        tag = " (Shannon)" if q == 1.0 else ""
        L.append(f"${q:g}${tag} & " + " & ".join([
            f(_avg(rows, "diversity_number"), 2), f(_avg(rows, "n_chunks"), 1),
            f(_avg(rows, "f1")), f(_avg(rows, "mrr")), f(_avg(rows, "ndcg")),
            f(_avg(rows, "answer_correct"))]) + " \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}\n")

    # E3 — before entropy vs after entropy (best q) vs after GA, per document
    L.append("% ── E3: Before entropy (S2) vs After entropy (S3) vs After GA (S7) ──")
    L.append("\\begin{table}[h]\\centering\\caption{Pipeline-stage effect (per document)}")
    L.append("\\begin{tabular}{@{}llccc@{}}\\toprule")
    L.append("Document & Stage & F1 & MRR & Ans.\\,Corr. \\\\\\midrule")
    for r in results:
        be = r.get("before_entropy", {})
        # best q row by f1
        qs = [(k, v) for k, v in r.get("q_sweep", {}).items()]
        best_q = max(qs, key=lambda kv: kv[1].get("f1", 0.0))[1] if qs else {}
        ga = r.get("after_ga", {})
        doc = r["doc"].replace("_", "\\_")
        L.append(f"\\multirow{{3}}{{*}}{{{doc}}} & Before entropy (S2) & "
                 f"{f(be.get('f1'))} & {f(be.get('mrr'))} & {f(be.get('answer_correct'))} \\\\")
        L.append(f" & After entropy (best $q$) & "
                 f"{f(best_q.get('f1'))} & {f(best_q.get('mrr'))} & {f(best_q.get('answer_correct'))} \\\\")
        L.append(f" & After GA (S7) & "
                 f"{f(ga.get('f1'))} & {f(ga.get('mrr'))} & {f(ga.get('answer_correct'))} \\\\\\midrule")
    L.append("\\bottomrule\\end{tabular}\\end{table}")

    with open(os.path.join(HERE, "benchmark_tables.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
