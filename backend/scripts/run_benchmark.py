"""
run_benchmark.py — professional benchmark for the qEntropy pipeline.

Auto-discovers every document in test_documents/ (PDF, text, code; PDFs detected
by magic bytes so extension-less files still parse), infers a domain, and for
each document records, with the Table-I metrics + LLM answer-correctness:

  baselines       : Fixed-size, RCTS 512/64, Semantic-P95
  before_entropy  : raw S2 chunks  (the "before q" reference)
  q_sweep         : qEntropy at each q  (records D_q, #chunks; best q per doc)
  after_ga        : GA-discovered q*  (GA fitness includes answer-correctness)
  ablation        : Full, -LSTM, -Structure, Shannon(q=1), -S4 filter

Emits scripts/benchmark_results.json (everything) and scripts/benchmark_tables.tex
(Tables V-IX + a per-document best-q table).

Run from backend/:  python scripts/run_benchmark.py
Requires OPENAI_API_KEY.  NOTE: judge-on GA across many documents is slow and
costs API calls; results are saved after every document.
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
from engine.chunking import (  # noqa: E402
    ChunkParams, fixed_size_chunks, rcts_chunks, semantic_percentile_chunks,
)
from pipeline import evaluation as EV  # noqa: E402
from pipeline.s1_profiler import profile_document  # noqa: E402
from pipeline.s2_chunkers import run_all_chunkers  # noqa: E402
from pipeline.s3_entropy import refine_boundaries  # noqa: E402
from pipeline.s4_boundary import filter_boundaries  # noqa: E402
from pipeline.s7_rl import run_rl_loop  # noqa: E402
from main import _parse_file  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(os.path.dirname(HERE), "..", "test_documents")
Q_SWEEP = [-1.0, -0.5, 0.0, 0.5, 1.0]
QA_COUNT = 10
GA_POP, GA_GEN = 4, 2
BACKEND = "openai"
JUDGE = True
MAX_DOC_TOKENS = 12000   # cap huge documents (memory/time); truncation is noted
REPORT_KEYS = ["precision", "recall", "f1", "mrr", "ndcg", "ss2fd", "srgt",
               "qcs", "retrieval_token_cost", "answer_correct"]
ABLATIONS = {
    "Full":          ({}, True),
    "-- LSTM":       ({"use_lstm": False}, True),
    "-- Structure":  ({"use_structure": False}, True),
    "Shannon (q=1)": ({"q_entropy_param": 1.0}, True),
    "-- S4 filter":  ({}, False),
}


def infer_domain(fn: str) -> str:
    n = fn.lower()
    if n.rsplit(".", 1)[-1] in ("py", "js", "java", "c", "cpp", "go", "ts"):
        return "Code"
    if any(k in n for k in ("irpp", "cdpf", "convention", "fiscal", "legal", "droit", "impot")):
        return "Legal"
    if any(k in n for k in ("pubmed", "cancer", "plasticit", "poumon", "clinical", "medic", "bio")):
        return "Medical"
    return "General"


def _to_dicts(result) -> list:
    out = [{"text": c.text, "tokens": c.tokens} for c in result.chunks]
    if out:
        out[0]["chunking_time_ms"] = getattr(result, "chunking_time_ms", 0.0)
    return out


def _pick(m: dict) -> dict:
    return {k: m.get(k, 0.0) for k in REPORT_KEYS}


def benchmark_doc(fn: str, domain: str) -> dict:
    # Free the module-level embedding caches between documents so memory does not
    # grow unbounded across the corpus (a 592-chunk doc on top of two large ones
    # segfaulted the previous run).
    import pipeline.s3_entropy as _S3
    _S3._EMB_CACHE.clear()
    EV._CE_CACHE.clear()

    path = os.path.join(DOC_DIR, fn)
    print(f"\n=== {fn}  ({domain}) ===", flush=True)
    text = _parse_file(fn, open(path, "rb").read())
    words = text.split()
    if len(words) > MAX_DOC_TOKENS:
        text = " ".join(words[:MAX_DOC_TOKENS])
        print(f"  (truncated from {len(words)} to {MAX_DOC_TOKENS} tokens)", flush=True)
    profile = profile_document(text, {})
    doc_type = profile.get("type", "prose")
    print(f"  parsed: {len(text.split())} tokens, type={doc_type}", flush=True)

    qa = qagen.generate_qa(text, n=QA_COUNT)
    ctx = EV.build_context(text, qa_pairs=qa, backend=BACKEND, judge=JUDGE)
    print(f"  QA={len(qa)} judge={ctx.judge} rel_thr={ctx.rel_threshold:.2f}", flush=True)

    embedder = get_embedder()
    params = ChunkParams(q=1.0, K=4, min_chunk_tokens=20, max_chunk_tokens=320)
    cfg_base = {"n_min": 80, "n_max": 500, "K": 4, "min_chunk_tokens": 20,
                "max_chunk_tokens": 320, "window": 1, "tau_sem": 0.75,
                "embedding_backend": BACKEND}
    res = {"doc": fn, "domain": domain, "n_tokens": len(text.split()),
           "n_queries": len(ctx.queries), "baselines": {}, "before_entropy": {},
           "q_sweep": {}, "after_ga": {}, "ablation": {}, "s4_sim": {}}

    # ── Baselines (judged) ───────────────────────────────────────────────────
    print("  baselines …", flush=True)
    for label, chunks in [
        ("Fixed-size",   _to_dicts(fixed_size_chunks(text, embedder, target_tokens=320))),
        ("RCTS 512/64",  _to_dicts(rcts_chunks(text, embedder, 512, 64))),
        ("Semantic-P95", _to_dicts(semantic_percentile_chunks(text, params, embedder, percentile=95))),
    ]:
        m = EV.score_run(chunks, ctx)
        res["baselines"][label] = {**_pick(m), "n_chunks": m.get("n_chunks", len(chunks))}

    # ── Representative S2 strategy (best pre-entropy cosine-rank, judge off) ──
    all_s2 = run_all_chunkers(text, doc_type, cfg_base)
    scored = {}
    for s, ch in all_s2.items():
        if ch:
            m = EV.score_run(ch, ctx, judge=False)
            scored[s] = (float(np.mean([m.get(k, 0.0) for k in ("mrr", "ndcg", "srgt", "qcs")])), ch)
    best_strat = max(scored, key=lambda s: scored[s][0])
    raw = scored[best_strat][1]
    res["strategy"] = best_strat
    print(f"  strategy={best_strat} ({len(raw)} raw chunks)", flush=True)

    # before entropy (judged)
    res["before_entropy"] = {**_pick(EV.score_run(raw, ctx)), "n_chunks": len(raw)}

    # ── q-sweep (judged) ─────────────────────────────────────────────────────
    print("  q-sweep …", flush=True)
    for q in Q_SWEEP:
        ref = refine_boundaries([dict(c) for c in raw], {**cfg_base, "q_entropy_param": q})
        m = EV.score_run(ref, ctx)
        res["q_sweep"][f"{q:g}"] = {**_pick(m), "n_chunks": len(ref),
                                    "diversity_number": (ref[0].get("diversity_number") if ref else None)}
        print(f"    q={q:>4}: D_q={res['q_sweep'][f'{q:g}']['diversity_number']} "
              f"f1={m.get('f1')} ac={m.get('answer_correct')}", flush=True)

    # ── S7 GA with answer-correctness in the fitness (judge ON) ──────────────
    print("  S7 GA (judge on) …", flush=True)
    ga_cfg = {**cfg_base, "ga_population": GA_POP, "ga_generations": GA_GEN,
              "ga_workers": 1, "judge_answerability": True, "qa_pairs": qa,
              "rel_threshold": ctx.rel_threshold, "q_entropy_param": 1.0}
    best_chunks, _h, final_cfg = run_rl_loop(text, profile, raw, ga_cfg)
    tq = final_cfg.get("q_entropy_param", 1.0)
    m = EV.score_run(best_chunks, ctx)
    res["after_ga"] = {**_pick(m), "n_chunks": len(best_chunks), "tuned_q": tq,
                       "winner_strategy": final_cfg.get("overall_winner_strategy")}
    print(f"    GA q*={tq} winner={res['after_ga']['winner_strategy']} "
          f"f1={m.get('f1')} ac={m.get('answer_correct')}", flush=True)

    # ── Ablation (judged), at the GA-tuned q ─────────────────────────────────
    print("  ablation …", flush=True)
    for label, (override, apply_s4) in ABLATIONS.items():
        cfg = {**cfg_base, "q_entropy_param": tq, **override}
        ref = refine_boundaries([dict(c) for c in raw], cfg)
        if apply_s4:
            ref = filter_boundaries(ref, doc_type, [], cfg)
        m = EV.score_run(ref, ctx)
        res["ablation"][label] = {**_pick(m), "n_chunks": len(ref)}

    # ── S4 similarity kernel A/B (cosine vs q-cosine), at the GA-tuned q ──────
    # Upstream S1–S3 and the S4 input embeddings are identical between arms, so
    # the only difference is the kernel.  At q=±1 the q-cosine == cosine.
    print("  S4 similarity (cosine vs q-cosine) …", flush=True)
    ref_s4 = refine_boundaries([dict(c) for c in raw], {**cfg_base, "q_entropy_param": tq})
    s4_emb = embedder.embed([c["text"] for c in ref_s4], BACKEND).tolist()
    for mode in ("cosine", "qcosine"):
        cfg = {**cfg_base, "q_entropy_param": tq, "s4_similarity": mode}
        filt = filter_boundaries([dict(c) for c in ref_s4], doc_type, s4_emb, cfg)
        m = EV.score_run(filt, ctx)
        res["s4_sim"][mode] = {**_pick(m), "n_chunks": len(filt), "q": tq}
    return res


# ── LaTeX emission ───────────────────────────────────────────────────────────
def _avg(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return mean(v) if v else 0.0


def _f(x, d=3):
    return ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "--"


def _emit_latex(R):
    if not R:
        return
    cols = ["precision", "recall", "f1", "mrr", "ndcg", "ss2fd", "srgt", "qcs",
            "retrieval_token_cost", "answer_correct"]
    hdr = "P & R & F1 & MRR & NDCG & ss2fd & SRGT & QCS & Tok & AC"
    def rowfmt(m):
        return " & ".join(_f(m.get(c), 1 if c == "retrieval_token_cost" else 3) for c in cols)
    L = []

    # TABLE V — classical baselines (avg)
    L += ["% TABLE V: Classical Baselines (avg across corpora)",
          "\\begin{table*}[t]\\centering\\caption{Classical Baselines (Average Across All Corpora)}",
          "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\\toprule",
          "Method & " + hdr + " \\\\\\midrule"]
    for label in ("Fixed-size", "RCTS 512/64", "Semantic-P95"):
        rows = [r["baselines"][label] for r in R if label in r.get("baselines", {})]
        L.append(f"{label} & " + rowfmt({c: _avg(rows, c) for c in cols}) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}\n"]

    # TABLE VI — entropy-enhanced (avg): Shannon, best-fixed-q (oracle), GA
    def best_fixed(r):
        qs = list(r.get("q_sweep", {}).values())
        return max(qs, key=lambda m: m.get("f1", 0.0)) if qs else {}
    rows_shannon = [r["q_sweep"].get("1") for r in R if r.get("q_sweep", {}).get("1")]
    rows_bestfx = [best_fixed(r) for r in R if r.get("q_sweep")]
    rows_ga = [r["after_ga"] for r in R if r.get("after_ga")]
    L += ["% TABLE VI: Entropy-Enhanced Results (avg across corpora)",
          "\\begin{table*}[t]\\centering\\caption{Entropy-Enhanced Results (Average Across All Corpora)}",
          "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\\toprule",
          "Method & " + hdr + " \\\\\\midrule"]
    for label, rows in [("Shannon ($q{=}1$)", rows_shannon),
                        ("Tsallis (best fixed $q$)", rows_bestfx),
                        ("\\textbf{Tsallis (GA $q^\\*$)}", rows_ga)]:
        rows = [x for x in rows if x]
        L.append(f"{label} & " + rowfmt({c: _avg(rows, c) for c in cols}) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}\n"]

    # TABLE VII — effect of q (avg)
    L += ["% TABLE VII: Effect of q on Retrieval Metrics (avg)",
          "\\begin{table}[t]\\centering\\caption{Effect of $q$ on Retrieval Metrics (Avg.)}",
          "\\begin{tabular}{@{}lcccccccc@{}}\\toprule",
          "$q$ & $D_q$ & \\#chk & F1 & MRR & NDCG & QCS & Tok & AC \\\\\\midrule"]
    for q in Q_SWEEP:
        rows = [r["q_sweep"].get(f"{q:g}") for r in R if r.get("q_sweep")]
        rows = [x for x in rows if x]
        tag = " (Shannon)" if q == 1.0 else ""
        L.append(f"${q:g}${tag} & " + " & ".join([
            _f(_avg(rows, "diversity_number"), 1), _f(_avg(rows, "n_chunks"), 1),
            _f(_avg(rows, "f1")), _f(_avg(rows, "mrr")), _f(_avg(rows, "ndcg")),
            _f(_avg(rows, "qcs")), _f(_avg(rows, "retrieval_token_cost"), 1),
            _f(_avg(rows, "answer_correct"))]) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}\n"]

    # TABLE VIII — GA-discovered q* by domain
    by_dom = defaultdict(list)
    for r in R:
        if r.get("after_ga"):
            by_dom[r["domain"]].append(r)
    L += ["% TABLE VIII: GA-Discovered Optimal q* by Domain",
          "\\begin{table}[t]\\centering\\caption{GA-Discovered Optimal $q^\\*$ by Domain}",
          "\\begin{tabular}{@{}lcccccc@{}}\\toprule",
          "Domain & $q^\\*$ (mean) & F1 & MRR & NDCG & Tok & AC \\\\\\midrule"]
    for dom in sorted(by_dom):
        rows = [r["after_ga"] for r in by_dom[dom]]
        L.append(f"{dom} & " + " & ".join([
            _f(_avg(rows, "tuned_q"), 2), _f(_avg(rows, "f1")), _f(_avg(rows, "mrr")),
            _f(_avg(rows, "ndcg")), _f(_avg(rows, "retrieval_token_cost"), 1),
            _f(_avg(rows, "answer_correct"))]) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}\n"]

    # TABLE IX — ablation (avg)
    L += ["% TABLE IX: Ablation (avg across corpora)",
          "\\begin{table}[t]\\centering\\caption{Ablation (Average Across All Corpora)}",
          "\\begin{tabular}{@{}lccccc@{}}\\toprule",
          "Configuration & F1 & MRR & NDCG & Tok & AC \\\\\\midrule"]
    for label in ABLATIONS:
        rows = [r["ablation"][label] for r in R if label in r.get("ablation", {})]
        L.append(f"{label} & " + " & ".join([
            _f(_avg(rows, "f1")), _f(_avg(rows, "mrr")), _f(_avg(rows, "ndcg")),
            _f(_avg(rows, "retrieval_token_cost"), 1),
            _f(_avg(rows, "answer_correct"))]) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}\n"]

    # PER-DOCUMENT q table (best q in F1 bolded; + before-entropy and GA columns)
    L += ["% Per-document F1 across q (best fixed q in bold) + before-entropy + GA",
          "\\begin{table*}[t]\\centering\\caption{Per-Document F1 across $q$ (best fixed $q$ in bold)}",
          "\\begin{tabular}{@{}ll" + "c" * len(Q_SWEEP) + "cc@{}}\\toprule",
          "Document & Domain & " + " & ".join(f"$q={q:g}$" for q in Q_SWEEP)
          + " & Before & GA($q^\\*$) \\\\\\midrule"]
    for r in R:
        qs = r.get("q_sweep", {})
        f1s = {f"{q:g}": (qs.get(f"{q:g}", {}) or {}).get("f1") for q in Q_SWEEP}
        best = max((k for k in f1s if f1s[k] is not None), key=lambda k: f1s[k], default=None)
        cells = []
        for q in Q_SWEEP:
            v = f1s[f"{q:g}"]
            s = _f(v)
            if f"{q:g}" == best:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        be = _f((r.get("before_entropy") or {}).get("f1"))
        ga = r.get("after_ga", {})
        gacell = f"{_f(ga.get('f1'))} ({_f(ga.get('tuned_q'),2)})"
        doc = r["doc"].replace("_", "\\_")
        L.append(f"{doc} & {r['domain']} & " + " & ".join(cells) + f" & {be} & {gacell} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}\n"]

    # TABLE — S4 similarity kernel (cosine vs q-cosine), avg across corpora
    s4_rows = {m: [r["s4_sim"][m] for r in R if m in r.get("s4_sim", {})]
               for m in ("cosine", "qcosine")}
    if s4_rows["cosine"]:
        L += ["% S4 similarity kernel: classical cosine vs q-cosine (judged, at GA q*)",
              "\\begin{table}[t]\\centering\\caption{S4 Similarity Kernel: Cosine vs.\\ "
              "Fitouhi--Bouzeffour $q$-Cosine (judged, at the GA-tuned $q^\\*$, averaged across corpora)}",
              "\\begin{tabular}{@{}lcccccc@{}}\\toprule",
              "Kernel & F1 & MRR & NDCG & QCS & \\#chk & AC \\\\\\midrule"]
        for m, name in (("cosine", "Cosine"), ("qcosine", "$q$-Cosine")):
            rows = s4_rows[m]
            L.append(f"{name} & " + " & ".join([
                _f(_avg(rows, "f1")), _f(_avg(rows, "mrr")), _f(_avg(rows, "ndcg")),
                _f(_avg(rows, "qcs")), _f(_avg(rows, "n_chunks"), 1),
                _f(_avg(rows, "answer_correct"))]) + " \\\\")
        L += ["\\bottomrule\\end{tabular}\\end{table}"]

    with open(os.path.join(HERE, "benchmark_tables.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


RESULTS_PATH = os.path.join(HERE, "benchmark_results.json")


def _load_results():
    if os.path.exists(RESULTS_PATH):
        try:
            return json.load(open(RESULTS_PATH, encoding="utf-8"))
        except Exception:
            return []
    return []


def process_one(fn, domain):
    """Benchmark a single document in THIS (fresh) process, then append + save."""
    res = benchmark_doc(fn, domain)
    allr = [r for r in _load_results() if r.get("doc") != fn] + [res]
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(allr, fh, indent=2)
    _emit_latex(allr)
    print(f"  SAVED {fn}  ({len(allr)} docs total)", flush=True)


def main():
    import argparse
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc")
    ap.add_argument("--domain")
    args = ap.parse_args()

    if not qagen.openai_configured():
        print("OPENAI_API_KEY not set — aborting."); sys.exit(1)

    # Child mode: process exactly one document, then exit (fresh process per doc
    # so a segfault in one document can never kill the whole benchmark, and torch/
    # numpy state never accumulates across documents).
    if args.doc:
        process_one(args.doc, args.domain or infer_domain(args.doc))
        return

    # Orchestrator mode: one fresh subprocess per undone document.
    docs = [(fn, infer_domain(fn)) for fn in sorted(os.listdir(DOC_DIR))
            if os.path.isfile(os.path.join(DOC_DIR, fn))]
    print(f"Discovered {len(docs)} documents:")
    for fn, d in docs:
        print(f"  {d:9s} {fn}")
    done = {r["doc"] for r in _load_results()}
    if done:
        print(f"Resuming — {len(done)} already done: {sorted(done)}")

    for fn, domain in docs:
        if fn in done:
            print(f"  skip (already done): {fn}", flush=True)
            continue
        print(f"\n>>> {fn}  (fresh subprocess) …", flush=True)
        rc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
                             "--doc", fn, "--domain", domain]).returncode
        if rc != 0:
            print(f"  !! {fn} crashed (rc={rc}) — skipping and continuing.", flush=True)

    allr = _load_results()
    _emit_latex(allr)
    print(f"\nDONE → {len(allr)}/{len(docs)} docs → benchmark_results.json, benchmark_tables.tex")


if __name__ == "__main__":
    main()
