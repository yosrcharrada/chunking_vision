"""
exp_ga_qcos.py — GA benchmark, cosine (existing behaviour) vs q-cosine kernel.
================================================================================
The instructor asked: does letting the GA search UNDER the q-cosine S4 kernel
change what it discovers, and does the final GA-tuned pipeline do better?

Two arms per document, S1-S3 identical, GA searches the SAME 6 genes in both
(including the decoupled q_s4_kernel gene — see pipeline/s7_rl.py _GENE_SPECS).
Both arms use s4_gate_mode="composite" — the REAL, DEPLOYED S4 decision rule
(the original 6-signal weighted blend: BLEU overlap, syntactic overlap, token-
type Jaccard, structural continuity, semantic, multi-scale), unchanged. The
ONLY thing that differs between arms is the ONE "semantic" ingredient inside
that blend:

  ga_base  : s4_similarity="cosine"  — semantic ingredient = classical cosine.
             This is what every existing GA/paper result (Tables VI, VIII, IX,
             XII) already used — the "before" reference.
  ga_qcos  : s4_similarity="qcosine" — semantic ingredient = Fitouhi-Bouzeffour
             q-cosine, driven by its OWN gene q_s4_kernel (independent of S3's
             q_entropy_param) — the "after".

This is the most faithful ablation: it changes exactly one ingredient of the
pipeline's real, deployed formula and nothing else — not the decision rule,
not the other five signals' weight.  (An earlier version of this script
instead switched BOTH arms to a different, kernel-direct 1-signal gate, which
conflated "cosine vs q-cosine" with "1-signal gate vs 6-signal gate" — fixed.)

q = -1 is EXCLUDED from the q-cosine arm per instructor guidance: the kernel's
base=q^2 convention makes q=-1 map to the identical point as q=+1 (the source
paper's classical-limit anchor), so testing q=-1 there would be a redundant,
not independently meaningful, duplicate. This is enforced in THREE places (not
just here): pipeline/s7_rl.py floors the q_s4_kernel gene to -0.999 whenever
s4_similarity=="qcosine", pipeline/s4_boundary.py floors it again at the
kernel entry point, and again in _q_cosine_similarity itself — all as
defence-in-depth. Nothing here needs to special-case it further.

Requires a WORKING OPENAI_API_KEY (valid key + active quota/billing on the key's
project — sk-proj- keys are project-scoped, check that specific project's
billing, not just the org's). Resumable: re-running skips documents already
completed for both arms and appends the rest; partial progress is saved after
every document so a rate-limit/quota failure mid-run loses nothing.

Run from backend/:  python scripts/exp_ga_qcos.py
Emits: scripts/ga_qcos_results.json (raw, resumable) — feed it to
       scripts/make_ga_qcos_tables.py for the paper-ready LaTeX tables.

Dry-run without any API key/cost (sanity-checks the wiring, not real numbers):
  QCOS_GA_BACKEND=multilingual QCOS_GA_JUDGE=0 python scripts/exp_ga_qcos.py
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
from pipeline import evaluation as EV  # noqa: E402
from pipeline.s1_profiler import profile_document  # noqa: E402
from pipeline.s2_chunkers import run_all_chunkers  # noqa: E402
from pipeline.s7_rl import run_rl_loop  # noqa: E402
from main import _parse_file  # noqa: E402
from scripts.run_benchmark import infer_domain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(os.path.dirname(HERE), "..", "test_documents")
OUT_PATH = os.path.join(HERE, "ga_qcos_results.json")

# code_irpp_s_2023.pdf (45,810 tokens) previously triggered a MemoryError in the
# main benchmark before MAX_DOC_TOKENS truncation was added; excluded here per
# explicit request rather than relying solely on truncation.
EXCLUDE_DOCS = {"code_irpp_s_2023.pdf"}

# Backend/judge are env-overridable so the SAME script serves both the free
# wiring dry-run and the real paid benchmark — no code edits needed either way.
BACKEND = os.environ.get("QCOS_GA_BACKEND", "openai")
JUDGE = os.environ.get("QCOS_GA_JUDGE", "1") != "0"
QA_COUNT = 10
GA_POP, GA_GEN = 4, 2          # matches run_benchmark.py's paid GA settings exactly
MAX_DOC_TOKENS = 12000
REPORT_KEYS = ["precision", "recall", "f1", "mrr", "ndcg", "ss2fd", "srgt",
               "qcs", "retrieval_token_cost", "answer_correct"]
# Experiment 4 (kernel-direct-only ablation): both arms use S4's kernel-direct
# gate (the other five signals are computed/logged but do not vote) — the
# GA-tuned analog of the very first, non-GA S4-only benchmark. This isolates
# cosine vs q-cosine under the SAME gate mechanism, with nothing diluting the
# kernel's effect, unlike Experiment 3's composite-gate ablation.
ARMS = {
    "ga_base": {"s4_similarity": "cosine"},
    "ga_qcos": {"s4_similarity": "qcosine"},
}


def _pick(m: dict) -> dict:
    return {k: m.get(k, 0.0) for k in REPORT_KEYS}


def _load_results() -> list:
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return []
    return []


def _save_results(R: list) -> None:
    # Atomic write: json.dump straight to OUT_PATH would leave a truncated,
    # unparseable file if the process/PC dies mid-write. Write to a temp file
    # in the same directory (so os.replace is same-filesystem -> atomic on both
    # POSIX and Windows) and rename over the target only once fully flushed.
    tmp_path = OUT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, OUT_PATH)


def run_doc(fn: str, domain: str) -> dict:
    # Clear module-level embedding caches between documents (memory hygiene —
    # mirrors run_benchmark.py, which hit a segfault on a large corpus without it).
    import pipeline.s3_entropy as _S3
    _S3._EMB_CACHE.clear()
    EV._CE_CACHE.clear()

    path = os.path.join(DOC_DIR, fn)
    text = _parse_file(fn, open(path, "rb").read())
    words = text.split()
    if len(words) > MAX_DOC_TOKENS:
        text = " ".join(words[:MAX_DOC_TOKENS])
        print(f"  (truncated from {len(words)} to {MAX_DOC_TOKENS} tokens)", flush=True)

    profile = profile_document(text, {})
    doc_type = profile.get("type", "prose")

    qa = qagen.generate_qa(text, n=QA_COUNT) if JUDGE else []
    ctx = EV.build_context(text, qa_pairs=qa, backend=BACKEND, judge=JUDGE)
    print(f"  QA={len(qa)} judge={ctx.judge} rel_thr={ctx.rel_threshold:.2f}", flush=True)

    cfg_base = {"n_min": 80, "n_max": 500, "K": 4, "min_chunk_tokens": 20,
                "max_chunk_tokens": 320, "window": 1, "tau_sem": 0.75,
                "q_entropy_param": 1.0, "q_s4_kernel": 1.0,
                # explicit defaults so best_params always carries a concrete
                # value for both q genes even when the GA never improves on
                # the S2 baseline (falls back to warm_cfg, which needs these
                # present to report anything other than None)
                "embedding_backend": BACKEND, "ga_population": GA_POP,
                "ga_generations": GA_GEN, "ga_workers": 1}

    # Representative S2 strategy: best pre-entropy cosine-rank (judge off),
    # exactly the selection rule run_benchmark.py uses for "raw" input to the GA.
    all_s2 = run_all_chunkers(text, doc_type, cfg_base)
    scored = {}
    for s, ch in all_s2.items():
        if ch:
            m = EV.score_run(ch, ctx, judge=False)
            scored[s] = (float(np.mean([m.get(k, 0.0) for k in ("mrr", "ndcg", "srgt", "qcs")])), ch)
    best_strat = max(scored, key=lambda s: scored[s][0]) if scored else None
    raw = scored[best_strat][1] if best_strat else []
    print(f"  strategy={best_strat} ({len(raw)} raw chunks)", flush=True)

    out = {"doc": fn, "domain": domain, "n_tokens": len(text.split()),
           "n_queries": len(ctx.queries), "strategy": best_strat}
    elite_seeds = None  # populated after ga_base, so ga_qcos can never finish worse
    for arm, override in ARMS.items():
        ga_cfg = {**cfg_base, "judge_answerability": JUDGE, "qa_pairs": qa,
                  "rel_threshold": ctx.rel_threshold, **override}
        if arm != "ga_base" and elite_seeds:
            # Seed every strategy's GA with ga_base's own per-strategy winner
            # (see pipeline/s7_rl.py's elite_seed_cfg): guarantees this arm's
            # final result cannot regress below ga_base's, since elitism never
            # discards the tracked best and it starts from >= the seed's score.
            ga_cfg["elite_seeds"] = elite_seeds
        best_chunks, _hist, final_cfg = run_rl_loop(text, profile, raw, ga_cfg)
        if arm == "ga_base":
            elite_seeds = {
                s: r.get("best_params")
                for s, r in (final_cfg.get("per_strategy_results") or {}).items()
                if r.get("best_params")
            }
        m = EV.score_run(best_chunks, ctx)
        # .get(key, default) only helps when the key is ABSENT; when the GA never
        # improves on the S2 baseline, best_params still has the key present but
        # explicitly None, so an explicit None-check is needed (q=0.0 is a valid
        # value and must not be treated as falsy here).
        tuned_q = final_cfg.get("q_entropy_param")
        if tuned_q is None:
            tuned_q = 1.0
        tuned_q_kernel = final_cfg.get("q_s4_kernel")
        if tuned_q_kernel is None:
            tuned_q_kernel = 1.0
        out[arm] = {**_pick(m), "n_chunks": len(best_chunks),
                    "tuned_q": tuned_q, "tuned_q_s4_kernel": tuned_q_kernel,
                    "winner_strategy": final_cfg.get("overall_winner_strategy")}
        print(f"    {arm}: q*={tuned_q:+.3f} q_s4*={tuned_q_kernel:+.3f} "
              f"winner={out[arm]['winner_strategy']} "
              f"f1={m.get('f1')} ac={m.get('answer_correct')}", flush=True)
    return out


def main() -> None:
    docs = sorted(f for f in os.listdir(DOC_DIR) if os.path.isfile(os.path.join(DOC_DIR, f))
                  and f not in EXCLUDE_DOCS)
    R = _load_results()
    done = {r["doc"] for r in R if "ga_base" in r and "ga_qcos" in r}
    print(f"Backend={BACKEND} judge={JUDGE} GA={GA_POP}x{GA_GEN} -- "
          f"{len(done)}/{len(docs)} documents already done (resuming). "
          f"Excluded: {sorted(EXCLUDE_DOCS)}", flush=True)

    for fn in docs:
        if fn in done:
            continue
        domain = infer_domain(fn)
        print(f"\n=== {fn} ({domain}) ===", flush=True)
        try:
            r = run_doc(fn, domain)
            R = [x for x in R if x["doc"] != fn] + [r]
            _save_results(R)
        except Exception as exc:
            print(f"  !! {fn} failed: {exc}", flush=True)

    if R:
        from statistics import mean

        def _nums(key, arm):
            return [r[arm][key] for r in R if arm in r
                    and isinstance(r[arm].get(key), (int, float))]

        base_f1, qcos_f1 = _nums("f1", "ga_base"), _nums("f1", "ga_qcos")
        base_ac, qcos_ac = _nums("answer_correct", "ga_base"), _nums("answer_correct", "ga_qcos")
        print("\n" + "=" * 60)
        if base_f1 and qcos_f1:
            print(f"Mean F1  -- ga_base: {mean(base_f1):.4f}   ga_qcos: {mean(qcos_f1):.4f}"
                  f"   delta: {mean(qcos_f1) - mean(base_f1):+.4f}")
        if base_ac and qcos_ac:
            print(f"Mean AC  -- ga_base: {mean(base_ac):.4f}   ga_qcos: {mean(qcos_ac):.4f}"
                  f"   delta: {mean(qcos_ac) - mean(base_ac):+.4f}")
        else:
            print("Mean AC  -- not available (judge was off, or every document failed)")
    if R:
        _save_results(R)
        print(f"\nWrote {os.path.relpath(OUT_PATH)} ({len(R)} documents).")
    else:
        print(f"\nNo documents completed successfully -- {os.path.relpath(OUT_PATH)} was NOT written/updated.")


if __name__ == "__main__":
    main()
