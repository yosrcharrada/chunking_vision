"""
make_s4_tables.py — clearer reporting for the S4 similarity-kernel A/B.
=======================================================================
Reads scripts/s4_similarity_results.json (produced by bench_s4_similarity.py)
and emits THREE views, because a single corpus average hides the effect:

  1. Per-document, side by side at a representative q (default q=0): you can see
     for EACH document whether the q-cosine changed the chunking and the metrics.
  2. Head-to-head activity across q: how many documents the kernel actually
     changed, and of those, F1 win / tie / loss + mean deltas (active docs only).
  3. Corpus average reported TWO ways — over all docs vs. over only the docs
     where the kernel was active — so the dilution is explicit.

Console output is a readable ASCII version; scripts/s4_similarity_tables.tex
holds the paper-ready LaTeX.  Run:  python scripts/make_s4_tables.py [q]
"""
from __future__ import annotations

import json
import os
import sys
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "s4_similarity_results.json")
Q_SWEEP = ["-1", "-0.5", "0", "0.5", "1"]
EPS = 1e-4


def short(doc: str) -> str:
    d = doc.rsplit(".", 1)[0]
    return (d[:22] + "…") if len(d) > 23 else d


def active(c: dict, qd: dict) -> bool:
    return c["n_chunks"] != qd["n_chunks"] or abs(qd["f1"] - c["f1"]) > EPS


def load():
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


# ───────────────────────── console (ASCII) ──────────────────────────────────
def print_per_doc(R, q):
    print(f"\n=== Per-document: Cosine vs q-Cosine at q={q} "
          f"(identical => kernel changed no merge) ===")
    h = f"{'Document':<24}{'Dom':<8}{'F1 cos':>8}{'F1 qco':>8}{'dF1':>8}{'chk c->q':>10}{'Tok cos':>9}{'Tok qco':>9}"
    print(h); print("-" * len(h))
    for r in R:
        c = r["by_q"][q]["cosine"]; qd = r["by_q"][q]["qcosine"]
        d = qd["f1"] - c["f1"]
        flag = "" if active(c, qd) else "  (=)"
        chk = f"{c['n_chunks']}->{qd['n_chunks']}"
        name = short(r['doc']).replace(chr(0x2026), '~')
        print(f"{name:<24}{r['domain']:<8}{c['f1']:>8.3f}{qd['f1']:>8.3f}"
              f"{d:>+8.3f}{chk:>10}"
              f"{c['retrieval_token_cost']:>9.0f}{qd['retrieval_token_cost']:>9.0f}{flag}")


def print_activity(R):
    print("\n=== Head-to-head across q (active documents only) ===")
    h = f"{'q':>5}{'#active/9':>11}{'F1 win':>8}{'tie':>6}{'loss':>6}{'mean dF1*':>11}{'mean dTok*':>12}"
    print(h); print("-" * len(h))
    for q in Q_SWEEP:
        acts, win, tie, loss, df1, dtok = 0, 0, 0, 0, [], []
        for r in R:
            c = r["by_q"][q]["cosine"]; qd = r["by_q"][q]["qcosine"]
            if not active(c, qd):
                continue
            acts += 1
            d = qd["f1"] - c["f1"]
            df1.append(d); dtok.append(qd["retrieval_token_cost"] - c["retrieval_token_cost"])
            if d > EPS: win += 1
            elif d < -EPS: loss += 1
            else: tie += 1
        mdf = f"{mean(df1):+.3f}" if df1 else "  --"
        mdt = f"{mean(dtok):+.0f}" if dtok else "  --"
        print(f"{q:>5}{acts:>11}{win:>8}{tie:>6}{loss:>6}{mdf:>11}{mdt:>12}")
    print("  *averaged over the active documents only (q=+/-1 are identical by construction).")


# ───────────────────────────── LaTeX ────────────────────────────────────────
def fmt(x, d=3):
    return ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "--"


def tex_per_doc(R, q):
    L = [f"% ===== TABLE: per-document S4 kernel at q={q} =====",
         "\\begin{table*}[t]\\centering",
         f"\\caption{{Per-document effect of the S4 Fitouhi--Bouzeffour $q$-cosine kernel "
         f"at $q={q}$, against the classical cosine (all upstream stages and S4 input embeddings "
         "identical). $\\Delta$F1 $=$ F1$_{q\\text{-cos}}-$F1$_{\\cos}$; rows marked "
         "``$=$'' are documents where the kernel changed no merge decision.}}",
         "\\label{tab:s4perdoc}",
         "\\begin{tabular}{@{}llcccccc@{}}\\toprule",
         "Document & Domain & F1$_{\\cos}$ & F1$_{q\\text{-cos}}$ & $\\Delta$F1 "
         "& \\#chk ($\\cos{\\to}q$) & Tok$_{\\cos}$ & Tok$_{q\\text{-cos}}$ \\\\\\midrule"]
    for r in R:
        c = r["by_q"][q]["cosine"]; qd = r["by_q"][q]["qcosine"]
        d = qd["f1"] - c["f1"]
        eq = "" if active(c, qd) else "$^{=}$"
        dd = f"\\textbf{{{d:+.3f}}}" if (active(c, qd) and abs(d) > EPS) else f"{d:+.3f}"
        doc = short(r["doc"]).replace("_", "\\_").replace("…", "\\ldots ")
        L.append(f"{doc}{eq} & {r['domain']} & {fmt(c['f1'])} & {fmt(qd['f1'])} & {dd} & "
                 f"{c['n_chunks']}$\\to${qd['n_chunks']} & {fmt(c['retrieval_token_cost'],0)} "
                 f"& {fmt(qd['retrieval_token_cost'],0)} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(L)


def tex_activity(R):
    L = ["% ===== TABLE: head-to-head activity across q =====",
         "\\begin{table}[t]\\centering",
         "\\caption{How often the Fitouhi--Bouzeffour $q$-cosine kernel actually changes the chunking, "
         "and whether it helps. ``Active'' $=$ documents whose merge decisions differ "
         "from cosine; win/tie/loss and the means are over those active documents only. "
         "$q=\\pm1$ reduce to cosine, so the kernel is inactive there by construction.}",
         "\\label{tab:s4activity}",
         "\\begin{tabular}{@{}cccccrr@{}}\\toprule",
         "$q$ & \\#active/9 & F1 win & tie & loss & mean $\\Delta$F1 & mean $\\Delta$Tok \\\\\\midrule"]
    for q in Q_SWEEP:
        acts, win, tie, loss, df1, dtok = 0, 0, 0, 0, [], []
        for r in R:
            c = r["by_q"][q]["cosine"]; qd = r["by_q"][q]["qcosine"]
            if not active(c, qd):
                continue
            acts += 1
            d = qd["f1"] - c["f1"]
            df1.append(d); dtok.append(qd["retrieval_token_cost"] - c["retrieval_token_cost"])
            if d > EPS: win += 1
            elif d < -EPS: loss += 1
            else: tie += 1
        mdf = f"{mean(df1):+.3f}" if df1 else "--"
        mdt = f"{mean(dtok):+.0f}" if dtok else "--"
        L.append(f"${q}$ & {acts} & {win} & {tie} & {loss} & {mdf} & {mdt} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L)


def tex_avg(R):
    keys = [("f1", "F1", 3), ("mrr", "MRR", 3), ("ndcg", "NDCG", 3),
            ("retrieval_token_cost", "Tok.C.", 0), ("n_chunks", "\\#chk", 1)]
    L = ["% ===== TABLE: corpus average, all docs vs active-only =====",
         "\\begin{table}[t]\\centering",
         "\\caption{Corpus average of the S4 kernels, reported over \\emph{all} documents "
         "(left) and over only the documents where the kernel was \\emph{active} at that $q$ "
         "(right). The all-docs average is diluted by the inactive / $q=\\pm1$ cases; the "
         "active-only average shows the kernel's true effect when it fires.}",
         "\\label{tab:s4avg}",
         "\\begin{tabular}{@{}cl" + "c" * len(keys) + "@{}}\\toprule",
         "$q$ & Kernel & " + " & ".join(k[1] for k in keys) + " \\\\\\midrule"]
    for scope in ("all", "active"):
        L.append("\\multicolumn{%d}{@{}l}{\\emph{%s}}\\\\" %
                 (2 + len(keys), "All documents" if scope == "all" else "Active documents only"))
        for q in Q_SWEEP:
            for mode, name in (("cosine", "Cosine"), ("qcosine", "$q$-Cosine")):
                rows = []
                for r in R:
                    c = r["by_q"][q]["cosine"]; qd = r["by_q"][q]["qcosine"]
                    if scope == "active" and not active(c, qd):
                        continue
                    rows.append(r["by_q"][q][mode])
                if not rows:
                    cells = ["--"] * len(keys)
                else:
                    cells = [fmt(mean(x[k] for x in rows), d) for k, _, d in keys]
                qcol = f"${q}$" if mode == "cosine" else ""
                L.append(f"{qcol} & {name} & " + " & ".join(cells) + " \\\\")
            L.append("\\addlinespace[1pt]")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}\\end{table}"]
    return "\n".join(L)


def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "0"
    R = load()
    print_per_doc(R, q)
    print_activity(R)
    tex = "\n\n".join([tex_per_doc(R, q), tex_activity(R), tex_avg(R)])
    out = os.path.join(HERE, "s4_similarity_tables.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(f"\nWrote {os.path.relpath(out)} (per-doc @q={q}, activity, all-vs-active average).")


if __name__ == "__main__":
    main()
