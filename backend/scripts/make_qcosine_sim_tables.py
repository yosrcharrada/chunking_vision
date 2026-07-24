"""
make_qcosine_sim_tables.py — the two tables the instructor asked for:
  Table Q1: qentropy at each q, BEFORE (cosine) vs AFTER (q-cosine) — corpus-wide
  Table Q2: qentropy per DOMAIN, BEFORE (cosine) vs AFTER (q-cosine)
Both from scripts/s4_similarity_results.json (S1-S3 identical between the two
arms; only the S4 similarity kernel differs — "before/after q-cosine").
Run from backend/:  python scripts/make_qcosine_sim_tables.py
Emits: scripts/qcosine_sim_tables.tex
"""
from __future__ import annotations
import json, os
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "s4_similarity_results.json")
Q_SWEEP = ["-1", "-0.5", "0", "0.5", "1"]
KEYS = ["f1", "mrr", "ndcg", "qcs", "retrieval_token_cost", "n_chunks"]


def load():
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


def f(x, d=3):
    return ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "--"


def avg_q(R, q, mode, key):
    vals = [r["by_q"][q][mode][key] for r in R if q in r.get("by_q", {})]
    return mean(vals) if vals else 0.0


def avg_dom(R, domain, mode, key):
    vals = []
    for r in R:
        if r["domain"] != domain:
            continue
        for q in Q_SWEEP:
            vals.append(r["by_q"][q][mode][key])
    return mean(vals) if vals else 0.0


def table_q1(R) -> str:
    L = ["% ===== TABLE Q1: qentropy at each q, before vs after the q-cosine kernel =====",
         "\\begin{table}[t]\\centering",
         "\\caption{Effect of Tsallis $q$ on retrieval quality, before (classical "
         "cosine, S4) and after (Fitouhi--Bouzeffour $q$-cosine, S4) the kernel swap. "
         "S1--S3 (the qentropy boundary selection) are identical in both columns; only "
         "the S4 similarity kernel differs. Averaged across all 9 corpus documents. "
         "At $q=\\pm1$ the $q$-cosine reduces exactly to cosine.}",
         "\\label{tab:qcos_by_q}",
         "\\begin{tabular}{@{}c l ccccc@{}}\\toprule",
         "$q$ & Kernel & F1$\\uparrow$ & MRR$\\uparrow$ & NDCG$\\uparrow$ & QCS$\\uparrow$ "
         "& \\#chk \\\\\\midrule"]
    for q in Q_SWEEP:
        for mode, name in (("cosine", "Before ($\\cos$)"), ("qcosine", "After ($q\\text{-}\\cos$)")):
            cells = [f(avg_q(R, q, mode, k)) for k in ("f1", "mrr", "ndcg", "qcs")]
            cells.append(f(avg_q(R, q, mode, "n_chunks"), 1))
            qcol = f"${q}$" if mode == "cosine" else ""
            L.append(f"{qcol} & {name} & " + " & ".join(cells) + " \\\\")
        if q != Q_SWEEP[-1]:
            L.append("\\addlinespace[1pt]")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L)


def table_q2(R) -> str:
    domains = sorted(set(r["domain"] for r in R))
    L = ["% ===== TABLE Q2: qentropy per domain, before vs after the q-cosine kernel =====",
         "\\begin{table}[t]\\centering",
         "\\caption{Effect of the S4 similarity kernel by document domain, averaged "
         "over $q\\in\\{-1,-0.5,0,0.5,1\\}$. ``Before'' = classical cosine; ``After'' "
         "= Fitouhi--Bouzeffour $q$-cosine. $\\Delta$F1 = After $-$ Before.}",
         "\\label{tab:qcos_by_domain}",
         "\\begin{tabular}{@{}l l ccccc c@{}}\\toprule",
         "Domain & Kernel & F1$\\uparrow$ & MRR$\\uparrow$ & NDCG$\\uparrow$ & QCS$\\uparrow$ "
         "& \\#chk & $\\Delta$F1 \\\\\\midrule"]
    for dom in domains:
        f1_before = avg_dom(R, dom, "cosine", "f1")
        f1_after = avg_dom(R, dom, "qcosine", "f1")
        d = f1_after - f1_before
        dcell = f"\\textbf{{{d:+.3f}}}" if abs(d) > 1e-3 else f"{d:+.3f}"
        for i, (mode, name) in enumerate((("cosine", "Before"), ("qcosine", "After"))):
            cells = [f(avg_dom(R, dom, mode, k)) for k in ("f1", "mrr", "ndcg", "qcs")]
            cells.append(f(avg_dom(R, dom, mode, "n_chunks"), 1))
            domcol = dom if i == 0 else ""
            trailing = f" & {dcell}" if i == 1 else " & "
            L.append(f"{domcol} & {name} & " + " & ".join(cells) + trailing + " \\\\")
        L.append("\\addlinespace[1pt]")
    L[-1] = "\\bottomrule\\end{tabular}\\end{table}"
    return "\n".join(L)


def main():
    R = load()
    tex = table_q1(R) + "\n\n" + table_q2(R)
    out = os.path.join(HERE, "qcosine_sim_tables.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(tex)
    print(f"\nWrote {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
