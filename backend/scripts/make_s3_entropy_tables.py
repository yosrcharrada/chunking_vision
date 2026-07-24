"""
make_s3_entropy_tables.py — paper-ready tables for the S3 entropy A/B study
(Tsallis q-entropy vs. the new q-logarithm entropy). Reads
scripts/s3_entropy_results.json (from bench_s3_entropy.py) and emits:
  Table E1: per-q aggregate (tsallis vs qlog), across the corpus
  Table E2: per-domain (averaged over q)
  Table E3: per-document F1 at each q, both entropies
Uses \\win{} (green cell highlight, already in the paper preamble).
Run from backend/:  python scripts/make_s3_entropy_tables.py
Emits: scripts/s3_entropy_tables.tex
"""
from __future__ import annotations
import json, os
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "s3_entropy_results.json")
QS = ["-0.5", "0", "0.5", "1"]


def load():
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


def f(x, d=3):
    return ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "--"


def avg_q(R, q, mode, key):
    return mean(r["by_q"][q][mode][key] for r in R)


def win(a, b, higher=True):
    """(a_str, b_str) with the better one wrapped in \\win{}; ties plain."""
    if abs(a - b) < 1e-3:
        return f(a), f(b)
    a_better = (a > b) if higher else (a < b)
    return (f"\\win{{{f(a)}}}", f(b)) if a_better else (f(a), f"\\win{{{f(b)}}}")


def table_e1(R):
    L = ["% ===== TABLE E1: S3 entropy A/B, per-q aggregate =====",
         "\\begin{table}[t]\\centering",
         "\\caption{S3 boundary-count entropy: Tsallis $q$-entropy vs.\\ the "
         "$q$-logarithm entropy (Eq.~\\ref{eq:qlogentropy}), per Tsallis $q$, "
         "averaged across the 8-document corpus. S1--S2 and S4 are held identical; "
         "only the S3 entropy that sets the boundary count differs. \\win{green} "
         "marks the better F1 of each pair.}",
         "\\label{tab:s3ent_byq}",
         "\\begin{tabular}{@{}c l ccccc c@{}}\\toprule",
         "$q$ & Entropy & F1$\\uparrow$ & MRR$\\uparrow$ & NDCG$\\uparrow$ & QCS$\\uparrow$ "
         "& AC$\\uparrow$ & \\#chk \\\\\\midrule"]
    for q in QS:
        f1t, f1q = win(avg_q(R, q, "tsallis", "f1"), avg_q(R, q, "qlog", "f1"))
        for mode, name, f1c in (("tsallis", "Tsallis", f1t), ("qlog", "$q$-log", f1q)):
            cells = [f1c,
                     f(avg_q(R, q, mode, "mrr")), f(avg_q(R, q, mode, "ndcg")),
                     f(avg_q(R, q, mode, "qcs")), f(avg_q(R, q, mode, "answer_correct")),
                     f(avg_q(R, q, mode, "n_chunks"), 1)]
            qcol = f"${q}$" if mode == "tsallis" else ""
            L.append(f"{qcol} & {name} & " + " & ".join(cells) + " \\\\")
        if q != QS[-1]:
            L.append("\\addlinespace[1pt]")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L)


def table_e2(R):
    doms = sorted(set(r["domain"] for r in R))
    L = ["% ===== TABLE E2: S3 entropy A/B, per domain =====",
         "\\begin{table}[t]\\centering",
         "\\caption{S3 entropy by document domain, averaged over "
         "$q\\in\\{-0.5,0,0.5,1\\}$. $\\Delta$F1 = $q$-log $-$ Tsallis; "
         "\\win{green} marks the better F1 of each pair.}",
         "\\label{tab:s3ent_domain}",
         "\\begin{tabular}{@{}l l cccc c@{}}\\toprule",
         "Domain & Entropy & F1$\\uparrow$ & MRR$\\uparrow$ & NDCG$\\uparrow$ & AC$\\uparrow$ "
         "& $\\Delta$F1 \\\\\\midrule"]
    for dom in doms:
        rows = [r for r in R if r["domain"] == dom]
        def dm(mode, key):
            return mean(r["by_q"][q][mode][key] for r in rows for q in QS)
        ft, fq = dm("tsallis", "f1"), dm("qlog", "f1")
        d = fq - ft
        dcell = f"\\textcolor{{green!45!black}}{{\\textbf{{{d:+.3f}}}}}" if d > 1e-3 else f"{d:+.3f}"
        f1t, f1q = win(ft, fq)
        for mode, name, f1c in (("tsallis", "Tsallis", f1t), ("qlog", "$q$-log", f1q)):
            cells = [f1c, f(dm(mode, "mrr")), f(dm(mode, "ndcg")), f(dm(mode, "answer_correct"))]
            dcol = dom if mode == "tsallis" else ""
            tail = f" & {dcell}" if mode == "qlog" else " & "
            L.append(f"{dcol} & {name} & " + " & ".join(cells) + tail + " \\\\")
        L.append("\\addlinespace[1pt]")
    L[-1] = "\\bottomrule\\end{tabular}\\end{table}"
    return "\n".join(L)


def table_e3(R):
    L = ["% ===== TABLE E3: per-document F1 at each q, both entropies =====",
         "\\begin{table*}[t]\\centering",
         "\\caption{Per-document F1, Tsallis (T) vs.\\ $q$-log (L) entropy in S3, "
         "at each Tsallis $q$. \\win{green} marks the better of each T/L pair "
         "(ties, notably at $q\\in\\{0,1\\}$ where the two effective counts "
         "coincide, are left plain).}",
         "\\label{tab:s3ent_perdoc}",
         "\\begin{tabular}{@{}ll cc cc cc cc@{}}\\toprule",
         "\\multirow{2}{*}{Document} & \\multirow{2}{*}{Domain} & "
         "\\multicolumn{2}{c}{$q=-0.5$} & \\multicolumn{2}{c}{$q=0$} & "
         "\\multicolumn{2}{c}{$q=0.5$} & \\multicolumn{2}{c}{$q=1$} \\\\"
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
         " & & T & L & T & L & T & L & T & L \\\\\\midrule"]
    for r in sorted(R, key=lambda x: (x["domain"], x["doc"])):
        cells = []
        for q in QS:
            t, l = r["by_q"][q]["tsallis"]["f1"], r["by_q"][q]["qlog"]["f1"]
            ts, ls = win(t, l)
            cells += [ts, ls]
        doc = r["doc"].replace("_", "\\_")
        if len(doc) > 30:
            doc = doc[:28] + "\\ldots"
        L.append(f"{doc} & {r['domain']} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(L)


def main():
    R = load()
    tex = "\n\n".join([table_e1(R), table_e2(R), table_e3(R)])
    out = os.path.join(HERE, "s3_entropy_tables.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(tex)
    print(f"\nWrote {os.path.relpath(out)} ({len(R)} documents).")


if __name__ == "__main__":
    main()
