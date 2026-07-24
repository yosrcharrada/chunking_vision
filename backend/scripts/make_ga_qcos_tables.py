"""
make_ga_qcos_tables.py — paper-ready tables for the GA cosine-vs-q-cosine study.
================================================================================
Reads scripts/ga_qcos_results.json (produced by exp_ga_qcos.py) and emits:
  Table GA1: aggregate before/after, across the corpus
  Table GA2: per-domain before/after
  Table GA3: per-document detail (F1, AC, tuned q*, #chunks)
Uses \\win{} (green cell highlight, already defined in the paper preamble) for
the better of the Before/After pair per metric.
Run from backend/:  python scripts/make_ga_qcos_tables.py
Emits: scripts/ga_qcos_tables.tex
"""
from __future__ import annotations
import json, os
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "ga_qcos_results.json")
KEYS = ["f1", "mrr", "ndcg", "qcs", "answer_correct", "retrieval_token_cost", "n_chunks"]


def load():
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


def f(x, d=3):
    return ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "--"


def avg(R, arm, key):
    vals = [r[arm][key] for r in R if arm in r and isinstance(r[arm].get(key), (int, float))]
    return mean(vals) if vals else 0.0


def _safe_mean(vals) -> float:
    """mean() of only the numeric entries; 0.0 if none (e.g. AC when judge was off)."""
    nums = [v for v in vals if isinstance(v, (int, float))]
    return mean(nums) if nums else 0.0


def win(before, after, higher_better=True):
    """Wrap the better of (before, after) in \\win{}; leave ties/None unmarked."""
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return f(before), f(after)
    if abs(before - after) < 1e-3:
        return f"{f(before)}", f"{f(after)}"
    b_wins = (before > after) if higher_better else (before < after)
    return (f"\\win{{{f(before)}}}", f(after)) if b_wins else (f(before), f"\\win{{{f(after)}}}")


def table_ga1(R) -> str:
    b_f1, a_f1 = avg(R, "ga_base", "f1"), avg(R, "ga_qcos", "f1")
    b_mrr, a_mrr = avg(R, "ga_base", "mrr"), avg(R, "ga_qcos", "mrr")
    b_ndcg, a_ndcg = avg(R, "ga_base", "ndcg"), avg(R, "ga_qcos", "ndcg")
    b_qcs, a_qcs = avg(R, "ga_base", "qcs"), avg(R, "ga_qcos", "qcs")
    b_ac, a_ac = avg(R, "ga_base", "answer_correct"), avg(R, "ga_qcos", "answer_correct")
    b_tok, a_tok = avg(R, "ga_base", "retrieval_token_cost"), avg(R, "ga_qcos", "retrieval_token_cost")
    b_n, a_n = avg(R, "ga_base", "n_chunks"), avg(R, "ga_qcos", "n_chunks")
    f1c = win(b_f1, a_f1); mrrc = win(b_mrr, a_mrr); ndcgc = win(b_ndcg, a_ndcg)
    qcsc = win(b_qcs, a_qcs); acc = win(b_ac, a_ac)
    L = ["% ===== TABLE GA1: GA before/after q-cosine kernel, aggregate =====",
         "\\begin{table}[t]\\centering",
         "\\caption{GA-tuned pipeline, before (S4 legacy gate) vs.\\ after (S4 "
         "Fitouhi--Bouzeffour $q$-cosine kernel, $q=-1$ excluded per kernel "
         "definition), averaged across the corpus. The GA searches the SAME 5 "
         "genes in both arms; \\win{green} marks the better of each pair.}",
         "\\label{tab:ga_qcos_agg}",
         "\\begin{tabular}{@{}l ccccc cc@{}}\\toprule",
         "Arm & F1$\\uparrow$ & MRR$\\uparrow$ & NDCG$\\uparrow$ & QCS$\\uparrow$ & AC$\\uparrow$ "
         "& Tok.C.$\\downarrow$ & \\#chk \\\\\\midrule",
         f"Before (S4 cosine) & {f1c[0]} & {mrrc[0]} & {ndcgc[0]} & {qcsc[0]} & {acc[0]} & "
         f"{f(b_tok,1)} & {f(b_n,1)} \\\\",
         f"After ($q$-cosine)  & {f1c[1]} & {mrrc[1]} & {ndcgc[1]} & {qcsc[1]} & {acc[1]} & "
         f"{f(a_tok,1)} & {f(a_n,1)} \\\\",
         "\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(L)


def table_ga2(R) -> str:
    domains = sorted(set(r["domain"] for r in R))
    L = ["% ===== TABLE GA2: GA before/after q-cosine kernel, per domain =====",
         "\\begin{table}[t]\\centering",
         "\\caption{GA-tuned pipeline by domain, before (S4 cosine) vs.\\ after "
         "($q$-cosine, $q=-1$ excluded). $\\Delta$F1 = After $-$ Before; "
         "\\win{green} marks the better of each pair.}",
         "\\label{tab:ga_qcos_domain}",
         "\\begin{tabular}{@{}l l ccccc c@{}}\\toprule",
         "Domain & Arm & F1$\\uparrow$ & MRR$\\uparrow$ & NDCG$\\uparrow$ & QCS$\\uparrow$ & AC$\\uparrow$ "
         "& $\\Delta$F1 \\\\\\midrule"]
    for dom in domains:
        rows = [r for r in R if r["domain"] == dom]
        b_f1 = _safe_mean(r["ga_base"]["f1"] for r in rows)
        a_f1 = _safe_mean(r["ga_qcos"]["f1"] for r in rows)
        d = a_f1 - b_f1
        dcell = f"\\textcolor{{green!45!black}}{{\\textbf{{{d:+.3f}}}}}" if d > 1e-3 else f"{d:+.3f}"
        for i, arm in enumerate(("ga_base", "ga_qcos")):
            cells = [_safe_mean(r[arm][k] for r in rows) for k in ("f1", "mrr", "ndcg", "qcs", "answer_correct")]
            name = "Before" if arm == "ga_base" else "After"
            domcol = dom if i == 0 else ""
            trailing = f" & {dcell}" if i == 1 else " & "
            L.append(f"{domcol} & {name} & " + " & ".join(f(c) for c in cells) + trailing + " \\\\")
        L.append("\\addlinespace[1pt]")
    L[-1] = "\\bottomrule\\end{tabular}\\end{table}"
    return "\n".join(L)


def table_ga3(R) -> str:
    L = ["% ===== TABLE GA3: per-document GA before/after q-cosine =====",
         "\\begin{table*}[t]\\centering",
         "\\caption{Per-document GA outcome, before (S4 cosine) vs.\\ after "
         "($q$-cosine, $q=-1$ excluded): F1, answer-correctness (AC), the "
         "GA-discovered $q^{*}$ (jointly driving S3's $D_q$ and the S4 kernel "
         "in the After arm), and final chunk count.}",
         "\\label{tab:ga_qcos_perdoc}",
         "\\begin{tabular}{@{}ll cc cc cc cc@{}}\\toprule",
         "Document & Domain & F1$_{\\text{Bef}}$ & F1$_{\\text{Aft}}$ & "
         "AC$_{\\text{Bef}}$ & AC$_{\\text{Aft}}$ & "
         "$q^{*}_{\\text{Bef}}$ & $q^{*}_{\\text{Aft}}$ & "
         "\\#chk$_{\\text{Bef}}$ & \\#chk$_{\\text{Aft}}$ \\\\\\midrule"]
    for r in sorted(R, key=lambda x: (x["domain"], x["doc"])):
        b, a = r["ga_base"], r["ga_qcos"]
        f1b, f1a = win(b["f1"], a["f1"])
        acb, aca = win(b["answer_correct"], a["answer_correct"])
        doc = r["doc"].replace("_", "\\_")
        L.append(f"{doc} & {r['domain']} & {f1b} & {f1a} & {acb} & {aca} & "
                 f"{f(b['tuned_q'],2)} & {f(a['tuned_q'],2)} & {b['n_chunks']} & {a['n_chunks']} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    return "\n".join(L)


def main():
    R = load()
    tex = "\n\n".join([table_ga1(R), table_ga2(R), table_ga3(R)])
    out = os.path.join(HERE, "ga_qcos_tables.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(tex + "\n")
    print(tex)
    print(f"\nWrote {os.path.relpath(out)} ({len(R)} documents).")


if __name__ == "__main__":
    main()
