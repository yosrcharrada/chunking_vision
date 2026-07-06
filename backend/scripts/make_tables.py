"""
make_tables.py — render publication-quality LaTeX tables + a pgfplots chart from
benchmark_results.json.  Pure-python (json + statistics); no model imports, so it
runs instantly and can be re-run any time you change the data.

Usage:  python scripts/make_tables.py
Output: scripts/benchmark_tables.tex   (Tables V-XII)
        scripts/benchmark_chart.tex    (pgfplots: metrics vs q)

Add to your preamble:  \\usepackage[table]{xcolor}  \\usepackage{pgfplots}  \\usepackage{booktabs}
"""
import json
import os
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "benchmark_results.json"), encoding="utf-8"))
QS = ["-1", "-0.5", "0", "0.5", "1"]
DOM_ORDER = ["Medical", "Code", "Legal", "General"]

# metric key, header (with arrow), decimals, is_points_delta
COLS = [
    ("precision", "Prec.$\\uparrow$", 3),
    ("recall",    "Rec.$\\uparrow$",  3),
    ("f1",        "F1$\\uparrow$",    3),
    ("mrr",       "MRR$\\uparrow$",   3),
    ("ndcg",      "NDCG$\\uparrow$",  3),
    ("ss2fd",     "ss2fd$\\downarrow$", 3),
    ("srgt",      "SRGT$\\uparrow$",  3),
    ("retrieval_token_cost", "Tok.C.$\\downarrow$", 0),
    ("qcs",       "QCS$\\uparrow$",   3),
    ("answer_correct", "AC$\\uparrow$", 3),
]


def avg(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return mean(v) if v else 0.0


def fnum(x, d=3):
    if not isinstance(x, (int, float)):
        return "--"
    return f"{x:.0f}" if d == 0 else f"{x:.{d}f}"


def cellrow(m):
    return " & ".join(fnum(m.get(k, 0.0), d) for k, _, d in COLS)


def best_fixed(r):
    qs = r.get("q_sweep", {})
    if not qs:
        return {}
    bk = max(qs, key=lambda k: qs[k].get("f1", 0.0))
    return qs[bk]


def best_q_key(r):
    qs = r.get("q_sweep", {})
    return max(qs, key=lambda k: qs[k].get("f1", 0.0)) if qs else None


L = []
hdr = " & ".join(h for _, h, _ in COLS)

# ── TABLE V — Classical Baselines ────────────────────────────────────────────
L += ["% ===== TABLE V =====",
      "\\begin{table*}[t]\\centering",
      "\\caption{Classical Baselines (Average Across All Corpora). $\\uparrow$=higher is better; $\\downarrow$=lower is better.}",
      "\\label{tab:baselines}",
      "\\begin{tabular}{@{}l" + "c" * len(COLS) + "@{}}\\toprule",
      "Method & " + hdr + " \\\\\\midrule"]
for label in ("Fixed-size", "RCTS 512/64", "Semantic-P95"):
    rows = [r["baselines"][label] for r in R if label in r.get("baselines", {})]
    L.append(f"{label} & " + cellrow({k: avg(rows, k) for k, _, _ in COLS}) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table*}", ""]

# ── TABLE VI — Entropy-Enhanced + Δ rows ─────────────────────────────────────
sem = [r["baselines"]["Semantic-P95"] for r in R if "Semantic-P95" in r.get("baselines", {})]
sha = [r["q_sweep"]["1"] for r in R if r.get("q_sweep", {}).get("1")]
qen = [best_fixed(r) for r in R if r.get("q_sweep")]
ga = [r["after_ga"] for r in R if r.get("after_ga")]
A = lambda rows: {k: avg(rows, k) for k, _, _ in COLS}
a_sem, a_sha, a_qen, a_ga = A(sem), A(sha), A(qen), A(ga)


def delta_row(name, a, b):
    parts = []
    for k, _, d in COLS:
        if k == "ss2fd":
            parts.append(f"{a[k]-b[k]:+.2f}")
        elif k == "retrieval_token_cost":
            parts.append(f"{a[k]-b[k]:+.0f}")
        else:
            parts.append(f"{(a[k]-b[k])*100:+.0f}")
    return f"{name} & " + " & ".join(parts) + " \\\\"


L += ["% ===== TABLE VI =====",
      "\\begin{table*}[t]\\centering",
      "\\caption{Entropy-Enhanced Results (Average Across All Corpora). Green = best overall. "
      "$\\Delta$ rows show absolute improvement (points $\\times100$; raw for ss2fd / Tok.C.).}",
      "\\label{tab:entropy}",
      "\\begin{tabular}{@{}l" + "c" * len(COLS) + "@{}}\\toprule",
      "Method & " + hdr + " \\\\\\midrule",
      "Semantic-95 (best baseline) & " + cellrow(a_sem) + " \\\\\\midrule",
      "Shannon ($q\\to1$) & " + cellrow(a_sha) + " \\\\",
      "$q$-entropy (best fixed $q$) & " + cellrow(a_qen) + " \\\\",
      "\\rowcolor{green!18}\\textbf{$q$-ent.\\ + GA} & " + cellrow(a_ga) + " \\\\\\midrule",
      delta_row("$\\Delta$: Shannon vs.\\ Sem-95", a_sha, a_sem),
      delta_row("$\\Delta$: $q$-ent.\\ vs.\\ Shannon", a_qen, a_sha),
      delta_row("$\\Delta$: GA vs.\\ $q$-ent.", a_ga, a_qen),
      "\\bottomrule\\end{tabular}\\end{table*}", ""]

# ── TABLE VII — Effect of q (avg) + Behaviour ────────────────────────────────
behav = {"-1": "Finest / over-segments", "-0.5": "Sensitive", "0": "Balanced",
         "0.5": "Optimal", "1": "Shannon (coarsest)"}
best_avg_q = max(QS, key=lambda q: avg([r["q_sweep"][q] for r in R if r.get("q_sweep", {}).get(q)], "f1"))
L += ["% ===== TABLE VII =====",
      "\\begin{table}[t]\\centering",
      "\\caption{Effect of $q$ on Retrieval Metrics (Average Across All Corpora).}",
      "\\label{tab:qsweep}",
      "\\begin{tabular}{@{}lcccccccl@{}}\\toprule",
      "$q$ & $D_q$ & \\#chk & F1$\\uparrow$ & MRR$\\uparrow$ & NDCG$\\uparrow$ & Tok.C.$\\downarrow$ & AC$\\uparrow$ & Behaviour \\\\\\midrule"]
for q in QS:
    rows = [r["q_sweep"][q] for r in R if r.get("q_sweep", {}).get(q)]
    cells = [fnum(avg(rows, "diversity_number"), 1), fnum(avg(rows, "n_chunks"), 1),
             fnum(avg(rows, "f1")), fnum(avg(rows, "mrr")), fnum(avg(rows, "ndcg")),
             fnum(avg(rows, "retrieval_token_cost"), 0), fnum(avg(rows, "answer_correct"))]
    b = behav[q]
    qlbl = f"${q}$"
    if q == best_avg_q:
        cells = [f"\\textbf{{{c}}}" for c in cells]; qlbl = f"\\textbf{{{qlbl}}}"; b = f"\\textbf{{{b}}}"
    L.append(qlbl + " & " + " & ".join(cells) + " & " + b + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}", ""]

# ── TABLE VIII — GA q* by domain (transposed) ────────────────────────────────
doms = [d for d in DOM_ORDER if any(r["domain"] == d for r in R)]
dom_rows = {d: [r["after_ga"] for r in R if r["domain"] == d and r.get("after_ga")] for d in doms}
L += ["% ===== TABLE VIII =====",
      "\\begin{table}[t]\\centering",
      "\\caption{GA-Discovered Optimal $q^{*}$ by Domain.}",
      "\\label{tab:qstar}",
      "\\begin{tabular}{@{}l" + "c" * len(doms) + "@{}}\\toprule",
      " & " + " & ".join(f"\\textbf{{{d}}}" for d in doms) + " \\\\\\midrule"]
for k, lab, d in [("precision", "Prec.$\\uparrow$", 3), ("recall", "Rec.$\\uparrow$", 3),
                  ("mrr", "MRR$\\uparrow$", 3), ("ndcg", "NDCG$\\uparrow$", 3),
                  ("answer_correct", "AC$\\uparrow$", 3),
                  ("retrieval_token_cost", "Tok.C.$\\downarrow$", 0)]:
    L.append(lab + " & " + " & ".join(fnum(avg(dom_rows[dm], k), d) for dm in doms) + " \\\\")
L.append("\\midrule $q^{*}$ & " + " & ".join(fnum(avg(dom_rows[dm], "tuned_q"), 2) for dm in doms) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}", ""]

# ── TABLE IX — Ablation + Δ(F1) ──────────────────────────────────────────────
ABL = ["Full", "-- LSTM", "-- Structure", "Shannon (q=1)", "-- S4 filter"]
full = [r["ablation"]["Full"] for r in R if "Full" in r.get("ablation", {})]
f1_full = avg(full, "f1")
L += ["% ===== TABLE IX =====",
      "\\begin{table}[t]\\centering",
      "\\caption{Component Ablation (Average Across All Corpora). $\\Delta$ = F1 change vs.\\ Full (points).}",
      "\\label{tab:ablation}",
      "\\begin{tabular}{@{}lccccc@{}}\\toprule",
      "Configuration & Prec.$\\uparrow$ & Rec.$\\uparrow$ & F1$\\uparrow$ & AC$\\uparrow$ & $\\Delta$F1 \\\\\\midrule"]
disp = {"Full": "\\textbf{Full ($q$+GA)}", "-- LSTM": "$-$ LSTM",
        "-- Structure": "$-$ Structural priors", "Shannon (q=1)": "Replace $q$-ent.\\ w/ Shannon",
        "-- S4 filter": "$-$ S4 quality filter"}
for label in ABL:
    rows = [r["ablation"][label] for r in R if label in r.get("ablation", {})]
    dlt = (avg(rows, "f1") - f1_full) * 100
    dl = "--" if label == "Full" else f"{dlt:+.0f}"
    L.append(disp[label] + " & " + " & ".join([fnum(avg(rows, "precision")), fnum(avg(rows, "recall")),
             fnum(avg(rows, "f1")), fnum(avg(rows, "answer_correct"))]) + f" & {dl} \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}", ""]

# ── TABLE X — Per-document F1 across q (best bold) + Before + GA ──────────────
L += ["% ===== TABLE X =====",
      "\\begin{table*}[t]\\centering",
      "\\caption{Per-Document F1 across $q$ (best fixed $q$ in \\textbf{bold}), with the pre-entropy "
      "baseline (Before) and the GA-tuned result.}",
      "\\label{tab:perdoc}",
      "\\begin{tabular}{@{}ll" + "c" * len(QS) + "ccc@{}}\\toprule",
      "Document & Domain & " + " & ".join(f"$q={q}$" for q in QS)
      + " & Before & $q^{*}$ & GA-F1 \\\\\\midrule"]
for r in R:
    qs = r.get("q_sweep", {})
    bk = best_q_key(r)
    cells = []
    for q in QS:
        s = fnum((qs.get(q) or {}).get("f1"))
        cells.append(f"\\textbf{{{s}}}" if q == bk else s)
    gad = r.get("after_ga", {})
    doc = r["doc"].replace("_", "\\_")[:26]
    L.append(f"{doc} & {r['domain']} & " + " & ".join(cells)
             + f" & {fnum((r.get('before_entropy') or {}).get('f1'))}"
             + f" & {fnum(gad.get('tuned_q'),2)} & {fnum(gad.get('f1'))} \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table*}", ""]

# ── TABLE XI — Optimal-q sign analysis (negative vs positive) ────────────────
def sgn(x):
    return "$-$ (finer)" if x < -1e-9 else ("$+$ (coarser)" if x > 1e-9 else "$0$")


L += ["% ===== TABLE XI =====",
      "\\begin{table}[t]\\centering",
      "\\caption{Optimal-$q$ polarity per document. Negative $q$ = finer chunks; positive $q$ = coarser.}",
      "\\label{tab:qsign}",
      "\\begin{tabular}{@{}llcccc@{}}\\toprule",
      "Document & Domain & best $q$ & sign & $q^{*}$(GA) & sign \\\\\\midrule"]
neg_ga = pos_ga = 0
for r in R:
    bk = best_q_key(r)
    gq = r.get("after_ga", {}).get("tuned_q", 0.0)
    neg_ga += gq < 0
    pos_ga += gq > 0
    doc = r["doc"].replace("_", "\\_")[:24]
    L.append(f"{doc} & {r['domain']} & ${bk}$ & {sgn(float(bk))} & {fnum(gq,2)} & {sgn(gq)} \\\\")
L += ["\\bottomrule\\end{tabular}",
      f"\\\\[2pt]\\footnotesize Negative (finer) $q^{{*}}$: {neg_ga}/{len(R)} documents; "
      f"positive (coarser): {pos_ga}/{len(R)}. The optimum is document-specific.",
      "\\end{table}", ""]

# ── TABLE XII — Stage-wise gains (Before -> best q -> GA) ────────────────────
before = [r["before_entropy"] for r in R if r.get("before_entropy")]
L += ["% ===== TABLE XII =====",
      "\\begin{table}[t]\\centering",
      "\\caption{Pipeline-Stage Gains (Average Across All Corpora).}",
      "\\label{tab:stages}",
      "\\begin{tabular}{@{}lcccc@{}}\\toprule",
      "Stage & F1$\\uparrow$ & MRR$\\uparrow$ & AC$\\uparrow$ & Tok.C.$\\downarrow$ \\\\\\midrule",
      "Before entropy (raw S2) & " + " & ".join([fnum(avg(before, "f1")), fnum(avg(before, "mrr")),
            fnum(avg(before, "answer_correct")), fnum(avg(before, "retrieval_token_cost"), 0)]) + " \\\\",
      "After entropy (best $q$) & " + " & ".join([fnum(avg(qen, "f1")), fnum(avg(qen, "mrr")),
            fnum(avg(qen, "answer_correct")), fnum(avg(qen, "retrieval_token_cost"), 0)]) + " \\\\",
      "\\rowcolor{green!18}After GA ($q^{*}$) & " + " & ".join([fnum(avg(ga, "f1")), fnum(avg(ga, "mrr")),
            fnum(avg(ga, "answer_correct")), fnum(avg(ga, "retrieval_token_cost"), 0)]) + " \\\\",
      "\\bottomrule\\end{tabular}\\end{table}"]

with open(os.path.join(HERE, "benchmark_tables.tex"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))

# ── CHART — metrics vs q (pgfplots) ──────────────────────────────────────────
series = [("F1", "f1"), ("MRR", "mrr"), ("NDCG", "ndcg"), ("AC", "answer_correct"), ("QCS", "qcs")]
coords = {name: [(float(q), avg([r["q_sweep"][q] for r in R if r.get("q_sweep", {}).get(q)], key))
                 for q in QS] for name, key in series}
C = ["% pgfplots chart for Table VII — \\usepackage{pgfplots}\\pgfplotsset{compat=1.18}",
     "\\begin{figure}[t]\\centering\\begin{tikzpicture}",
     "\\begin{axis}[width=.95\\linewidth,height=6cm,xlabel={Tsallis $q$},ylabel={Metric},",
     "  legend pos=outer north east,grid=both,xtick={-1,-0.5,0,0.5,1},ymin=0.4,ymax=1.0,",
     "  legend style={font=\\footnotesize}]"]
for name, _ in series:
    pts = " ".join(f"({x:g},{y:.3f})" for x, y in coords[name])
    C.append(f"\\addplot+[mark=*] coordinates {{{pts}}}; \\addlegendentry{{{name}}}")
C += ["\\end{axis}\\end{tikzpicture}",
      "\\caption{Effect of the Tsallis $q$ on retrieval metrics (averaged across all corpora).}",
      "\\label{fig:qsweep}\\end{figure}"]
with open(os.path.join(HERE, "benchmark_chart.tex"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(C))

print("wrote benchmark_tables.tex (Tables V-XII) and benchmark_chart.tex (pgfplots)")
print(f"GA q* polarity: {neg_ga} negative / {pos_ga} positive of {len(R)} docs")
