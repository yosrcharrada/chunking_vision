# Documentation Index — qEntropy Chunker

The accurate, current doc set for the merged platform. (The old P1‑era umbrella
docs — `COMPLETE_DOCUMENTATION.md`, `COMPREHENSIVE_COMMENTS.md`,
`FRONTEND_DOCUMENTATION.md`, `INTEGRATION_NOTES.md`, `QUICK_ANSWERS.md`,
`S3_ENHANCEMENT_SUMMARY.md`, `S3_ENTROPY_README_v4.md`, `S7_RL_DEEP_DIVE.md` —
described the pre‑merge JSD/PPL/ensemble design and were removed.)

## Start here
- **`README.md`** — architecture, how qEntropy drives S3, how evaluation follows
  project 2, how to run, OpenAI‑key notes.
- **`SCORING_FORMULAS_COMPLETE.md`** — the math (Tsallis `S_q`, diversity number
  `D_q`, tree entropy `Σ log Z_K`, Table‑I metrics, GA fitness). Mirrors the
  in‑app Scoring Formulas panel.

## Per‑stage notes (`backend/pipeline/`)
| doc | stage |
|-----|-------|
| `S1_PROFILER_README.md` | document profiling |
| `S2_CHUNKERS_README.md` | candidate chunkers |
| **`S3_ENTROPY_README.md`** | **qEntropy boundary refinement (rewritten)** |
| `S4_BOUNDARY_README.md` | similarity merge |
| `S5_GRAPH_README.md` | entity graph |
| **`S6_EMBEDDING_README.md`** | **shared EmbeddingService (rewritten)** |
| **`S7_RL_README.md`** | **GA tuning q + Table‑I fitness (rewritten)** |

## qEntropy engine (`backend/engine/`)
`entropy.py`, `tree_entropy.py`, `embeddings.py`, `metrics.py`, `quality.py`,
`qagen.py`, `answerability.py`, `chunking.py`, `structure.py`, `segeval.py` —
each module carries a module‑level docstring describing its formulas and contract.

## Evaluation
`backend/pipeline/evaluation.py` — the single source of metric truth: Table‑I
scoring (`score_run`), GA fitness (`ga_fitness`), and the bootstrap‑CI +
token‑cost winner policy (`rank_runs`). Its output is surfaced as the
`p2_evaluation` block in `GET /results`.
