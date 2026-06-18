# qEntropy Chunker — merged platform

A document‑chunking research platform that fuses **two projects**:

1. an **8‑stage chunking pipeline** (profiler → multi‑strategy chunkers → entropy
   refinement → boundary filter → graph → embeddings → genetic‑algorithm tuner →
   evaluation) with a React/Vite frontend, and
2. a focused **qentropy engine** (Tsallis q‑entropy + Hill diversity number,
   paper‑faithful tree entropy, and the Table‑I retrieval metrics from the
   chunking‑comparison literature).

The merge makes the **qentropy engine the sole entropy authority**: stage S3 no
longer uses the old JSD/Hellinger/PMI/PPL signal palette — the Tsallis q‑entropy
and its diversity number drive every boundary decision, the genetic algorithm
tunes the Tsallis `q`, and **every chunking is scored with the qentropy project's
Table‑I metrics** (used both per‑strategy and after the GA).

---

## Architecture

```
upload ─► S1 Profiler ─► QA gen ─► S2 Chunkers ─► [per strategy] S3→S4→S5→S6 ─► S7 GA ─► Evaluation ─► results
                         (engine.qagen)    │                                        │            │
                                           │     S3 = qEntropy (engine.entropy,      │   engine.metrics (Table I)
                                           │          engine.tree_entropy)           │   + winner policy
                                           └─ multiple candidate chunkings           └─ GA tunes q, S2 size, τ_sem
```

| Stage | Module | Role |
|-------|--------|------|
| S1 | `pipeline/s1_profiler.py` | document type / domain / length profile |
| QA | `engine/qagen.py` | auto‑generate `(query, ground_truth)` eval pairs (optional, needs key) |
| S2 | `pipeline/s2_chunkers.py` | many candidate chunkings (recursive, structure, semantic, legal, …) |
| **S3** | **`pipeline/s3_entropy.py`** | **qEntropy boundary refinement** — Tsallis `S_q` + diversity number `D_q` set the boundary count; P1's LSTM is re‑wired over the qentropy features; tree entropy `h_K` is exposed |
| S4 | `pipeline/s4_boundary.py` | similarity‑based boundary merge (τ_sem) |
| S5 | `pipeline/s5_graph.py` | entity graph enrichment |
| S6 | `pipeline/s6_embedding.py` | chunk embeddings via the shared `EmbeddingService` |
| **S7** | **`pipeline/s7_rl.py`** | **genetic algorithm** — tunes Tsallis `q ∈ [-1, 1]`, S2 sizing, τ_sem, min‑tokens; fitness = Table‑I metrics |
| Eval | **`pipeline/evaluation.py`** | **Table‑I metrics + winner policy** for every strategy and the GA winner |

### The qentropy engine (`backend/engine/`, vendored from project 2)
`entropy.py` (Shannon + Tsallis `S_q` + Hill diversity number `D_q`),
`tree_entropy.py` (paper‑faithful `Σ log Z_K`), `embeddings.py`
(`EmbeddingService`: openai → multilingual → english → tfidf), `metrics.py`
(Table I), `quality.py` (label‑free q selection), `qagen.py` (auto QA),
`answerability.py` (opt‑in LLM judge), `chunking.py`/`structure.py`/`segeval.py`.

---

## How qEntropy drives S3 (the heart of the merge)

1. Each S2 chunk is a *unit*; the semantic‑shift signal is `dᵢ = 1 − cos(uᵢ, uᵢ₊₁)`.
2. Significant shifts (above the document's own median coherence) form a
   probability distribution `p` (`engine.entropy.normalize`).
3. The generalised entropy sets the granularity:
   `H = Shannon(p)`, `S_q = Tsallis(p, q)`, **`D_q = diversity_number(p, q)`**, and
   **`target_boundaries = round(D_q)`** (token‑feasibility guarded).
   `q < 1` → finer, `q > 1` → coarser, `q = 1` → Shannon perplexity `exp(H)`.
4. P1's deterministic forward **LSTM** runs over a 5‑dim feature sequence derived
   from the qentropy signal and blends `0.65·shift + 0.35·LSTM` to rank gaps.
5. Structural headings (Article/Chapitre/Section) are protected priors.
6. Paper‑faithful **tree entropy** (`Σ log Z_K`, rate `h_K`) is reported per run.

## How evaluation follows project 2

`pipeline/evaluation.py` scores each chunking with `engine.metrics.evaluate_chunking`
(Precision/Recall/F1, MRR, NDCG@5, ss2fd, SRGT, QCS, retrieval token cost, time)
and picks the winner with the honest policy:
**primary signal** (LLM answer‑correctness when judged, else cosine‑rank mean) →
**bootstrap‑CI ties** over queries → **lowest retrieval token cost** breaks ties.
The same `ga_fitness` is the S7 GA's fitness signal, so the GA optimises exactly
what the final ranking rewards. Offline (no key) it falls back to a label‑free
coherence/separation/balance quality.

---

## Running it

### Backend
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r ../requirements.txt
uvicorn main:app --port 8000
```
Quick offline check (no key needed): `python verify_merge.py`.

### Frontend
```bash
cd frontend
npm install
npm run dev          # Vite dev server; set the backend URL in the sidebar
```

### Endpoints
`POST /upload` · `POST /run/{document_id}` · `GET /status/{job_id}` ·
`GET /results/{job_id}` · `GET /export/{job_id}/{fmt}` · `GET /backends` · `GET /health`.

The `results` payload's **`p2_evaluation`** block is the authoritative output:
per‑strategy Table‑I metrics, per‑strategy qentropy fingerprint (q, D_q, Shannon/
Tsallis bits, entropy rate), and the overall winner + how it was decided.

---

## OpenAI key (optional)

Everything runs **offline** with local embeddings. A key only unlocks three extras:
OpenAI embeddings, **auto‑QA generation** (`engine.qagen`), and the **LLM
answerability judge** (`engine.answerability`).

- There is **no free API tier**; ChatGPT Plus does **not** include API access.
  You need a pay‑as‑you‑go account with billing enabled.
- Cost is negligible: `text-embedding-3-small` (~$0.02 / 1M tokens) and
  `gpt-4o-mini` (~$0.15 / 1M input tokens) → a full run costs a fraction of a cent.

```bash
set OPENAI_API_KEY=sk-...           # Windows (or export on *nix)
set OPENAI_QA_MODEL=gpt-4o-mini     # optional
```
With a key, set the embedding backend to `openai` and toggle the LLM judge in the
sidebar; the metrics gain real queries (MRR/NDCG/Precision/Recall) and the winner
is decided by LLM answer‑correctness.

## Configuration (sidebar / `config.yaml` / `/run` config)

`q_entropy_param` (Tsallis q, −1…1) · `K` (tree branching) · `min_chunk_tokens` ·
`max_chunk_tokens` · `window` · `tau_sem` (S4 merge) · `n_min`/`n_max` (S2) ·
`embedding_backend` · `judge_answerability` · `qa_count` ·
`ga_population`/`ga_generations`.

## Docs

See `DOCUMENTATION_INDEX.md`. Per‑stage notes live in `backend/pipeline/*_README.md`;
the math is in `SCORING_FORMULAS_COMPLETE.md`.
