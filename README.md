# AutoChunker Platform

A full-stack document chunking platform powered by a 7-stage hybrid deterministic pipeline with reinforcement-learning threshold calibration.

---

## Features

- **7-Stage Pipeline**: Profile → Parallel Chunk → Entropy Refinement → Boundary Filter → Graph Enrichment → Contextual Embedding → RL Calibration
- **Multi-format support**: PDF, TXT, Markdown, Python, JS, Java, C/C++ and more
- **Real-time progress** via polling with live stage indicators
- **Chunk Explorer**: color-coded quality view with NER entities and graph neighbours
- **Pipeline Inspector**: JSD boundary chart, RL reward curve, entity graph SVG
- **Configurable**: sidebar sliders for all hyperparameters
- **Export**: JSON, CSV, and Markdown downloads

---

## Repository Structure

```
backend/
  main.py                 ← FastAPI app (all 5 endpoints)
  pipeline/
    __init__.py
    s1_profiler.py        ← Document type / domain / quality metrics
    s2_chunkers.py        ← Recursive · Sliding-window · Structure chunkers
    s3_entropy.py         ← Jensen-Shannon Divergence boundary refinement
    s4_boundary.py        ← CodeBLEU-inspired boundary quality filter
    s5_graph.py           ← spaCy NER + shared-entity chunk graph
    s6_embedding.py       ← Late-chunking / context-header embeddings
    s7_rl.py              ← RL reward calibration loop
frontend/
  index.html              ← Single-page app (vanilla JS + Chart.js)
requirements.txt
README.md
```

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the spaCy model (required for NER)

```bash
python -m spacy download en_core_web_sm
```

> **Note**: spaCy and sentence-transformers are optional — the platform falls
> back to lightweight heuristics when they are unavailable.

### 3. Start the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the frontend

Open `frontend/index.html` directly in your browser (no build step needed).

The sidebar's **Backend URL** field defaults to `http://localhost:8000`.
Adjust it if you deploy the API elsewhere.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload a document; returns `document_id` |
| `POST` | `/run/{document_id}` | Start the pipeline; returns `job_id` |
| `GET`  | `/status/{job_id}` | Poll stage & progress percentage |
| `GET`  | `/results/{job_id}` | Retrieve full chunk results JSON |
| `GET`  | `/export/{job_id}/{fmt}` | Download `json` · `csv` · `markdown` |
| `GET`  | `/health` | Health check |

### Configuration parameters (POST /run body)

| Key | Default | Description |
|-----|---------|-------------|
| `n_min` | 100 | Minimum chunk size (tokens) |
| `n_max` | 500 | Maximum chunk size (tokens) |
| `tau_jsd_low` | 0.15 | JSD merge threshold |
| `tau_jsd_high` | 0.45 | JSD hard-boundary threshold |
| `tau_sem` | 0.75 | Semantic similarity merge threshold |
| `max_iterations` | 10 | RL calibration iterations |
| `alpha` | 0.4 | RL reward: CodeBLEU weight |
| `beta` | 0.4 | RL reward: recall proxy weight |
| `lambda` | 0.2 | RL reward: chunk-count penalty |
| `embedding_model` | `all-MiniLM-L6-v2` | sentence-transformers model |

---

## Pipeline Details

| Stage | Module | Description |
|-------|--------|-------------|
| S1 | `s1_profiler.py` | Classifies type (prose/code/table/mixed), domain, length bucket; computes RC/ICC/DCC/BI/SC quality metrics |
| S2 | `s2_chunkers.py` | Runs recursive character split, sliding-window, and structure-based chunkers in parallel |
| S3 | `s3_entropy.py` | Computes JSD between adjacent chunks; true LSTM cell (input/forget/cell/output gates, cell state cₜ and hidden state hₜ) tracks entropy drift across boundaries for macro topic shift detection |
| S4 | `s4_boundary.py` | Scores each boundary with BLEU n-gram + syntactic overlap + token-type match; merges over-similar pairs |
| S5 | `s5_graph.py` | spaCy NER on every chunk; persistent KGStore boosts edge weights from prior runs; builds shared-entity graph; computes weighted-neighbour graph vectors; writes new entity co-occurrences back to KG |
| S6 | `s6_embedding.py` | Late chunking for short docs; context-header prepend + per-chunk embedding for long docs |
| S7 | `s7_rl.py` | LSTMQAgent encodes state through LSTM, selects actions epsilon-greedily from a tabular Q-table, applies Bellman Q-update Q(s,a)←r+γ·max Q(s′,a′) after each trial; re-runs S2–S6 and keeps improvements |

---

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

Optional (fall-backs are provided):
- `spacy` + `en_core_web_sm` — for named-entity recognition
- `sentence-transformers` + `torch` — for dense embeddings
- `pdfplumber` / `PyPDF2` — for PDF parsing
