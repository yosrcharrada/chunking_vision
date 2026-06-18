# S6 — Embeddings (shared EmbeddingService)

## Overview

After the merge, S6 standardises on the qentropy engine's single embedding
authority, `engine/embeddings.py::EmbeddingService`. The old multi‑model
"ensemble" (mxbai‑embed‑large + MiniLM + mpnet + jina with a shared JL
projection) is **removed** — every embedding in the system (S3 unit shifts, S6
chunk vectors, the metrics, and the GA fitness) now lives in **one** vector
space, so cosine numbers are directly comparable everywhere.

## Backends (best → fallback)

| id | model | notes |
|----|-------|-------|
| `openai` | text-embedding-3-small (1536‑d) | default when `OPENAI_API_KEY` is set |
| `openai-large` | text-embedding-3-large (3072‑d) | strongest, needs key |
| `multilingual` | paraphrase-multilingual-MiniLM-L12-v2 (384‑d) | offline default |
| `english` | all-MiniLM-L6-v2 (384‑d) | English‑only |
| `tfidf` | TF‑IDF + SVD | always available last resort |

Selected via `config["embedding_backend"]` (or the sidebar / `GET /backends`).
All backends return L2‑normalised float32 vectors (cosine == dot product);
token counting uses `tiktoken` when available.

## API

```python
embed_chunks(chunks, full_text, doc_profile, model_name, config) -> (chunks, embeddings)
embed_texts(texts, model_name="")                                -> List[np.ndarray]
```

`embed_chunks` prepends a domain‑aware context header to longer documents
(`_build_input_texts`) and then calls `get_embedder().embed(...)`. Each chunk gets
`embedding` plus a small `ensemble_embedding` descriptor (`models=[resolved_backend]`,
`projection_dim=len(vec)`) kept for frontend compatibility.

## Why this matters

The original ensemble averaged projected vectors from several models; with one
shared `EmbeddingService` there is no projection mismatch, the metrics in
`engine/metrics.py` and the GA fitness in `pipeline/evaluation.py` see the same
geometry as the chunks, and OpenAI calls are batched + parallelised so large
documents stay fast.
