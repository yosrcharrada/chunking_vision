# S3 — qEntropy Boundary Refinement

## Overview

S3 decides **how many** of S2's candidate boundaries are *real* semantic shifts,
keeps that many, and merges across the rest. After the merge of the two projects,
S3 works **only** with the qentropy engine (`engine/entropy.py`,
`engine/tree_entropy.py`). The old JSD / Hellinger / PMI / depth / drift palette
and the GPT‑2 perplexity validator are **gone**.

Public contract is unchanged so S4–S7 keep working:

```python
refine_boundaries(chunks, config) -> List[Dict]
get_jsd_series(chunks)            -> List[float]   # per-chunk shift series (chart)
```

## Algorithm

1. **Units & shift signal.** Each S2 chunk is a *unit*; embed units with the
   shared `engine.embeddings.EmbeddingService`, window‑average, and compute the
   semantic‑shift signal `dᵢ = 1 − cos(uᵢ, uᵢ₊₁)`.
2. **Significant‑shift distribution.** Keep gaps above the document's own median
   coherence; normalise them into a probability vector `p`
   (`engine.entropy.normalize`).
3. **Generalised entropy drives granularity.**
   - `shannon_bits = engine.entropy.shannon_entropy(p)`
   - `tsallis_bits = engine.entropy.tsallis_entropy(p, q)`
   - **`D_q = engine.entropy.diversity_number(p, q)`** — the *effective number of
     real boundaries*.
   - **`target_boundaries = round(D_q)`**, clamped by a token‑feasibility floor
     (`min_chunk_tokens`). `q ∈ [-1, 1]` from `config["q_entropy_param"]`
     (tuned by the S7 GA). `q<1` → finer, `q>1` → coarser, `q=1` → Shannon
     perplexity `exp(H)`.
4. **LSTM refinement (re‑wired from project 1).** A deterministic forward LSTM
   (fixed seed) runs over a **5‑dim feature sequence derived from the qentropy
   signal** — normalised shift, probability mass `pᵢ`, excess‑over‑baseline,
   pointwise Tsallis weight `pᵢ^q`, structural‑prior flag. Its context‑aware score
   is blended `salience = 0.65·shift + 0.35·LSTM` and used to rank gaps.
5. **Selection.** Open the `target_boundaries` strongest gaps; structural headings
   (Article/Chapitre/Section, `_is_protected_boundary`) are forced in first with a
   relaxed token floor.
6. **Tree entropy.** A divisive K‑ary tree over the units feeds
   `engine.tree_entropy.tree_entropy` → `Σ log Z_K`, entropy rate `h_K`, Tsallis
   tree entropy.

## Output fields added to each chunk

`boundary_type` (`single`|`start`|`hard`|`protected_structure_boundary`) ·
`jsd_score`/`metric_score` (LSTM‑refined shift at the lead boundary, kept for the
chart + `get_jsd_series`) · `boundary_signal` · `q` · `diversity_number` ·
`shannon_bits` · `tsallis_bits` · `entropy_rate` · `s3_stats` (`merged_count`,
`hard_count`, `protected_count`, `mean_signal`, `target_boundaries`,
`tree_entropy`, …) · `thresholds` (`q`, `K`, `target_boundaries`, `baseline`).

## Config keys

`q_entropy_param` (Tsallis q, −1…1) · `K` (tree branching) · `min_chunk_tokens` ·
`window` (shift smoothing) · `embedding_backend`. `use_dq=False` falls back to
opening every significant gap (ablation).

## Notes

- Unit embeddings are memoised per text, so the S7 GA — which re‑runs S3 many
  times with the same units but different `q` — never re‑embeds.
- If the embedder is unavailable (fully offline, no models), S3 falls back to a
  deterministic hash embedding so the stage still runs.
