# Scoring & Formulas — qEntropy edition

This matches the in‑app **Scoring Formulas** panel
(`frontend/src/components/ScoringFormulas.jsx`). The old JSD/PPL/ensemble formulas
are obsolete and have been removed.

---

## 1. Tsallis q‑entropy (S3) — `engine/entropy.py`

Let `p` be the normalised distribution over significant semantic‑shift gaps.

```
S_q(p) = (1 − Σ pᵢ^q) / (q − 1)          (Tsallis, in bits)
H(p)   = − Σ pᵢ log₂ pᵢ                   (Shannon; S_q → H as q → 1)
```

`q ∈ [-1, 1]` (platform spec). At `q = 1` the platform recovers the Zhong et al.
Shannon model exactly.

- `q < 1` amplifies weak/rare shifts → **finer** chunks
- `q = 1` Shannon entropy (perplexity baseline)
- `q > 1` suppresses weak shifts → **coarser** chunks

## 2. Hill diversity number (S3 boundary‑count driver)

```
D_q = (Σ pᵢ^q)^(1/(1−q))      D_1 = exp(H)      target_boundaries = round(D_q)
```

`D_q` is the *effective number of real boundaries* and is non‑increasing in `q`,
so `q` alone decides how many of the semantic‑shift peaks become boundaries. This
is what makes Shannon (`q=1`) and Tsallis (`q≠1`) produce genuinely different
chunkings — the entropy *drives* the granularity, it doesn't merely measure it.

## 3. LSTM boundary refinement (S3)

```
salienceᵢ = 0.65 · d̂ᵢ + 0.35 · σ(Wₚ · hᵢ),   hᵢ = LSTM(x₀ … xᵢ)
```

A deterministic forward LSTM (fixed seed → reproducible) adds document‑order
context over a 5‑dim qentropy feature vector `xᵢ = [d̂ᵢ, pᵢ, excessᵢ, pᵢ^q,
structuralᵢ]` before the top `D_q` gaps are opened.

## 4. Tree entropy (S3 fingerprint) — `engine/tree_entropy.py`

```
Z_K(n) = C(n + K − 1, n − 1)      (K‑way split multiplicity)
H(N)   = Σ_internal log Z_K(size)  (Zhong et al., Eq. 9)
h_K    = H(N) / N                  (entropy rate, Eq. 11)
```

Reported per document as Shannon nats, entropy rate `h_K`, and a Tsallis‑q tree
entropy `S_q` (→ `H` as `q → 1`).

## 5. Table‑I retrieval metrics (evaluation) — `engine/metrics.py`

All cosines use L2‑normalised embeddings. For each query, chunks are ranked by
`cos(query, chunk)`; relevance threshold is auto‑calibrated to the model's cosine
scale.

| metric | definition | ranking? |
|--------|-----------|----------|
| Precision / Recall / F1 | cosine‑threshold TP/FP/FN | diagnostic only |
| MRR | reciprocal rank of first relevant chunk | ✓ |
| NDCG@5 | graded relevance `max(0, cos(chunk, GT))` | ✓ |
| ss2fd | mean `cos(chunk, full‑document)` | — |
| SRGT | mean `cos(top‑5 retrieved, GT)` | ✓ |
| QCS | mean `cos(query, top‑5 retrieved)` | ✓ |
| Retrieval token cost | mean tokens across top‑5 retrieved | lower better |
| Chunk time | wall‑clock ms to chunk | lower better |

**Winner policy** (`pipeline/evaluation.py::rank_runs`):
primary signal (LLM answer‑correctness when judged, else mean of MRR/NDCG/SRGT/QCS)
→ bootstrap‑CI ties over queries → lowest retrieval token cost breaks ties.
Precision/Recall/F1 are excluded from ranking because both sides are cosine rules
(double counting).

## 6. S7 GA fitness — `pipeline/evaluation.py::ga_fitness`

```
fitness = mean(MRR, NDCG, SRGT, QCS)      # or LLM answer‑correct when judged
        = label‑free quality              # offline fallback (no queries)
```

The GA evolves `n_min`, `n_max`, `tau_sem`, **`q ∈ [-1, 1]`**, and
`min_chunk_tokens`, scoring each individual with the same signal that decides the
final ranking — so the GA optimises exactly what is rewarded after it.
