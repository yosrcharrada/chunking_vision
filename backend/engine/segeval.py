"""Reference-based segmentation metrics: Pk and WindowDiff.

These are THE standard metrics for "did you cut in the right places", used by
every text-segmentation benchmark (Choi, Wiki-727k, WikiSection):

  * Pk          (Beeferman, Berger & Lafferty 1999): slide a window of k
                sentences over the text; count positions where the reference
                and the hypothesis disagree on "are the two window ends in the
                same segment?".  Lower is better; ~0.5 is random.
  * WindowDiff  (Pevzner & Hearst 2002): same sliding window, but compares the
                NUMBER of boundaries inside the window, penalising near-misses
                and false positives more evenly.  Lower is better.

Boundaries are gap indices: boundary i means "split after sentence i", matching
ChunkingResult.boundaries.  k defaults to half the mean reference segment
length, the standard convention.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def _boundary_vector(boundaries: Iterable[int], n_sentences: int) -> np.ndarray:
    b = np.zeros(max(n_sentences - 1, 0), dtype=bool)
    for g in boundaries:
        if 0 <= int(g) < n_sentences - 1:
            b[int(g)] = True
    return b


def _default_k(ref: np.ndarray, n_sentences: int) -> int:
    n_segments = int(ref.sum()) + 1
    return max(1, int(round(n_sentences / (2.0 * n_segments))))


def pk(
    ref_boundaries: Iterable[int],
    hyp_boundaries: Iterable[int],
    n_sentences: int,
    k: Optional[int] = None,
) -> float:
    """Pk error in [0, 1]; lower is better (0 = perfect agreement)."""
    n = int(n_sentences)
    if n < 2:
        return 0.0
    ref = _boundary_vector(ref_boundaries, n)
    hyp = _boundary_vector(hyp_boundaries, n)
    kk = k if k is not None else _default_k(ref, n)
    kk = max(1, min(kk, n - 1))
    pr = np.concatenate([[0], np.cumsum(ref)])
    ph = np.concatenate([[0], np.cumsum(hyp)])
    disagree = 0
    total = 0
    for i in range(0, n - kk):
        same_ref = (pr[i + kk] - pr[i]) == 0
        same_hyp = (ph[i + kk] - ph[i]) == 0
        disagree += int(same_ref != same_hyp)
        total += 1
    return disagree / total if total else 0.0


def window_diff(
    ref_boundaries: Iterable[int],
    hyp_boundaries: Iterable[int],
    n_sentences: int,
    k: Optional[int] = None,
) -> float:
    """WindowDiff error in [0, 1]; lower is better (0 = perfect agreement)."""
    n = int(n_sentences)
    if n < 2:
        return 0.0
    ref = _boundary_vector(ref_boundaries, n)
    hyp = _boundary_vector(hyp_boundaries, n)
    kk = k if k is not None else _default_k(ref, n)
    kk = max(1, min(kk, n - 1))
    pr = np.concatenate([[0], np.cumsum(ref)])
    ph = np.concatenate([[0], np.cumsum(hyp)])
    diff = 0
    total = 0
    for i in range(0, n - kk):
        diff += int((pr[i + kk] - pr[i]) != (ph[i + kk] - ph[i]))
        total += 1
    return diff / total if total else 0.0


def boundary_jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard overlap of two boundary sets (1 = identical chunkings).  Used by
    the ablation to ask: do two methods even produce different chunks?"""
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
