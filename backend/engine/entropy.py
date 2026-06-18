"""Generalized entropy functionals used to drive and measure chunking.

Two families are implemented:

  * Shannon entropy   H(p)   = -sum p_i log p_i              (the paper's baseline)
  * Tsallis q-entropy S_q(p) = (1 - sum p_i^q) / (q - 1)     (the proposed extension)

At q -> 1, Tsallis reduces to Shannon exactly (L'Hopital), so the original
Zhong et al. (2026) model is recovered as the q=1 special case.

The *diversity number* (Hill number) D_q = (sum p_i^q)^(1/(1-q)) is the
"effective number of outcomes" implied by an entropy.  D_1 = exp(H) is the
familiar perplexity.  We use D_q to translate an entropy into a concrete
number of chunk boundaries, which is what lets q actually change the chunking
(not just the measured entropy number).

q is restricted to [-1, 1] per the platform spec.  All probabilities fed to
these functions are strictly positive (we normalise with a floor), so every
expression below is finite even for q <= 0.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

EPS = 1e-12
# log base 2 -> entropies reported in bits, matching Shannon's "bits/character".
_LOG2 = math.log(2.0)


def normalize(weights: Sequence[float]) -> np.ndarray:
    """Turn non-negative weights into a strictly-positive probability vector."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= EPS:
        # Degenerate (all-zero) signal -> uniform distribution.
        return np.full(len(w), 1.0 / max(len(w), 1))
    p = w / total
    # Floor + renormalise so p_i^q is finite for q <= 0.
    p = np.clip(p, EPS, None)
    return p / p.sum()


def shannon_entropy(p: Sequence[float]) -> float:
    """Shannon entropy in bits: H = -sum p_i log2 p_i."""
    p = normalize(p)
    return float(-np.sum(p * np.log(p)) / _LOG2)


def tsallis_entropy(p: Sequence[float], q: float) -> float:
    """Tsallis q-entropy.  Returns Shannon entropy (bits) when q ~ 1."""
    p = normalize(p)
    if abs(q - 1.0) < 1e-9:
        return shannon_entropy(p)
    # Sq in nats-like units; (1 - sum p^q)/(q-1). Report in bits for comparability.
    sq = (1.0 - np.sum(np.power(p, q))) / (q - 1.0)
    return float(sq / _LOG2)


def diversity_number(p: Sequence[float], q: float) -> float:
    """Hill number D_q = (sum p_i^q)^(1/(1-q)); D_1 = exp(Shannon H)."""
    p = normalize(p)
    if abs(q - 1.0) < 1e-9:
        h_nats = -np.sum(p * np.log(p))
        return float(math.exp(h_nats))
    s = float(np.sum(np.power(p, q)))
    s = max(s, EPS)
    return float(s ** (1.0 / (1.0 - q)))


def entropy_value(p: Sequence[float], q: float) -> float:
    """Dispatch: Shannon at q=1, Tsallis otherwise (always in bits)."""
    return tsallis_entropy(p, q)
