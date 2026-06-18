"""Paper-faithful entropy of a semantic tree (Zhong et al., arXiv:2602.13194).

The paper models the recursive K-ary segmentation of a length-N token text as a
random ordered-partition process.  Conditioned on a parent span of size n, the
K-tuple of child sizes is uniform over weak compositions, so the number of
distinct K-way splittings of a size-n node is the multiplicity

    Z_K(n) = C(n + K - 1, n - 1).                                   (Eq. 3)

The Shannon entropy of the tree ensemble then decomposes additively over nodes
(Eq. 9):

    H(N) = sum over internal nodes of  log Z_K(size(node)),

and is asymptotically extensive, H(N) ~= h_K * N (Eq. 11), so the *entropy rate*
h_K = H(N) / N depends (to leading order) only on the branching factor K.

This module computes those quantities for a concrete semantic tree produced by
the chunker, plus a Tsallis-q generalisation S_q that reduces to the Shannon
value at q -> 1 (our novel extension -- the paper itself uses Shannon only).

Sizes here are measured in *tokens* so that h_K is a per-token entropy rate
directly comparable to the paper's ~2.5 nats/leaf at K=4.
"""

from __future__ import annotations

import math
from typing import List, Optional

# log Z_K(n) via lgamma so large token counts don't overflow.
def log_ZK(n: int, K: int) -> float:
    """log of the K-way splitting multiplicity Z_K(n) = C(n+K-1, n-1)."""
    if K <= 1 or n <= 1:
        return 0.0
    # C(n+K-1, n-1) = C(n+K-1, K)
    a = n + K - 1
    return (
        math.lgamma(a + 1)
        - math.lgamma(K + 1)
        - math.lgamma(a - K + 1)
    )


def _internal_sizes(node, sizes: List[int]) -> None:
    """Collect token sizes of every INTERNAL node (those that were split)."""
    children = getattr(node, "children", None) or []
    if not children:
        return
    sizes.append(int(getattr(node, "tokens", 0)))
    for c in children:
        _internal_sizes(c, sizes)


def tree_entropy(root, K: int, total_tokens: Optional[int] = None,
                 q: float = 1.0) -> dict:
    """Shannon H, entropy rate h_K, and the Tsallis-q tree entropy S_q.

    H   = sum_internal log Z_K(size)                         (paper, Eq. 9)
    h_K = H / N                                              (paper, Eq. 11)
    S_q = sum_internal [ (Z_K(size)^(1-q) - 1) / (1 - q) ]   (our q-extension;
          -> H as q -> 1, since d/d? of Z^(1-q) recovers log Z)

    Returns values in nats (natural log).  N defaults to the root token count.
    """
    if root is None:
        return {"shannon_nats": 0.0, "entropy_rate": 0.0, "tsallis_nats": 0.0,
                "n_internal": 0, "K": K}

    sizes: List[int] = []
    _internal_sizes(root, sizes)
    n_total = int(total_tokens or getattr(root, "tokens", 0) or 1)

    h_shannon = sum(log_ZK(n, K) for n in sizes)

    if abs(q - 1.0) < 1e-9:
        s_q = h_shannon
    else:
        acc = 0.0
        for n in sizes:
            lz = log_ZK(n, K)            # = log Z_K(n)
            # Z^(1-q) = exp((1-q) log Z); guard against overflow for big spans.
            z_pow = math.exp(min((1.0 - q) * lz, 700.0))
            acc += (z_pow - 1.0) / (1.0 - q)
        s_q = acc

    return {
        "shannon_nats": float(h_shannon),
        "entropy_rate": float(h_shannon / max(n_total, 1)),
        "tsallis_nats": float(s_q),
        "n_internal": len(sizes),
        "K": int(K),
    }
