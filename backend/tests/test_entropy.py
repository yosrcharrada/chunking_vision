import numpy as np

from engine import entropy as ent
from engine import tree_entropy as te


def test_normalize_is_probability():
    p = ent.normalize([3.0, 1.0, 0.0, 0.5])
    assert abs(float(np.sum(p)) - 1.0) < 1e-9
    assert (p > 0).all()


def test_tsallis_reduces_to_shannon_at_q1():
    p = ent.normalize([5.0, 3.0, 2.0, 1.0])
    assert abs(ent.tsallis_entropy(p, 1.0) - ent.shannon_entropy(p)) < 1e-9


def test_diversity_number_non_increasing_in_q():
    p = ent.normalize([6.0, 3.0, 2.0, 1.0, 0.5])
    ds = [ent.diversity_number(p, q) for q in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]]
    for a, b in zip(ds, ds[1:]):
        assert a >= b - 1e-9


def test_tree_entropy_q1_equals_shannon():
    # A tiny stand-in tree object with .tokens and .children.
    class N:
        def __init__(self, tokens, children=None):
            self.tokens = tokens
            self.children = children or []

    root = N(10, [N(4, [N(2), N(2)]), N(6)])
    out = te.tree_entropy(root, K=4, q=1.0)
    assert abs(out["tsallis_nats"] - out["shannon_nats"]) < 1e-9
    assert out["shannon_nats"] > 0
    assert out["entropy_rate"] == out["shannon_nats"] / 10
