from engine.segeval import boundary_jaccard, pk, window_diff


def test_perfect_agreement_is_zero_error():
    ref = [4, 9, 14]
    assert pk(ref, ref, 20) == 0.0
    assert window_diff(ref, ref, 20) == 0.0


def test_total_disagreement_is_high_error():
    ref = [10]
    hyp = []  # misses the only boundary
    assert pk(ref, hyp, 22) > 0.3
    assert window_diff(ref, hyp, 22) > 0.3


def test_near_miss_better_than_far_miss():
    ref = [10]
    near = window_diff(ref, [11], 30)
    far = window_diff(ref, [25], 30)
    assert near <= far


def test_degenerate_inputs():
    assert pk([], [], 1) == 0.0
    assert window_diff([], [], 0) == 0.0
    assert pk([], [], 10) == 0.0  # no boundaries at all: agree everywhere


def test_boundary_jaccard():
    assert boundary_jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert boundary_jaccard([], []) == 1.0
    assert boundary_jaccard([1, 2], [3, 4]) == 0.0
    assert abs(boundary_jaccard([1, 2, 3], [2, 3, 4]) - 0.5) < 1e-9
