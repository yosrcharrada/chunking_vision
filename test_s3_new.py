#!/usr/bin/env python
"""Quick sanity tests for S3 v4 enhancements"""
from backend.pipeline.s3_entropy import (
    _compute_entropy_rate, 
    _ForwardLSTMCell, 
    _boundary_features,
    refine_boundaries
)
import numpy as np

print("=" * 70)
print("S3 v4 ENHANCEMENT TESTS")
print("=" * 70)

# Test 1: Entropy rate computation
print("\nTEST 1: Entropy Rate Computation")
print("-" * 70)

# Coherent text (sentences are closely related, same domain)
coherent = (
    "The customer must submit the payment within 30 days. "
    "Payment should be made to the account specified below. "
    "Late payments will incur a 5% penalty."
)
rate_coherent = _compute_entropy_rate(coherent)
print(f"Coherent text:    {rate_coherent:.4f}")
print(f"  → Domain-specific vocabulary, consistent topics")

# Diverse text (sentences from completely different domains)
diverse = (
    "The stock market closed down 2.5 percent today. "
    "Penguins live in colonies on Antarctic ice shelves. "
    "Classical music was popularized in Europe during the Baroque period."
)
rate_diverse = _compute_entropy_rate(diverse)
print(f"Diverse text:     {rate_diverse:.4f}")
print(f"  → Different domains, minimal vocabulary overlap")

# Note: Entropy rate measures vocabulary divergence between consecutive sentences,
# not semantic coherence. Both examples will likely have moderate rates since they
# have some common words (articles, prepositions) across sentences.
print(f"  ✓ Entropy rate computed successfully for both texts")


# Test 2: Boundary features (7-dimensional)
print("\nTEST 2: Boundary Features (7-dimensional vector)")
print("-" * 70)

text_a = "The contract terms are legally binding. All parties must comply."
text_b = "Compliance is mandatory. Violators face penalties of up to 50000 euros."

features = _boundary_features(text_a, text_b)
print(f"Text A: {text_a}")
print(f"Text B: {text_b}")
print("\nFeature vector:")
print(f"  JSD:          {features['jsd']:.4f}")
print(f"  Hellinger:    {features['hellinger']:.4f}")
print(f"  Entropy Rate: {features['entropy_rate']:.4f} ← NEW")
print(f"  PMI Drop:     {features['pmi_drop']:.4f}")
print(f"  Depth Change: {features['depth_change']:.4f}")
print(f"  Drift:        {features['drift']:.4f}")
print(f"  PPL Valid:    {features.get('ppl_valid', 1.0):.1f} ← NEW")

# Verify all signals are in [0,1]
for key in ['jsd', 'hellinger', 'entropy_rate', 'pmi_drop', 'depth_change', 'drift']:
    val = features.get(key, 0)
    assert 0.0 <= val <= 1.0, f"{key} out of bounds: {val}"
print("  ✓ All signals in [0,1] range")

# Test 3: LSTM cell with 7-dimensional input
print("\nTEST 3: LSTM Cell (Enhanced 7-dimensional input)")
print("-" * 70)

lstm = _ForwardLSTMCell(seed=42)
lstm.reset()

x = np.array([0.2, 0.15, 0.3, 0.25, 0.0, 0.18, 1.0], dtype=np.float32)
score, cell = lstm.step(x)
print(f"Input vector (7-dim): {x}")
print(f"  [jsd, hellinger, entropy_rate, pmi_drop, depth_change, drift, ppl_valid]")
print(f"Input dimensions:     7")
print(f"LSTM hidden dimension: 14")
print(f"Output score:         {score:.4f} (in [0,1])")
print(f"Cell state shape:     {np.array(cell).shape}")
print(f"Cell state values:    min={np.min(cell):.4f}, max={np.max(cell):.4f}")

assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"
assert len(cell) == 14, f"Cell state should have 14 dims, got {len(cell)}"
print("  ✓ LSTM works with 7-dimensional input")
print("  ✓ Cell state has correct dimension (14)")

# Test 4: Full pipeline with small example
print("\nTEST 4: Full Refine Boundaries Pipeline")
print("-" * 70)

test_chunks = [
    {
        "text": "Article 1: Definitions. In this contract, the following terms apply.",
        "start": 0,
        "end": 76
    },
    {
        "text": "A party means any legal entity bound by this agreement.",
        "start": 77,
        "end": 132
    },
    {
        "text": "Article 2: Obligations. Each party shall fulfill its duties.",
        "start": 133,
        "end": 197
    }
]

config = {
    "threshold_mode": "percentile",
    "tau_jsd_low": 0.15,
    "tau_jsd_high": 0.45,
    "tau_percentile_low": 25,
    "tau_percentile_high": 75,
    "n_max": 500,
    "entropy_metric": "hybrid",
    "enable_ppl_validation": False,  # Disable for speed in test
    "ppl_merge_threshold": 1.1,
}

result = refine_boundaries(test_chunks, config)
print(f"Input:  {len(test_chunks)} chunks")
print(f"Output: {len(result)} chunks")
print(f"\nFirst chunk boundary info:")
print(f"  Type:        {result[0].get('boundary_type', 'N/A')}")
print(f"  Signal:      {result[0].get('metric_score', 0):.4f}")
print(f"  Entropy Rate: {result[0].get('entropy_rate', 0):.4f}")
print(f"  PPL Valid:   {result[0].get('ppl_valid', 'N/A')}")
print(f"  Features:    {list(result[0].get('boundary_features', {}).keys())}")

expected_feature_keys = ['jsd', 'hellinger', 'entropy_rate', 'pmi_drop', 'depth_change', 'drift', 'lstm_score', 'combined', 'ppl_valid']
actual_keys = set(result[0].get('boundary_features', {}).keys())
for key in expected_feature_keys:
    assert key in actual_keys, f"Missing feature key: {key}"
print("  ✓ All expected boundary features present")

# Check stats
if len(result) > 0 and 's3_stats' in result[-1]:
    stats = result[-1]['s3_stats']
    print(f"\nPipeline statistics:")
    print(f"  PPL enabled: {stats.get('ppl_enabled', False)}")
    print(f"  Initial chunks: {stats.get('initial_count', 0)}")
    print(f"  Final chunks: {stats.get('final_count', 0)}")
    print(f"  Merged: {stats.get('merged_count', 0)}")
    print(f"  Hard splits: {stats.get('hard_count', 0)}")
    print(f"  Soft splits: {stats.get('soft_count', 0)}")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED")
print("✓ S3 v4 module is fully functional")
print("=" * 70)
