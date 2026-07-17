"""Tests for Halton QMC sequence generator.

Uses pytest style following conventions from the cuthbert library.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "scripts" / "qmc"))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from qmc import Halton


class TestHaltonBasic:
    """Basic functionality tests for Halton sequence."""

    @pytest.mark.parametrize("n", [1, 10, 100])
    @pytest.mark.parametrize("d", [1, 2, 5, 10])
    def test_output_shape(self, n, d):
        """Test that output shape is (n, d)."""
        halton = Halton(d=d, scramble=False)
        samples = halton.sample(n=n)
        assert samples.shape == (n, d), f"Expected shape ({n}, {d}), got {samples.shape}"

    def test_values_in_range(self):
        """Test that all values are in [0, 1)."""
        halton = Halton(d=5, scramble=False)
        samples = halton.sample(n=1000)
        assert jnp.all(samples >= 0), "Found negative values"
        assert jnp.all(samples < 1), "Found values >= 1"

    def test_no_nan_or_inf(self):
        """Test that there are no NaN or Inf values."""
        halton = Halton(d=10, scramble=False)
        samples = halton.sample(n=1000)
        assert not jnp.any(jnp.isnan(samples)), "Found NaN values"
        assert not jnp.any(jnp.isinf(samples)), "Found Inf values"


class TestHaltonDeterminism:
    """Tests for determinism and reproducibility."""

    def test_determinism_non_scrambled(self):
        """Test that non-scrambled Halton is deterministic."""
        halton1 = Halton(d=3, scramble=False)
        halton2 = Halton(d=3, scramble=False)
        samples1 = halton1.sample(n=100)
        samples2 = halton2.sample(n=100)
        assert jnp.allclose(samples1, samples2), "Non-scrambled sequences differ"

    def test_determinism_scrambled(self):
        """Test that scrambled Halton is deterministic with same key."""
        key = jax.random.PRNGKey(42)
        halton1 = Halton(d=3, scramble=True, key=key)
        halton2 = Halton(d=3, scramble=True, key=key)
        samples1 = halton1.sample(n=100)
        samples2 = halton2.sample(n=100)
        assert jnp.allclose(samples1, samples2), "Scrambled sequences with same key differ"

    def test_different_keys_different_sequences(self):
        """Test that different keys produce different scrambled sequences."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)
        halton1 = Halton(d=2, scramble=True, key=key1)
        halton2 = Halton(d=2, scramble=True, key=key2)
        samples1 = halton1.sample(n=100)
        samples2 = halton2.sample(n=100)
        assert not jnp.allclose(samples1, samples2), "Different keys produced same sequence"


class TestHaltonCorrectness:
    """Tests for correctness of Halton sequence values."""

    def test_expected_halton_values(self):
        """Test first few values match expected Halton sequence.
        
        Base 2: 0, 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
        Base 3: 0, 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9, ...
        """
        halton = Halton(d=2, scramble=False)
        samples = halton.sample(n=8)
        
        expected_dim0 = jnp.array([0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875])  # base 2
        expected_dim1 = jnp.array([0.0, 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9])  # base 3
        
        assert jnp.allclose(samples[:, 0], expected_dim0, atol=1e-6), f"Dim 0 mismatch: {samples[:, 0]} vs {expected_dim0}"
        assert jnp.allclose(samples[:, 1], expected_dim1, atol=1e-6), f"Dim 1 mismatch: {samples[:, 1]} vs {expected_dim1}"

    def test_scrambled_vs_non_scrambled_different(self):
        """Test that scrambled and non-scrambled produce different results."""
        halton_non_scrambled = Halton(d=2, scramble=False)
        halton_scrambled = Halton(d=2, scramble=True, key=jax.random.PRNGKey(0))
        
        samples_non_scrambled = halton_non_scrambled.sample(n=100)
        samples_scrambled = halton_scrambled.sample(n=100)
        
        assert not jnp.allclose(samples_non_scrambled, samples_scrambled), "Scrambled equals non-scrambled"

    def test_multiple_dimensions(self):
        """Test that higher dimensions use correct prime bases."""
        halton = Halton(d=5, scramble=False)
        samples = halton.sample(n=10)
        
        # Check bases are [2, 3, 5, 7, 11]
        expected_bases = [2, 3, 5, 7, 11]
        assert list(halton.base) == expected_bases, f"Bases mismatch: {list(halton.base)} vs {expected_bases}"


class TestHaltonUniformity:
    """Tests for uniformity properties of Halton sequence."""

    def test_uniformity(self):
        """Test that values are roughly uniform (basic histogram test)."""
        halton = Halton(d=1, scramble=False)
        samples = halton.sample(n=10000).flatten()
        
        # Check histogram is roughly uniform across 10 bins
        hist, _ = np.histogram(np.array(samples), bins=10, range=(0, 1))
        expected_count = 10000 / 10
        # Allow 20% deviation
        assert np.all(np.abs(hist - expected_count) < expected_count * 0.2), f"Non-uniform distribution: {hist}"

    def test_scrambled_uniformity(self):
        """Test that scrambled sequence is still uniform."""
        key = jax.random.PRNGKey(42)
        halton = Halton(d=1, scramble=True, key=key)
        samples = halton.sample(n=10000).flatten()
        
        hist, _ = np.histogram(np.array(samples), bins=10, range=(0, 1))
        expected_count = 10000 / 10
        assert np.all(np.abs(hist - expected_count) < expected_count * 0.25), f"Scrambled non-uniform: {hist}"


class TestHaltonEdgeCases:
    """Tests for edge cases."""

    def test_n1_d1(self):
        """Test minimal case: n=1, d=1."""
        halton = Halton(d=1, scramble=False)
        samples = halton.sample(n=1)
        assert samples.shape == (1, 1)
        assert samples[0, 0] == 0.0, "First sample should be 0"

    def test_large_n(self):
        """Test large n for performance/smoke test."""
        halton = Halton(d=2, scramble=False)
        samples = halton.sample(n=100000)
        assert samples.shape == (100000, 2)
        assert jnp.all(samples >= 0) and jnp.all(samples < 1)
