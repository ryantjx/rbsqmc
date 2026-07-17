"""Tests for Sobol sequence generator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import jax.numpy as jnp
from src.scripts.qmc.qmc import Sobol


class TestSobol:
    """Test cases for Sobol sequence generator."""
    
    def test_initialization_basic(self):
        """Test basic initialization."""
        sobol = Sobol(d=2, scramble=False)
        assert sobol.d == 2
        assert sobol.scramble is False
        assert sobol.bits == 30
        assert sobol._num_generated == 0
        
    def test_initialization_scrambled(self):
        """Test initialization with scrambling."""
        sobol = Sobol(d=5, scramble=True, key=42)
        assert sobol.d == 5
        assert sobol.scramble is True
        assert sobol._sv is not None
        
    def test_max_dimension(self):
        """Test maximum dimension limit."""
        # Should work at max dimension
        sobol = Sobol(d=21201, scramble=False)
        assert sobol.d == 21201
        
        # Should raise error above max
        with pytest.raises(ValueError, match="Maximum dimension is"):
            Sobol(d=21202, scramble=False)
            
    def test_sample_shape(self):
        """Test sample output shape."""
        sobol = Sobol(d=3, scramble=False)
        samples = sobol.sample(n=100)
        
        assert samples.shape == (100, 3)
        assert isinstance(samples, jnp.ndarray)
        
    def test_sample_values_in_range(self):
        """Test that samples are in [0, 1)."""
        sobol = Sobol(d=5, scramble=False)
        samples = sobol.sample(n=1000)
        
        # All values should be in [0, 1)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 1.0)
        
    def test_sample_reproducibility(self):
        """Test that same seed produces same samples."""
        sobol1 = Sobol(d=2, scramble=True, key=42)
        samples1 = sobol1.sample(n=100)
        
        sobol2 = Sobol(d=2, scramble=True, key=42)
        samples2 = sobol2.sample(n=100)
        
        assert jnp.allclose(samples1, samples2)
        
    def test_sample_different_seeds(self):
        """Test that different seeds produce different samples."""
        sobol1 = Sobol(d=2, scramble=True, key=42)
        samples1 = sobol1.sample(n=100)
        
        sobol2 = Sobol(d=2, scramble=True, key=43)
        samples2 = sobol2.sample(n=100)
        
        assert not jnp.allclose(samples1, samples2)
        
    def test_reset(self):
        """Test reset functionality."""
        sobol = Sobol(d=2, scramble=False)
        
        # Generate some samples
        samples1 = sobol.sample(n=50)
        assert sobol._num_generated == 50
        
        # Reset
        sobol.reset()
        assert sobol._num_generated == 0
        
        # Generate same number of samples
        samples2 = sobol.sample(n=50)
        
        # Should be identical
        assert jnp.allclose(samples1, samples2)
        
    def test_continuation(self):
        """Test that sampling can be continued (produces different Gray code positions)."""
        sobol = Sobol(d=2, scramble=False)
        
        # Generate in two batches
        samples1 = sobol.sample(n=50)
        samples2 = sobol.sample(n=50)
        
        # Combined should have 100 unique points
        combined = jnp.vstack([samples1, samples2])
        assert combined.shape == (100, 2)
        
        # All values should be in [0, 1)
        assert jnp.all(combined >= 0.0)
        assert jnp.all(combined < 1.0)
        
    def test_unscrambled_first_point(self):
        """Test first point of unscrambled sequence."""
        sobol = Sobol(d=2, scramble=False)
        samples = sobol.sample(n=1)
        
        # First point should be (0, 0) or very close
        assert samples[0, 0] < 0.01
        assert samples[0, 1] < 0.01
        
    def test_scrambled_not_zero(self):
        """Test that scrambled sequence doesn't start at zero."""
        sobol = Sobol(d=2, scramble=True, key=42)
        samples = sobol.sample(n=1)
        
        # First point should NOT be (0, 0)
        assert samples[0, 0] > 0.01 or samples[0, 1] > 0.01
        
    def test_uniformity_2d(self):
        """Test basic uniformity in 2D."""
        sobol = Sobol(d=2, scramble=False)
        samples = sobol.sample(n=1024)
        
        # Mean should be close to 0.5
        mean = jnp.mean(samples, axis=0)
        assert jnp.allclose(mean, jnp.array([0.5, 0.5]), atol=0.05)
        
        # Values should span most of [0, 1)
        assert jnp.min(samples) < 0.1
        assert jnp.max(samples) > 0.9
        
    def test_high_dimension(self):
        """Test high-dimensional sampling."""
        sobol = Sobol(d=100, scramble=False)
        samples = sobol.sample(n=100)
        
        assert samples.shape == (100, 100)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 1.0)
        
    def test_single_sample(self):
        """Test sampling single point."""
        sobol = Sobol(d=2, scramble=False)
        sample = sobol.sample(n=1)
        
        assert sample.shape == (1, 2)
        
    def test_large_n(self):
        """Test large sample size."""
        sobol = Sobol(d=2, scramble=False)
        samples = sobol.sample(n=10000)
        
        assert samples.shape == (10000, 2)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 1.0)
        
    def test_dtype(self):
        """Test output dtype."""
        sobol = Sobol(d=2, scramble=False)
        samples = sobol.sample(n=10)
        
        # Should be float32
        assert samples.dtype == jnp.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
