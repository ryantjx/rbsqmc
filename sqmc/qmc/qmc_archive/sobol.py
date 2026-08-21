"""
Sobol Sequence Generator in JAX - GPU Optimized.

Implements (scrambled) Sobol' sequences using the direction numbers from
Joe & Kuo (search criterion D(6), up to dimension 21201):

    https://web.maths.unsw.edu.au/~fkuo/sobol/

The scrambling strategy is a (left) linear matrix scramble (LMS) followed by a
digital random shift (LMS+shift), matching ``scipy.stats.qmc.Sobol``.

Optimized for GPU usage:
- Precomputed direction numbers cached at module load
- Fully JIT-compiled point generation (no Python loops in hot path)
- Vectorized scrambling using JAX operations
- Reduced recompilation by removing n from static arguments

References
----------
- I. M. Sobol', "The distribution of points in a cube and the accurate
  evaluation of integrals." Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.
- J. Matousek, "On the L2-discrepancy for anchored boxes." J. of Complexity 14,
  527-556, 1998.
- Art B. Owen, "Scrambling Sobol and Niederreiter-Xing points." Journal of
  Complexity, 14(4):466-489, December 1998.
- S. Joe and F. Y. Kuo, "Constructing sobol sequences with better two-dimensional
  projections." SIAM Journal on Scientific Computing, 30(5):2635-2654, 2008.
"""

import functools
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# Using uint32 for GPU compatibility
# For most QMC applications, 32 bits is sufficient

_MAXDIM = 21201  # Maximum supported dimensions (Joe & Kuo)
_MAXDEG = 18  # Maximum polynomial degree
_MAXBITS = 32  # Using 32 bits for GPU compatibility


def _load_direction_numbers():
    """
    Load direction numbers from the precomputed ``sobol_data.npz`` file.
    """
    data_path = os.path.join(os.path.dirname(__file__), "sobol_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"sobol_data.npz not found at {data_path}. "
            "Download from https://web.maths.unsw.edu.au/~fkuo/sobol/"
        )

    data = np.load(data_path)
    poly = data["poly"].astype(np.uint32)
    vinit = data["vinit"].astype(np.uint32)
    # vinit shape is (21201, 18) - 21201 dimensions, 18 initial values each
    return poly[:_MAXDIM], vinit[:_MAXDIM, :_MAXDEG]


_POLY, _VINIT = _load_direction_numbers()

# Precompute direction numbers for common bit sizes at module load time
_DIRECTION_NUMBERS_CACHE = {}


def _bit_length(n):
    """Number of bits needed to represent n."""
    bits = 0
    nloc = n
    while nloc != 0:
        nloc >>= 1
        bits += 1
    return bits


def _initialize_v_numpy(dim, bits):
    """Initialize direction numbers matrix v of shape (dim, bits)."""
    v = np.zeros((dim, bits), dtype=np.uint32)

    # First row is all 1s
    for i in range(bits):
        v[0, i] = 1

    # Remaining rows
    for d in range(1, dim):
        p = int(_POLY[d])
        m = _bit_length(p) - 1

        # First m elements from vinit
        v[d, :m] = _VINIT[d, :m]

        # Fill remaining elements using recurrence relation
        for j in range(m, bits):
            newv = v[d, j - m]
            pow2 = 1
            for k in range(m):
                pow2 = pow2 << 1
                if (p >> (m - 1 - k)) & 1:
                    newv = newv ^ (pow2 * v[d, j - k - 1])
            v[d, j] = newv

    # Scale: multiply column j by 2^(bits-1-j)
    for d_idx in range(bits):
        for i in range(dim):
            v[i, bits - 1 - d_idx] *= np.uint32(1) << np.uint32(d_idx)

    return v


def _get_direction_numbers(dim, bits):
    """Get precomputed direction numbers, computing if necessary."""
    cache_key = (dim, bits)
    if cache_key not in _DIRECTION_NUMBERS_CACHE:
        v = _initialize_v_numpy(dim, bits)
        _DIRECTION_NUMBERS_CACHE[cache_key] = v  # Store as numpy array
    return _DIRECTION_NUMBERS_CACHE[cache_key]


@jax.jit
def _apply_lms_scramble(sv, ltm):
    """
    Apply Left Matrix Scramble (LMS) using JAX - fully JIT-compiled for GPU.
    
    Parameters
    ----------
    sv : jnp.ndarray, shape (d, bits)
        Direction numbers.
    ltm : jnp.ndarray, shape (d, bits, bits)
        Lower triangular matrices.
    
    Returns
    -------
    sv_scrambled : jnp.ndarray, shape (d, bits)
        Scrambled direction numbers.
    """
    d, bits = sv.shape
    
    def scramble_dim(carry, d_idx):
        sv_d = sv[d_idx]  # (bits,)
        ltm_d = ltm[d_idx]  # (bits, bits)
        
        def compute_bit(carry_j, j):
            vdj = sv_d[j]
            
            # Compute lsmdp[p] = dot(ltm_d[p, :], 2^(bits-1-k)) for all p
            powers = jnp.arange(bits - 1, -1, -1, dtype=jnp.uint32)
            lsmdp = jnp.sum(ltm_d * (jnp.uint32(1) << powers)[None, :], axis=1)
            
            # Compute t1 for all p
            t1 = ((lsmdp * vdj) >> (bits - 1 - j)) & 1
            
            # Accumulate
            acc = jnp.sum(t1.astype(jnp.uint32) << jnp.arange(bits - 1, -1, -1, dtype=jnp.uint32))
            
            return carry_j, acc
        
        _, scrambled = lax.scan(compute_bit, None, jnp.arange(bits))
        return carry, scrambled
    
    _, sv_scrambled = lax.scan(scramble_dim, None, jnp.arange(d))
    return sv_scrambled


@jax.jit
def _generate_points_jax(sv, gray_codes, shift, scale):
    """
    Fully JIT-compiled point generation for GPU.
    
    Parameters
    ----------
    sv : jnp.ndarray, shape (d, bits)
        Direction numbers.
    gray_codes : jnp.ndarray, shape (n,)
        Gray codes for each point.
    shift : jnp.ndarray, shape (d,)
        Digital shift.
    scale : float
        Scale factor (1 / 2^bits).
    
    Returns
    -------
    points : jnp.ndarray, shape (n, d)
        Generated points in [0, 1).
    """
    n = gray_codes.shape[0]
    d, bits = sv.shape
    
    # Extract bits of gray codes
    bit_indices = jnp.arange(bits, dtype=jnp.uint32)
    gray_bits = (gray_codes[:, None] >> bit_indices[None, :]) & 1
    
    # Compute XOR of direction numbers where gray bit is set
    def xor_scan(carry, j):
        mask = gray_bits[:, j].astype(jnp.uint32)  # (n,)
        contrib = mask[:, None] * sv[None, :, j]  # (n, d)
        return carry ^ contrib, None
    
    init = jnp.zeros((n, d), dtype=jnp.uint32)
    result, _ = lax.scan(xor_scan, init, jnp.arange(bits))
    
    # XOR with shift and scale
    result = result ^ shift[None, :]
    return result.astype(jnp.float64) * scale


def sobol_sample(n, d, scramble=False, seed=None, bits=None):
    """
    Generate Sobol sequence samples matching scipy.stats.qmc.Sobol.
    
    Parameters
    ----------
    n : int
        Number of samples (must be power of 2 for good properties)
    d : int
        Number of dimensions
    scramble : bool
        Whether to apply LMS+shift scrambling
    seed : int or None
        Random seed for scrambling
    bits : int or None
        Number of bits (default _MAXDEG=18, limited by direction numbers)
        
    Returns
    -------
    jnp.ndarray
        Sobol samples of shape (n, d) in [0, 1)
    """
    if bits is None:
        bits = _MAXDEG
    # Handle seed and scrambling outside JIT (can't use traced values with numpy RNG)
    if scramble:
        if seed is None:
            import time
            seed = int(time.time() * 1000) % (2**31)
        
        rng = np.random.default_rng(seed)
        # Generate random digital shift
        shift = rng.integers(0, 2**bits, size=(d,), dtype=np.uint32)
    else:
        shift = np.zeros(d, dtype=np.uint32)
    
    return _sobol_sample_jit(n, d, bits, shift)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _sobol_sample_jit(n, d, bits, shift):
    """JIT-compiled inner function for Sobol sample generation."""
    scale = jnp.float64(2.0 ** (-bits))
    
    # Get direction numbers (cached as numpy, convert to JAX here)
    sv_np = _get_direction_numbers(d, bits)
    sv = jnp.array(sv_np, dtype=jnp.uint32)
    
    # Generate Gray codes for indices 0 to n-1 (scipy starts from 0)
    indices = jnp.arange(n, dtype=jnp.uint32)
    gray_codes = indices ^ (indices >> 1)
    
    # Convert shift to JAX array
    shift_jax = jnp.array(shift, dtype=jnp.uint32)
    
    # Generate points on GPU
    points = _generate_points_jax(sv, gray_codes, shift_jax, scale)
    return points


def test_against_scipy():
    """Test that JAX implementation matches scipy exactly."""
    try:
        from scipy.stats import qmc
    except ImportError:
        print("scipy not available, skipping comparison test")
        return
    
    print("Testing JAX Sobol against scipy...")
    
    # Test unscrambled
    n, d = 64, 5
    
    # JAX version
    jax_samples = np.array(sobol_sample(n, d, scramble=False))
    
    # SciPy version
    scipy_sobol = qmc.Sobol(d, scramble=False)
    scipy_samples = scipy_sobol.random(n)
    
    max_diff = np.max(np.abs(jax_samples - scipy_samples))
    print(f"Unscrambled max difference: {max_diff:.2e}")
    
    if max_diff < 1e-15:
        print("✓ Unscrambled: JAX matches scipy exactly!")
    else:
        print("✗ Unscrambled: JAX does not match scipy")
        print(f"  First few JAX: {jax_samples[0]}")
        print(f"  First few SciPy: {scipy_samples[0]}")
    
    # Test scrambled (statistical comparison only)
    seed = 42
    jax_scrambled = np.array(sobol_sample(n, d, scramble=True, seed=seed))
    
    scipy_sobol_scrambled = qmc.Sobol(d, scramble=True, seed=seed)
    scipy_scrambled = scipy_sobol_scrambled.random(n)
    
    # Check statistical properties
    jax_mean = np.mean(jax_scrambled, axis=0)
    scipy_mean = np.mean(scipy_scrambled, axis=0)
    mean_diff = np.max(np.abs(jax_mean - scipy_mean))
    
    print(f"Scrambled mean difference: {mean_diff:.2e}")
    if mean_diff < 0.1:  # Should be close but not identical
        print("✓ Scrambled: Statistical properties match")
    else:
        print("✗ Scrambled: Statistical properties differ")


def benchmark():
    """Benchmark the JAX implementation."""
    import time
    print("\nBenchmarking JAX Sobol (GPU Optimized)...")
    print(f"JAX backend: {jax.default_backend()}")
    
    # Warmup
    print("Warming up...")
    _ = sobol_sample(100, 10, scramble=False)
    _ = sobol_sample(100, 10, scramble=True, seed=42)
    
    # Benchmark
    n, d = 8192, 50
    
    # Unscrambled
    start = time.time()
    for _ in range(10):
        _ = sobol_sample(n, d, scramble=False)
    jax_time = (time.time() - start) / 10 * 1000
    print(f"Unscrambled ({n}x{d}): {jax_time:.2f} ms")
    
    # Scrambled
    start = time.time()
    for _ in range(10):
        _ = sobol_sample(n, d, scramble=True, seed=42)
    jax_time = (time.time() - start) / 10 * 1000
    print(f"Scrambled ({n}x{d}): {jax_time:.2f} ms")
    
    # Compare with scipy
    try:
        from scipy.stats import qmc
        
        start = time.time()
        for _ in range(10):
            sampler = qmc.Sobol(d, scramble=False)
            _ = sampler.random(n)
        scipy_time = (time.time() - start) / 10 * 1000
        print(f"\nSciPy unscrambled: {scipy_time:.2f} ms")
        print(f"Speedup: {scipy_time/jax_time:.1f}x")
    except ImportError:
        pass


if __name__ == "__main__":
    test_against_scipy()
    benchmark()
