"""
Quasi Monte Carlo (QMC) - return a QMC point set

Deterministic QMC - GridSample, SobolSample, SobolSample, FaureSample, LatticeRuleSample, HaltonSample, GoldenSample, KroneckerSample

Randomized QMC - LatinHypercubeSample, RandomizedHaltonSample
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# Cache for loaded direction numbers data
_SOBOL_A = None
_SOBOL_MINIT = None


def load_sobol_data(filepath: str = None) -> tuple:
    """
    Load Sobol direction numbers from .npz file.
    
    Parameters
    ----------
    filepath : str, optional
        Path to sobol_data.npz. If None, looks in same directory.
        
    Returns
    -------
    sobol_a : jnp.ndarray
        Primitive polynomial coefficients (shape: [21200]).
    sobol_minit : jnp.ndarray
        Starting direction numbers (shape: [40, 21200]).
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "sobol_data.npz")
    
    data = np.load(filepath)
    return jnp.array(data['sobol_a']), jnp.array(data['sobol_minit'])


def compute_direction_numbers(
    d: int,
    sobol_a: jnp.ndarray,
    sobol_minit: jnp.ndarray,
    bits: int = 32
) -> jnp.ndarray:
    """
    Compute direction numbers for dimensions 1 to d.
    
    Uses primitive polynomials (sobol_a) and initial values (sobol_minit)
    to compute full direction number matrix via recurrence.
    
    Parameters
    ----------
    d : int
        Number of dimensions.
    sobol_a : jnp.ndarray
        Primitive polynomial coefficients.
    sobol_minit : jnp.ndarray
        Starting direction numbers.
    bits : int
        Number of bits (default: 32).
        
    Returns
    -------
    m : jnp.ndarray
        Direction numbers matrix of shape (d, bits).
    """
    m = jnp.ones((d, bits), dtype=jnp.uint32)
    
    if d == 0:
        return m
    
    # Dimension 1: m[0, j] = 1 << (bits - j - 1)
    for j in range(bits):
        m = m.at[0, j].set(1 << (bits - j - 1))
    
    if d == 1:
        return m
    
    # Dimensions 2 to d: compute using recurrence
    for dim in range(2, d + 1):
        idx = dim - 2  # Index in sobol_a and sobol_minit
        a = int(sobol_a[idx])
        
        # Find degree (count non-zero initial values)
        degree = 0
        for j in range(40):
            if sobol_minit[j, idx] > 0:
                degree = j + 1
        
        if degree == 0:
            # Fallback: use dimension 1's pattern
            for j in range(bits):
                m = m.at[dim - 1, j].set(1 << (bits - j - 1))
            continue
        
        # Copy initial values from sobol_minit
        for j in range(degree):
            m = m.at[dim - 1, j].set(int(sobol_minit[j, idx]))
        
        # Compute remaining values using recurrence
        # m[j] = m[j-d] XOR (a * m[j-d+1..j-1])
        for j in range(degree, bits):
            new_val = int(m[dim - 1, j - degree])
            ac = a
            for k in range(degree):
                if ac & 1:
                    new_val ^= int(m[dim - 1, j - degree + k]) << (degree - k)
                ac >>= 1
            m = m.at[dim - 1, j].set(new_val)
    
    return m


def get_direction_numbers(d: int, bits: int = 32) -> jnp.ndarray:
    """
    Get direction numbers for d dimensions (cached).
    
    Loads sobol_a and sobol_minit from .npz file on first call,
    then computes direction numbers for requested dimensions.
    
    Parameters
    ----------
    d : int
        Number of dimensions.
    bits : int
        Number of bits (default: 32).
        
    Returns
    -------
    m : jnp.ndarray
        Direction numbers matrix of shape (d, bits).
    """
    global _SOBOL_A, _SOBOL_MINIT
    
    if _SOBOL_A is None or _SOBOL_MINIT is None:
        _SOBOL_A, _SOBOL_MINIT = load_sobol_data()
    
    return compute_direction_numbers(d, _SOBOL_A, _SOBOL_MINIT, bits)


def gray_code(n: jnp.ndarray) -> jnp.ndarray:
    """Compute Gray code: g = n ^ (n >> 1)"""
    return n ^ (n >> 1)


def sobol_sample(
    n: int,
    d: int,
    scramble: bool = False,
    bits: int = 32
) -> jnp.ndarray:
    """
    Return Sobol point set of size n in d dimensions using JAX.
    
    Uses precomputed direction numbers from Joe & Kuo (new-joe-kuo-6.21201).
    
    Parameters
    ----------
    n : int
        Number of points to generate.
    d : int
        Number of dimensions (max 21201).
    scramble : bool
        If True, apply random scrambling (LMS+shift).
    bits : int
        Number of bits for the sequence (default: 32).
    
    Returns
    -------
    jnp.ndarray
        Array of shape (n, d) with points in [0, 1).
    """
    # Load and compute direction numbers (cached)
    direction_numbers = get_direction_numbers(d, bits)
    
    # Generate indices 0 to n-1
    indices = jnp.arange(n, dtype=jnp.uint32)
    
    # Compute Gray code for efficient generation
    gray_indices = gray_code(indices)
    
    # Vectorized Sobol generation
    # For each point: XOR direction numbers where Gray code bits are set
    def compute_point(gray_idx):
        """Compute single point across all dimensions"""
        result = jnp.zeros(d, dtype=jnp.uint32)
        for k in range(bits):
            mask = (gray_idx >> k) & 1
            result = jnp.where(mask, result ^ direction_numbers[:, k], result)
        return result
    
    # Vectorize over all indices
    points_uint = jax.vmap(compute_point)(gray_indices)
    
    # Convert to float in [0, 1)
    points = points_uint.astype(jnp.float64) / (1 << bits)
    
    # Apply scrambling if requested
    if scramble:
        key = jax.random.PRNGKey(0)
        # TODO: Implement LMS + shift scrambling
        pass
    
    return points
    """
    
    direction_numbers = 
