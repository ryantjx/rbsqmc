"""
Halton Sequence Generator

1. Generate first d prime numbers (for d dimensions)
2. For n index in base b: n = a_0 + a_1 * b + a_2 * b^2 + ... + a_k * b^k
3. Compute radical inverse: phi_b(n) = a_0 * b^-1 + a_1 * b^-2 + a_2 * b^-3 + ... + a_k * b^-(k+1)
4. Randomization: Apply a random shift to the sequence to reduce correlation and improve uniformity. Apply Owen (2017) scambling.

https://en.wikipedia.org/wiki/Halton_sequence

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html

https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/mcmc/sample_halton_sequence_lib.py
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

_MAX_DIMENSION = 10000

def _compute_max_bits(n: int) -> int:
    """Compute max_bits needed for radical inverse based on n."""
    import math
    return min(int(math.ceil(math.log2(n + 1))) + 2, 32)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 4))
def halton_sample(
    n: int,
    d: int,
    scramble: bool = False,
    seed: jax.Array = jax.random.PRNGKey(0),
    max_bits: int = 32,
) -> jnp.ndarray:
    """
    Generate a Halton sequence of n points in d dimensions.

    Parameters
    ----------
    n : int
        Number of points to generate.
    d : int
        Number of dimensions.
    scramble : bool, optional
        Whether to apply scrambling to the sequence (default: False).
    seed : jax.Array, optional
        Random seed for scrambling (default: jax.random.PRNGKey(0)).
    max_bits : int, optional
        Maximum number of bits for radical inverse (default: 32).

    Returns
    -------
    jnp.ndarray
        An array of shape (n, d) containing the Halton sequence points.
    """
    primes = _PRIMES[:d]

    # n is static_argnums, so jnp.arange works
    indices = jnp.arange(1, n + 1, dtype=jnp.uint32)  # Halton sequence starts from index 1

    # Use vmap to compute radical inverse for each prime (dimension)
    samples = jax.vmap(
        lambda base: _radical_inverse_fast(indices, base, max_bits), in_axes=0, out_axes=1
    )(primes)

    if scramble:
        samples = _randomize_fast(samples, primes, seed)
    return samples


@functools.partial(jax.jit, static_argnums=(2,))
def _radical_inverse_fast(indices: jnp.ndarray, base: int, max_bits: int) -> jnp.ndarray:
    """
    Fully JIT-compiled radical inverse using scan.
    No Python loops - runs entirely on accelerator.
    """
    def step(carry, _):
        result, idx, factor = carry
        digit = idx % base
        result = result + factor * digit.astype(jnp.float32)
        idx = idx // base
        factor = factor / base
        return (result, idx.astype(jnp.uint32), factor), None
    
    init = (jnp.zeros_like(indices, dtype=jnp.float32), 
            indices, 
            jnp.float32(1.0 / base))
    
    (result, _, _), _ = jax.lax.scan(step, init, None, length=max_bits)
    return result


@jax.jit
def _randomize_fast(
    samples: jnp.ndarray,
    bases: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Fully JIT-compiled randomization.
    Vectorized shift application.
    """
    n, d = samples.shape
    # Generate random shifts for each dimension
    shifts = jax.random.uniform(key, (d,)) / bases
    # Broadcast and apply shift modulo 1
    return (samples + shifts[jnp.newaxis, :]) % 1.0

def radical_inverse_jax(indices: jnp.ndarray, base: int, max_bits: int) -> jnp.ndarray:
    """
    Vectorized radical inverse using scan.
    
    indices: shape (n,)
    returns: shape (n,)
    """
    def step(carry, _):
        result, idx, factor = carry
        digit = idx % base
        result += factor * digit.astype(jnp.float32)
        idx = (idx // base).astype(indices.dtype)
        factor /= base
        return (result, idx, factor), None
    
    init = (jnp.zeros_like(indices, dtype=jnp.float32), 
            indices, 
            1.0 / base)
    
    (result, _, _), _ = jax.lax.scan(step, init, None, length=max_bits)
    return result

def _randomize(
    samples: jnp.ndarray,
    bases: jnp.ndarray,
    key: jax.Array,
    max_bits: int
) -> jnp.ndarray:
    """
    Apply Owen (2017) randomization to the Halton sequence.
    
    Uses simple random shift scrambling (faster than full Owen scrambling).
    """
    n, d = samples.shape
    
    # Generate random shifts for each dimension
    shifts = jax.random.uniform(key, (d,)) / bases
    
    # Apply shift modulo 1 (broadcast shifts to (n, d))
    return (samples + shifts[None, :]) % 1.0


# Precompute primes on module load
def _primes_less_than(n: int) -> np.ndarray:
    """Generate first n prime numbers in jax array. Generated with numpy Sieve of Eratosthenes."""
    sieve = np.ones(n // 2, dtype=bool)
    for i in range(3, int(n**0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i*i//2::i] = False
    primes = 2 * np.where(sieve)[0] + 1
    primes[0] = 2
    return primes[primes > 0]

_PRIMES = jnp.array(_primes_less_than(104729 + 1))
assert len(_PRIMES) == _MAX_DIMENSION

def main():
    n = 1000
    d = 2
    samples = halton_sample(n, d, scramble=True)
    
    import matplotlib.pyplot as plt
    plt.scatter(samples[:, 0], samples[:, 1], s=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Halton Sequence (scrambled)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.axis('equal')
    plt.show()
if __name__ == "__main__":
    main()