"""
QMC

Sobol

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html

Halton 
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html#scipy.stats.qmc.Halton
"""
from functools import partial
from typing import ClassVar
import math

import jax
import jax.numpy as jnp
import numpy as np
import os

_MAXBITS = 30  # Direction integers file has 30 columns
_DIRECTION_INTEGERS = np.load(os.path.join(os.path.dirname(__file__), 'sobol_data.npz'))['direction_integers']

class QMC:
    def __init__(self, d: int) -> None:
        self._initialize(d=d)
    
    def _initialize(self, d: int) -> None:
        self.d = d
    
    @partial(jax.jit, static_argnums=(0, 1))
    def sample(self, n: int) -> jnp.ndarray:
        pass

class Sobol(QMC):
    MAXDIM: ClassVar[int] = 21201  # Hard limit from direction numbers
    def __init__(self, d: int, scramble : bool = False, key : jax.random.PRNGKey = jax.random.PRNGKey(0)) -> None:
        if d > self.MAXDIM:
            raise ValueError(f"Maximum dimension is {self.MAXDIM}")
        self.scramble = scramble
        self.key = key
        self.bits = _MAXBITS
        # track number of generated samples
        self._num_generated = 0
        self._quasi = None # Current quasi-random state (last generated point)
        super().__init__(d=d)

    def _initialize(self, d: int) -> None:
        super()._initialize(d=d)
        self._initialize_direction_numbers()
        if self.scramble:
            self._scramble()
    
    def _initialize_direction_numbers(self):
        """Load precomputed direction numbers from file."""
        # The file contains precomputed direction numbers (not raw m_i values)
        # Shape: (21201, 30), values are already scaled
        self._sv = jnp.array(
            _DIRECTION_INTEGERS[:self.d, :self.bits], 
            dtype=jnp.uint32
        )
        # Scale factor: max value is around 2^30, so scale by 2^30
        self._scale = 1.0 / (1 << 30)
        # Initialize quasi-random state (current point)
        self._quasi = jnp.zeros(self.d, dtype=jnp.uint32)
    
    def _scramble(self):
        """Apply LMS+shift scrambling."""
        # Generate random shift and LTM using numpy
        # Convert JAX key or int to integer seed for numpy
        if isinstance(self.key, jnp.ndarray):
            seed = int(self.key[0])
        elif isinstance(self.key, int):
            seed = self.key
        else:
            seed = int(self.key)
        rng = np.random.default_rng(seed)
        
        # Digital shift
        shift = rng.integers(0, 2**self.bits, size=self.d, dtype=np.uint64)
        self._shift = jnp.array(shift, dtype=jnp.uint64)
        
        # Lower triangular matrices for LMS
        ltm = np.tril(rng.integers(0, 2, size=(self.d, self.bits, self.bits)))
        self._sv = self._apply_lms(self._sv, jnp.array(ltm))

    @partial(jax.jit, static_argnums=(0,))
    def _apply_lms(self, sv: jnp.ndarray, ltm: jnp.ndarray) -> jnp.ndarray:
        """Apply Left Matrix Scramble (JIT-compiled) over GF(2).
        
        Performs matrix multiplication M @ v where operations are over GF(2):
        - AND for multiplication
        - XOR for addition
        """
        d, bits = sv.shape
        
        def scramble_row(row_idx):
            v = sv[row_idx]  # (bits,)
            M = ltm[row_idx]  # (bits, bits)
            
            # For each output bit i: result[i] = XOR_j(M[i,j] AND v[j])
            # M[i,j] * v[j] is AND (since M is 0/1)
            # Sum over j is XOR (addition in GF(2))
            
            # Compute: result = M @ v over GF(2)
            # Using the fact that XOR of selected bits = bitwise XOR reduction
            
            # Expand for broadcasting: M (bits, bits), v (bits,)
            # M[i,j] * v[j] gives AND result
            and_result = M.astype(jnp.uint32) * v[None, :]  # (bits, bits)
            
            # XOR reduction over columns (j dimension)
            # Start with 0, XOR with each column
            result = jnp.zeros(bits, dtype=jnp.uint32)
            for j in range(bits):
                result = result ^ and_result[:, j]
            
            return result
        
        return jax.vmap(scramble_row)(jnp.arange(d))
    
    @partial(jax.jit, static_argnums=(0, 1))
    def sample(self, n: int) -> jnp.ndarray:
        """
        Generate n Sobol points.
        
        Uses Gray code for efficient O(1) per point generation.
        """
        # Generate Gray codes for indices
        start = self._num_generated
        indices = jnp.arange(start, start + n, dtype=jnp.uint32)
        gray_codes = indices ^ (indices >> 1)
        
        points = self._generate_points(gray_codes)

        self._num_generated += n
        self._quasi = points[-1] if n > 0 else self._quasi
        
        return points

    @partial(jax.jit, static_argnums=(0,))
    def _generate_points(self, gray_codes: jnp.ndarray) -> jnp.ndarray:
        """
        Core point generation - fully JIT-compiled.
        
        For each point n and dimension d:
            x[n,d] = XOR_{k where gray[n]_k=1}(sv[d,k]) * scale
        """
        n = gray_codes.shape[0]
        d, bits = self._sv.shape
        
        # Extract bits of Gray codes: (n, bits) array
        bit_positions = jnp.arange(bits, dtype=jnp.uint32)
        gray_bits = (gray_codes[:, None] >> bit_positions[None, :]) & 1
        
        # XOR direction numbers where Gray code bit is set
        # Using scan for memory efficiency on GPU
        def xor_step(carry, j):
            bit = gray_bits[:, j]  # (n,)
            contrib = bit[:, None].astype(jnp.uint32) * self._sv[:, j]  # (n, d)
            return carry ^ contrib, None
        
        init = jnp.zeros((n, d), dtype=jnp.uint32)
        result, _ = jax.lax.scan(xor_step, init, jnp.arange(bits))
        
        # Apply shift if scrambled
        if self.scramble:
            result = result ^ self._shift[None, :]
        
        # Scale to [0, 1)
        return result.astype(jnp.float32) * self._scale
    
    def reset(self):
        """Reset to initial state."""
        self._num_generated = 0
        self._quasi = jnp.zeros(self.d, dtype=jnp.uint32)
        return self
    @partial(jax.jit, static_argnums=(0,))
    def _compute_point_from_gray(self, gray):
        """Compute single point from Gray code."""
        d, bits = self._sv.shape
        result = jnp.zeros(d, dtype=jnp.uint32)
        for i in range(bits):
            if (gray >> i) & 1:
                result = result ^ self._sv[:, i]
        return result
    
class Halton(QMC):
    def __init__(self, d: int, scramble: bool = False, key : jax.random.PRNGKey = jax.random.PRNGKey(0)) -> None:
        super().__init__(d=d)
        self.scramble = scramble
        self.base = _PRIMES[:d]
        self.key = key
        self.start_index = 0
        if self.scramble:
            self._initialize_permutations()

    @partial(jax.jit, static_argnums=(0, 1))
    def sample(self, n: int) -> jnp.ndarray:
        """Generate n Halton samples in d dimensions."""
        samples = [
            self._van_der_corput(
                n=n,
                base=base,
                start_index=0,
                permutations=perms
            )
            for base, perms in zip(self.base, self._permutations if self.scramble else [None] * self.d)
        ]
        return jnp.stack(samples, axis=1)  # Shape: (n, d)
    
    def _initialize_permutations(self)-> None:
        self._permutations = []
        for bdim in self.base:
            n_perms = math.ceil(54 / math.log2(bdim)) - 1
            # Split key for each permutation
            keys = jax.random.split(self.key, n_perms)
            
            # Generate n_perms random permutations of [0, ..., bdim-1]
            perms = jax.vmap(
                lambda k: jax.random.permutation(k, jnp.arange(bdim))
            )(keys)
            self._permutations.append(perms)
            self.key = jax.random.split(self.key)[1]

    def _van_der_corput(self, n: int, base: int, start_index: int, permutations: jnp.ndarray) -> jnp.ndarray:
        indices = jnp.arange(start_index, start_index + n, dtype=jnp.uint32)

        if self.scramble:
            return self._scrambled_radical_inverse(indices, base, permutations)
        else:
            return self._radical_inverse(indices, base)

    def _radical_inverse(self, indices: jnp.ndarray, base: int) -> jnp.ndarray:
        result = jnp.zeros_like(indices, dtype=jnp.float32)
        denom = 1.0
        # Fixed number of iterations for JIT (based on max digits needed)
        max_digits = 54  # For double precision
        indices = indices.astype(jnp.float64)
        for _ in range(max_digits):
            denom *= base
            result += (indices % base) / denom
            indices = indices // base        
        return result
    def _scrambled_radical_inverse(self, indices: jnp.ndarray, base: int, permutations: jnp.ndarray) -> jnp.ndarray:
        result = jnp.zeros_like(indices, dtype=jnp.float32)
        denom = 1.0
        digit_pos = 0
        max_digits = permutations.shape[0]
        for _ in range(max_digits):
            denom *= base
            digit = indices % base
            # Apply permutation for this digit position
            digit = jnp.where(
                digit_pos < max_digits,
                permutations[digit_pos, digit],
                digit
            )
            result += digit / denom
            indices = indices // base
            digit_pos += 1
        return result

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

# 10,000th prime is 104729, so we generate primes up to 104729 + 1
_PRIMES = _primes_less_than(104729 + 1)
# _PRIMES = _primes_less_than(21201 + 1)  # Generate more than enough primes for 21201 dimensions
assert len(_PRIMES) == 10000

def main():
    """Sample for Halton / Sobol sequences."""
    n = 1000
    d = 2
    ############### HALTON ###############
    # print("Initializing Halton sequence with d =", d, "and n =", n)
    # halton = Halton(d=d, scramble=False)
    # samples = halton.sample(n=n)
    # print(samples)
    # print("Sample shape:", samples.shape)


    ############### SOBOL ###############
    print("Initializing Sobol sequence with d =", d, "and n =", n)
    sobol = Sobol(d=d, scramble=True)
    # set to factorial of 2 for Sobol sequence properties.
    samples = sobol.sample(n=n)
    print(samples)
    print("Sample shape:", samples.shape)

if __name__ == "__main__":
    main()