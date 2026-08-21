"""
Module for Quasi-Monte Carlo (QMC) sampling methods, including Halton and Sobol sequences.

- QMC
- Halton(QMC)
- Sobol(QMC)

This follows design in Scipy. Also includes modules for scrambling

- Scramble(ABC)

This follows the implementation in QuasiMonteCarlo.jl
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
# check that we have exactly 10,000 primes
assert len(_PRIMES) == 10000

class QMC:
    """
    Interface for QMC classs.

    """
    def __init__(self, d: int) -> None:
        self._initialize(d=d)
    
    def _initialize(self, d: int) -> None:
        self.d = d
    
    @partial(jax.jit, static_argnums=(0, 1))
    def sample(self, n: int) -> jnp.ndarray:
        pass

class Halton(QMC):
    def __init__(self, d: int, scramble: bool = False):
        super().__init__(d=d)
        self.scramble = scramble
        self._primes = _PRIMES[:d]