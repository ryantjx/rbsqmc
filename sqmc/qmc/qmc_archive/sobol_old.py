"""
Sobol Sequence Generator

Sobol sequences are low-discrepancy sequences that use direction numbers
and Gray code for efficient generation.

References:
- Joe & Kuo direction numbers: https://web.maths.unsw.edu.au/~fkuo/sobol/
- Wikipedia: https://en.wikipedia.org/wiki/Sobol_sequence
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp

_MAX_DIMENSION = 21201  # Maximum supported dimensions (Joe & Kuo)
_MAX_BITS = 30  # Number of bits for direction numbers


def _load_direction_numbers():
    """
    Load direction numbers from Joe & Kuo sobol_data.npz file.
    
    The file contains:
    - sobol_a: polynomial coefficients a (21200,)
    - sobol_minit: initial direction numbers m (40, 21200)
    
    The degree s for each dimension is determined by the number of non-zero
    values in sobol_minit for that dimension.
    
    Returns direction numbers of shape (_MAX_DIMENSION, _MAX_BITS)
    """
    import os
    
    data_path = os.path.join(os.path.dirname(__file__), 'sobol_data.npz')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"sobol_data.npz not found at {data_path}")
    
    # Load Joe & Kuo data
    data = np.load(data_path)
    sobol_a = data['sobol_a']  # Shape: (21200,) - polynomial coefficients
    sobol_minit = data['sobol_minit']  # Shape: (40, 21200) - initial direction numbers
    
    # Initialize direction numbers array
    direction_numbers = np.zeros((_MAX_DIMENSION, _MAX_BITS), dtype=np.uint32)
    
    # For all dimensions (sobol_a has 21200 entries for dimensions 1-21200)
    max_dim = min(sobol_a.shape[0], _MAX_DIMENSION)
    
    for dim in range(max_dim):
        # Get polynomial coefficient a
        a = int(sobol_a[dim])
        
        # Determine degree from sobol_minit (last non-zero entry)
        m_init = sobol_minit[:, dim]
        non_zero_indices = np.where(m_init != 0)[0]
        if len(non_zero_indices) > 0:
            degree = int(non_zero_indices[-1]) + 1  # +1 because indices are 0-based
        else:
            degree = 1
        
        # Get initial direction numbers from sobol_minit
        # m_init[:degree] contains m[0], m[1], ..., m[degree-1]
        for j in range(min(degree, _MAX_BITS)):
            direction_numbers[dim, j] = int(m_init[j])
        
        # Compute remaining direction numbers using recurrence relation:
        # m[j] = 2*a[1]*m[j-1] XOR 2^2*a[2]*m[j-2] XOR ... XOR 2^s*m[j-s] XOR m[j-s]
        # where a[k] is the k-th bit of the polynomial coefficient
        for j in range(degree, _MAX_BITS):
            # Start with m[j-s] (the term that's always XORed)
            m_new = direction_numbers[dim, j - degree]
            
            # XOR with 2^k * a[k] * m[j-k] for k = 1 to s
            # a[k] is the k-th bit of 'a' (0-indexed: bit 0 is a[1])
            for k in range(1, degree + 1):
                if (a >> (k - 1)) & 1:  # Check if a[k] is 1
                    m_new ^= direction_numbers[dim, j - k] << k
            
            direction_numbers[dim, j] = m_new
    
    # Scale direction numbers: v[j] = m[j] * 2^(bits - j - 1) / 2^bits = m[j] / 2^(j+1)
    # We store v[j] * 2^bits = m[j] * 2^(bits - j - 1) as integers for XOR operations
    direction_numbers_scaled = np.zeros_like(direction_numbers)
    for dim in range(_MAX_DIMENSION):
        for j in range(_MAX_BITS):
            # v[j] = m[j] / 2^(j+1), scaled by 2^bits
            direction_numbers_scaled[dim, j] = direction_numbers[dim, j] << (_MAX_BITS - j - 1)
    
    return jnp.array(direction_numbers_scaled)


# Load direction numbers at module initialization
_DIRECTION_NUMBERS = _load_direction_numbers()


def _gray_code(n: jnp.ndarray) -> jnp.ndarray:
    """Compute Gray code: g = n ^ (n >> 1)"""
    return n ^ (n >> 1)


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 4))
def sobol_sample(
    n: int,
    d: int,
    scramble: bool = False,
    seed: jax.Array = jax.random.PRNGKey(0),
    bits: int = 30,
) -> jnp.ndarray:
    """
    Generate a Sobol sequence of n points in d dimensions.

    Parameters
    ----------
    n : int
        Number of points to generate.
    d : int
        Number of dimensions (max 21201).
    scramble : bool, optional
        Whether to apply scrambling (default: False).
    seed : jax.Array, optional
        Random seed for scrambling.
    bits : int, optional
        Number of bits for direction numbers (default: 30).

    Returns
    -------
    jnp.ndarray
        An array of shape (n, d) containing the Sobol sequence points.
    """
    if d > _MAX_DIMENSION:
        raise ValueError(f"Maximum dimension is {_MAX_DIMENSION}")
    
    # Get direction numbers for requested dimensions
    # direction_numbers contains v[j] * 2^bits values (pre-scaled integers)
    direction_nums = _DIRECTION_NUMBERS[:d, :bits]  # (d, bits)

    # Generate indices and compute Gray code
    indices = jnp.arange(n, dtype=jnp.uint32)
    gray_codes = _gray_code(indices)  # (n,)

    # Vectorized Sobol generation using proper XOR algorithm
    def generate_dimension(direction_num_dim):
        """Generate Sobol sequence for one dimension using XOR."""
        # direction_num_dim: (bits,) - contains v[j] * 2^bits values (integers)

        # For each point, XOR the direction numbers where Gray code bit is set
        # Start with 0, then XOR in each direction number where bit is set
        result_int = jnp.zeros(n, dtype=jnp.uint32)

        for bit in range(bits):
            # Check if bit is set in Gray code
            bit_mask = (gray_codes >> bit) & 1  # (n,) of 0 or 1
            # XOR the direction number where bit is set
            # XOR with 0 is identity, XOR with v[bit]*2^bits when bit is set
            result_int = result_int ^ (bit_mask.astype(jnp.uint32) * direction_num_dim[bit].astype(jnp.uint32))

        # Normalize to [0, 1) by dividing by 2^bits
        return result_int.astype(jnp.float32) / (2.0 ** bits)

    # Generate all dimensions using vmap
    samples = jax.vmap(generate_dimension, in_axes=0, out_axes=1)(direction_nums)
    
    if scramble:
        samples = _randomize_sobol(samples, seed)
    
    return samples


def _randomize_sobol(
    samples: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """
    Apply random linear scrambling to Sobol sequence.
    
    This is a simplified scrambling - full Owen scrambling is more complex.
    """
    n, d = samples.shape
    
    # Generate random shifts for each dimension
    shifts = jax.random.uniform(key, (d,))
    
    # Apply shift modulo 1
    return (samples + shifts[None, :]) % 1.0


def main():
    """Demo the Sobol sequence generator."""
    import matplotlib.pyplot as plt
    import os

    n = 1000
    d = 2

    # Generate samples
    samples = sobol_sample(n, d, scramble=False)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"Sobol Sequence (n={n}, d={d})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    # Save to specific path
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'images')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sobol_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main() 
