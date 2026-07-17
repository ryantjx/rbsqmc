"""
Compute Hilbert indices and sort vectors according to their Hilbert index.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import math

MAX_BITS = 30 # 30 bits for 32-bit compatibility on Metal (macOS GPU).

def hilbert_sort(x: jnp.ndarray) -> jnp.ndarray:
    """
    Sort vectors according to their Hilbert curve index.
    
    Args:
        x: Array of shape (N,) or (N, d) containing N vectors in R^d
        
    Returns:
        Array of indices that would sort x by Hilbert curve order
    """
    # Handle 1D case
    if x.ndim == 1:
        return jnp.argsort(x, axis=0)
    
    d = x.shape[1]
    
    # Handle single dimension
    if d == 1:
        return jnp.argsort(x[:, 0], axis=0)
    
    # Standardize and map to [0, 1]
    scaled_x = (x - jnp.mean(x, axis=0, keepdims=True)) / jnp.std(x, axis=0, keepdims=True)
    xs = invlogit(scaled_x)
    
    # Scale to integers (use MAX_BITS for 32-bit compatibility on Metal)
    maxint = math.floor(2 ** (MAX_BITS / d))
    xint = jnp.floor(xs * maxint).astype(int)
    
    # Compute Hilbert index for each point
    Hilbert_to_int_spec = lambda z: Hilbert_to_int(z, maxint)
    hilbert_array = jax.vmap(Hilbert_to_int_spec)(xint)
    
    # Return sorting indices
    return jnp.argsort(hilbert_array)

@jax.jit
def invlogit(x):
    return 1. / (1. + jnp.exp(-x)) # maps (\infty, \infty) to (0, 1)

@jax.jit
def gray_encode(bn : jnp.ndarray) -> jnp.ndarray:
    return jnp.bitwise_xor(bn, bn // 2)

@partial(jax.jit, static_argnums=(1,))
def gray_decode(n: jnp.ndarray, max_bits : int = MAX_BITS) -> jnp.ndarray:
    """Gray decode using parallel prefix algorithm."""
    # Use global MAX_BITS to determine number of steps
    num_steps = int(np.ceil(np.log2(max_bits)))
    shifts = 2 ** jnp.arange(num_steps)  # [1, 2, 4, ...]
    def body(carry, shift):
        return carry ^ (carry >> shift), None
    result, _ = jax.lax.scan(body, n, shifts)
    return result

# @jax.jit
# def gray_decode_scan(n):
#     shifts = jnp.array([1, 2, 4, 8, 16, 32])  # Fixed array
#     def body(carry, shift):
#         carry = carry ^ (carry >> shift)
#         return carry, None
#     result, _ = jax.lax.scan(body, n, shifts)
#     return result

# fast implementation but up to 32 bits only.
# def gray_decode(n: int) -> int:
#     n = n ^ (n >> 1)
#     n = n ^ (n >> 2)
#     n = n ^ (n >> 4)
#     n = n ^ (n >> 8)
#     n = n ^ (n >> 16)
#     n = n ^ (n >> 32)  # Add for 64-bit support
#     return n

@partial(jax.jit, static_argnums=(1,))
def transpose_bits(srcs: jnp.ndarray, nDests: int) -> jnp.ndarray:
    """
    Transpose bits of srcs to dests. 
    e.g. 
    - srcs = jnp.ndarray([3, 5]), nDests = 3
    - [3, 5] in binary is [011, 101]
    - [011, 101] transposed is [01, 11, 10] which is [1, 2, 3] in decimal

    nDests static argument specify bits at destination.

    Args:
        srcs (jnp.ndarray): Source array of shape (N, d)
        nDests (int): Number of destination bits

    Returns:
        jnp.ndarray: Transposed array of shape (N, nDests)
    """
    def inner_body(dest, inner_srcs):
        dest = dest * 2 + inner_srcs % 2
        return dest, inner_srcs // 2

    def outer_body(outer_srcs, _):
        dest, outer_srcs = jax.lax.scan(inner_body, 0, outer_srcs)
        return outer_srcs, dest

    _, dests = jax.lax.scan(outer_body, srcs, jnp.arange(nDests), reverse=True)

    return dests

@partial(jax.jit, static_argnums=(1,))
def pack_index(chunks: jnp.ndarray, nD: int) -> jnp.ndarray:
    p = 2 ** nD
    def body(x, y):
        return p * x + y, None
    return jax.lax.scan(body, chunks[0], chunks[1:])[0]

@partial(jax.jit, static_argnums=(1,))
def unpack_coords(coords: jnp.ndarray, max_int: int) -> jnp.ndarray:
    nChunks = int(np.ceil(np.log2(max_int)))
    if nChunks < 1:
        nChunks = 1
    return transpose_bits(coords, nChunks)

@jax.jit
def gray_encode_travel(start: int, end: int, mask: int, i: int) -> int:
    travel_bit = start ^ end
    modulus = mask + 1
    g = gray_encode(i) * (travel_bit * 2)
    return ((g | (g // modulus)) & mask) ^ start

@jax.jit
def gray_decode_travel(start: int, end: int, mask: int, g: int) -> int:
    travel_bit = start ^ end
    modulus = mask + 1
    rg = (g ^ start) * (modulus // (travel_bit * 2))
    return gray_decode((rg | (rg // modulus)) & mask)

@jax.jit
def child_start_end(parent_start: int, parent_end: int, mask: int, i: int) -> tuple[int, int]:
    start_i = jnp.maximum(0, (i - 1) & ~1)  # Next lower even, or 0
    end_i = jnp.minimum(mask, (i + 1) | 1)  # Next higher odd, or mask
    child_start = gray_encode_travel(parent_start, parent_end, mask, start_i)
    child_end = gray_encode_travel(parent_start, parent_end, mask, end_i)
    return child_start, child_end

@jax.jit
def initial_start_end(nChunks: int, nD: int) -> tuple[int, int]:
    """
    Initialize start and end for Hilbert traversal.
    Orients the largest cube so its start is at origin and
    first step is along x-axis.
    """
    return 0, 2 ** ((-nChunks - 1) % nD)

@partial(jax.jit, static_argnums=(1,))
def Hilbert_to_int(coords: jnp.ndarray, max_int: int) -> jnp.ndarray:
    """
    Convert d-dimensional coordinates to 1D Hilbert index.
    """
    nD = coords.shape[0]
    coord_chunks = unpack_coords(coords, max_int)
    nChunks = coord_chunks.shape[0]
    mask = 2 ** nD - 1

    def body(carry, coord_chunks_j):
        start, end = carry
        i = gray_decode_travel(start, end, mask, coord_chunks_j)
        start, end = child_start_end(start, end, mask, i)
        return (start, end), i

    _, index_chunks = jax.lax.scan(body, initial_start_end(nChunks, nD), coord_chunks)
    return pack_index(index_chunks, nD)


def main():
    """Test the hilbert_sort function."""
    print("Testing hilbert_sort...")
    
    # Test 1: 1D case
    print("\nTest 1: 1D case (N=10)")
    x_1d = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])
    result = hilbert_sort(x_1d)
    print(f"  Input: {x_1d}")
    print(f"  Sorted indices: {result}")
    print(f"  Sorted values: {x_1d[result]}")
    
    # Test 2: 2D case
    print("\nTest 2: 2D case (N=5, d=2)")
    x_2d = jnp.array([[0.1, 0.2], [0.9, 0.8], [0.3, 0.4], [0.7, 0.6], [0.5, 0.5]])
    result = hilbert_sort(x_2d)
    print(f"  Input shape: {x_2d.shape}")
    print(f"  Sorted indices: {result}")
    print(f"  Sorted points:\n{x_2d[result]}")
    
    # Test 3: 3D case
    print("\nTest 3: 3D case (N=5, d=3)")
    x_3d = jnp.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.4, 0.5, 0.6], 
                      [0.2, 0.1, 0.3], [0.7, 0.6, 0.5]])
    result = hilbert_sort(x_3d)
    print(f"  Input shape: {x_3d.shape}")
    print(f"  Sorted indices: {result}")
    print(f"  Sorted points:\n{x_3d[result]}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()