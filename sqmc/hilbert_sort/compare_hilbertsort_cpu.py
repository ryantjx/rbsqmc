"""
Compare JAX (CPU/GPU) vs Numba Hilbert sort implementations.

Generates heatmaps showing runtime performance across different
dimensions and dataset sizes.
"""

import os
import time
import sys
from functools import partial

# Output directory configuration
OUTPUT_DIR = "/Users/ryant/Github/ryantjx/rbsqmc/src/scripts/hilbert_sort/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import jax
import jax.numpy as jnp

# Import the JAX implementation
import hilbert_sort as jax_backend

# Import the Numba implementation
sys.path.insert(0, '/Users/ryant/Github/ryantjx/rbsqmc/src/scripts/hilbert_sort')
from hilbert_particles import hilbert_sort as nb_hilbert_sort

# Note: 64-bit support disabled for Metal compatibility on macOS
# Use 32-bit floats for JAX operations
jax.config.update("jax_enable_x64", False)

# Parameters
DIMENSIONS = np.array([1, 2, 3, 4, 5])
N_VALUES = np.array([10, 100, 1000, 10000, 100000])
N_RUNS = 5  # Number of runs for averaging


def create_custom_colormap():
    """Create a custom colormap: Green (fast) to Red (slow)."""
    colors = ['#00ff00', '#ffff00', '#ff0000']  # Green -> Yellow -> Red
    return LinearSegmentedColormap.from_list('green_red', colors)


def run_benchmark(backend: str = "cpu"):
    """
    Run benchmark for JAX vs Numba.
    
    Args:
        backend: "cpu" or "gpu" for JAX backend
        
    Returns:
        runtime_nb: Numba runtimes (shape: len(DIMENSIONS), len(N_VALUES))
        runtime_jax: JAX runtimes (shape: len(DIMENSIONS), len(N_VALUES))
    """
    print(f"\n{'='*60}")
    print(f"Running benchmark with JAX backend: {backend.upper()}")
    print(f"{'='*60}\n")
    
    # JIT compile the JAX hilbert_sort
    jax_hilbert_sort = jax.jit(jax_backend.hilbert_sort, backend=backend)
    
    runtime_nb = np.empty((len(DIMENSIONS), len(N_VALUES)))
    runtime_jax = np.empty((len(DIMENSIONS), len(N_VALUES)))
    
    for i, n in enumerate(N_VALUES):
        print(f"\nN = {n:,} points")
        print("-" * 40)
        
        for j, d in enumerate(DIMENSIONS):
            print(f"  Dimension {d}...", end=" ")
            
            # Generate random data (float32 for JAX Metal compatibility)
            x = np.random.randn(n, d).astype(np.float32)
            
            # --- Numba Benchmark ---
            # Warm-up run
            _ = nb_hilbert_sort(x)
            
            # Timed runs
            tic = time.time()
            for _ in range(N_RUNS):
                _ = nb_hilbert_sort(x)
            nb_time = (time.time() - tic) / N_RUNS
            runtime_nb[j, i] = nb_time
            
            # --- JAX Benchmark ---
            x_jax = jnp.array(x)
            
            # Warm-up run (compilation)
            res = jax_hilbert_sort(x_jax)
            res.block_until_ready()
            
            # Timed runs
            tic = time.time()
            for _ in range(N_RUNS):
                res = jax_hilbert_sort(x_jax)
                res.block_until_ready()
            jax_time = (time.time() - tic) / N_RUNS
            runtime_jax[j, i] = jax_time
            
            print(f"Numba: {nb_time:.4f}s, JAX: {jax_time:.4f}s")
    
    return runtime_nb, runtime_jax


def plot_comparison(runtime_nb, runtime_jax, backend: str):
    """
    Create comparison heatmaps.
    
    Args:
        runtime_nb: Numba runtimes
        runtime_jax: JAX runtimes
        backend: "CPU" or "GPU" (for title)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create meshgrid for plotting
    NN, DD = np.meshgrid(N_VALUES, DIMENSIONS)
    
    # Calculate % difference vs Numba (Base)
    pct_diff = (runtime_jax - runtime_nb) / runtime_nb * 100
    
    # Use RdYlGn_r colormap for all plots (green=fast/better, red=slow/worse, white=neutral)
    cmap = 'RdYlGn_r'
    
    # Plot 1: Numba CPU
    im0 = axes[0].pcolormesh(
        np.log10(NN), DD, runtime_nb,
        norm=LogNorm(vmin=runtime_nb.min(), vmax=runtime_nb.max()),
        cmap=cmap, shading='auto'
    )
    axes[0].set_title("Numba CPU", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("$\\log_{10}(N)$", fontsize=12)
    axes[0].set_ylabel("Dimension ($d_X$)", fontsize=12)
    axes[0].set_yticks(DIMENSIONS)
    
    # Plot 2: JAX CPU
    im1 = axes[1].pcolormesh(
        np.log10(NN), DD, runtime_jax,
        norm=LogNorm(vmin=runtime_jax.min(), vmax=runtime_jax.max()),
        cmap=cmap, shading='auto'
    )
    axes[1].set_title(f"JAX {backend}", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("$\\log_{10}(N)$", fontsize=12)
    axes[1].set_yticks(DIMENSIONS)
    
    # Plot 3: % Difference vs Numba CPU (Base)
    # Center at 0 (white), green = JAX faster (negative), red = JAX slower (positive)
    im2 = axes[2].pcolormesh(
        np.log10(NN), DD, pct_diff,
        cmap=cmap, vmin=-50, vmax=50, shading='auto'
    )
    axes[2].set_title("% Difference vs Numba CPU (Base)", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("$\\log_{10}(N)$", fontsize=12)
    axes[2].set_yticks(DIMENSIONS)
    
    # Add colorbars
    cbar_ax0 = fig.add_axes([0.08, 0.02, 0.22, 0.03])
    cbar0 = fig.colorbar(im0, cax=cbar_ax0, orientation='horizontal')
    cbar0.set_label('Runtime (s)', fontsize=9)
    
    cbar_ax1 = fig.add_axes([0.40, 0.02, 0.22, 0.03])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Runtime (s)', fontsize=9)
    
    cbar_ax2 = fig.add_axes([0.72, 0.02, 0.22, 0.03])
    cbar2 = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('% Difference (0 = same)', fontsize=9)
    
    # Add text annotations for min/max on runtime plots
    for ax, data, name in [(axes[0], runtime_nb, "Numba"), 
                           (axes[1], runtime_jax, f"JAX {backend}")]:
        min_idx = np.unravel_index(np.argmin(data), data.shape)
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        
        # Mark fastest (green circle)
        ax.plot(np.log10(N_VALUES[min_idx[1]]), DIMENSIONS[min_idx[0]], 
                'go', markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                label=f'Fastest: {data[min_idx]:.4f}s')
        
        # Mark slowest (red circle)
        ax.plot(np.log10(N_VALUES[max_idx[1]]), DIMENSIONS[max_idx[0]], 
                'ro', markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                label=f'Slowest: {data[max_idx]:.4f}s')
        
        ax.legend(loc='upper left', fontsize=7)
    
    fig.suptitle(
        f"Hilbert Sort Performance: Numba vs JAX ({backend})",
        fontsize=16, fontweight='bold', y=1.02
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Make room for colorbars at bottom
    output_path = os.path.join(OUTPUT_DIR, f'hilbert_comparison_{backend.lower()}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved plot to {output_path}")
    plt.close()


def print_summary(runtime_nb, runtime_jax, backend: str):
    """Print a text summary of results."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: Numba vs JAX ({backend})")
    print(f"{'='*60}")
    
    print(f"\n{'Dimension':<10} {'N':<10} {'Numba (s)':<12} {'JAX (s)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for i, n in enumerate(N_VALUES):
        for j, d in enumerate(DIMENSIONS):
            nb_t = runtime_nb[j, i]
            jax_t = runtime_jax[j, i]
            speedup = nb_t / jax_t
            print(f"{d:<10} {n:<10} {nb_t:<12.4f} {jax_t:<12.4f} {speedup:<10.2f}x")
    
    print(f"\n{'='*60}")
    print(f"Overall Statistics:")
    print(f"  Numba - Min: {runtime_nb.min():.4f}s, Max: {runtime_nb.max():.4f}s")
    print(f"  JAX   - Min: {runtime_jax.min():.4f}s, Max: {runtime_jax.max():.4f}s")
    print(f"  Average Speedup: {(runtime_nb / runtime_jax).mean():.2f}x")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    # Configuration: Set to "cpu" or "gpu"
    JAX_BACKEND = "cpu"  # Change to "gpu" if available
    
    print("="*60)
    print("Hilbert Sort Benchmark: JAX vs Numba")
    print("="*60)
    print(f"\nParameters:")
    print(f"  JAX Backend: {JAX_BACKEND.upper()}")
    print(f"  Dimensions: {DIMENSIONS.tolist()}")
    print(f"  N values: {N_VALUES.tolist()}")
    print(f"  Runs per config: {N_RUNS}")
    
    # Check available devices
    print(f"\nJAX Devices:")
    print(f"  CPUs: {jax.device_count('cpu')}")
    try:
        print(f"  GPUs: {jax.device_count('gpu')}")
    except:
        print(f"  GPUs: 0 (not available)")
    
    # Validate backend choice
    if JAX_BACKEND == "gpu":
        try:
            if jax.device_count('gpu') == 0:
                print("\n⚠ GPU requested but not available. Falling back to CPU.")
                JAX_BACKEND = "cpu"
        except:
            print("\n⚠ GPU requested but not available. Falling back to CPU.")
            JAX_BACKEND = "cpu"
    
    # Run benchmark
    runtime_nb, runtime_jax = run_benchmark(JAX_BACKEND)
    plot_comparison(runtime_nb, runtime_jax, JAX_BACKEND.upper())
    print_summary(runtime_nb, runtime_jax, JAX_BACKEND.upper())
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
