"""
Comprehensive QMC Comparison:
- JAX Halton (custom)
- JAX Sobol (custom)
- SciPy Halton
- SciPy Sobol
- PyTorch SobolEngine
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

# Import JAX implementations
import jax
import jax.numpy as jnp
from scripts.qmc.qmc_archive.halton import halton_sample
from scripts.qmc.qmc_archive.sobol import sobol_sample

# Detect backend and platform
BACKEND = jax.default_backend()
DEVICES = jax.devices()
IS_COLAB = os.path.exists("/content") or os.environ.get("COLAB_RELEASE_TAG") is not None
if IS_COLAB:
    PLATFORM = "Colab"
else:
    PLATFORM = "Local"
BACKEND_LABEL = f"{PLATFORM} ({BACKEND.upper()})"

print(f"JAX version: {jax.__version__}")
print(f"Backend: {BACKEND}")
print(f"Devices: {DEVICES}")
print(f"Platform: {PLATFORM}")
print()

# Create output directory
OUTPUT_DIR = "/Users/ryant/Github/ryantjx/rbsqmc/src/scripts/qmc/outputs/images"
if IS_COLAB:
    OUTPUT_DIR = "/content/qmc_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to import PyTorch
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: uv pip install torch")


def time_implementation(func, *args, num_runs=10, **kwargs):
    """Time a function over multiple runs."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        # Ensure JAX computation is complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)


def plot_comparison(samples_dict, title_suffix=""):
    """Plot comparison of all implementations."""
    n_impls = len(samples_dict)
    
    # Create figure with proper size
    fig = plt.figure(figsize=(5*n_impls, 10))
    gs = fig.add_gridspec(2, n_impls, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f"QMC Sequence Comparison {title_suffix}\n[{BACKEND_LABEL}]", fontsize=14, y=0.98)
    
    # Color map for different implementations
    colors = {
        'JAX Halton': '#1f77b4',
        'JAX Sobol': '#2ca02c', 
        'SciPy Halton': '#ff7f0e',
        'SciPy Sobol': '#d62728',
        'PyTorch Sobol': '#9467bd'
    }
    
    for idx, (name, samples) in enumerate(samples_dict.items()):
        n = len(samples)
        color = colors.get(name, 'gray')
        
        # 2D scatter plot
        ax_scatter = fig.add_subplot(gs[0, idx])
        ax_scatter.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.6, c=color, edgecolors='none')
        ax_scatter.set_title(f"{name}\n(n={n})", fontsize=11, fontweight='bold')
        ax_scatter.set_xlabel("Dimension 1")
        ax_scatter.set_ylabel("Dimension 2")
        ax_scatter.set_xlim(0, 1)
        ax_scatter.set_ylim(0, 1)
        ax_scatter.set_aspect('equal')
        ax_scatter.grid(True, alpha=0.3, linestyle='--')
        
        # Histogram with better styling
        ax_hist = fig.add_subplot(gs[1, idx])
        ax_hist.hist(samples[:, 0], bins=30, alpha=0.7, color=color, density=True, edgecolor='white', linewidth=0.5)
        ax_hist.set_title(f"{name} - Dim 1 Distribution", fontsize=11)
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_xlim(0, 1)
        ax_hist.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    return fig


def run_comparison(n_values=[100, 1000, 5000], d=2, scramble=True):
    """Run comprehensive comparison."""
    
    print("=" * 80)
    print("QMC IMPLEMENTATION COMPARISON")
    print("=" * 80)
    print(f"Dimensions: {d}, Scrambling: {scramble}")
    print("=" * 80)
    
    results = {
        'n': [],
        'jax_halton': [],
        'jax_sobol': [],
        'scipy_halton': [],
        'scipy_sobol': [],
    }
    
    if PYTORCH_AVAILABLE:
        results['pytorch_sobol'] = []
    
    for n in n_values:
        print(f"\n--- Sample size: n={n} ---")
        samples_dict = {}
        
        # JAX Halton
        key = jax.random.PRNGKey(42)
        max_bits = min(int(np.ceil(np.log2(n + 1))) + 2, 32)
        jax_halton_mean, jax_halton_std = time_implementation(
            halton_sample, n, d, scramble=scramble, seed=key, max_bits=max_bits, num_runs=10
        )
        jax_halton_samples = np.array(halton_sample(n, d, scramble=scramble, seed=key, max_bits=max_bits))
        samples_dict['JAX Halton'] = jax_halton_samples
        results['jax_halton'].append(jax_halton_mean * 1000)
        print(f"JAX Halton:   {jax_halton_mean*1000:.3f} ± {jax_halton_std*1000:.3f} ms")
        
        # JAX Sobol (use integer seed, not JAX PRNGKey)
        sobol_seed = 42
        jax_sobol_mean, jax_sobol_std = time_implementation(
            sobol_sample, n, d, scramble=scramble, seed=sobol_seed, num_runs=10
        )
        jax_sobol_samples = np.array(sobol_sample(n, d, scramble=scramble, seed=sobol_seed))
        samples_dict['JAX Sobol'] = jax_sobol_samples
        results['jax_sobol'].append(jax_sobol_mean * 1000)
        print(f"JAX Sobol:    {jax_sobol_mean*1000:.3f} ± {jax_sobol_std*1000:.3f} ms")
        
        # Debug: Check JAX Sobol samples
        if n == max(n_values):
            print(f"  JAX Sobol samples shape: {jax_sobol_samples.shape}")
            print(f"  JAX Sobol X range: {jax_sobol_samples[:,0].min():.4f} - {jax_sobol_samples[:,0].max():.4f}")
            print(f"  JAX Sobol Y range: {jax_sobol_samples[:,1].min():.4f} - {jax_sobol_samples[:,1].max():.4f}")
            print(f"  JAX Sobol unique X count: {len(np.unique(jax_sobol_samples[:,0]))}")
            print(f"  JAX Sobol unique Y count: {len(np.unique(jax_sobol_samples[:,1]))}")
            print(f"  JAX Sobol first 10 points:")
            for i in range(10):
                print(f"    {i}: ({jax_sobol_samples[i,0]:.6f}, {jax_sobol_samples[i,1]:.6f})")
        
        # SciPy Halton
        def scipy_halton(n, d, scramble):
            sampler = qmc.Halton(d=d, scramble=scramble, seed=42)
            return sampler.random(n)
        
        scipy_halton_mean, scipy_halton_std = time_implementation(
            scipy_halton, n, d, scramble, num_runs=10
        )
        scipy_halton_samples = scipy_halton(n, d, scramble)
        samples_dict['SciPy Halton'] = scipy_halton_samples
        results['scipy_halton'].append(scipy_halton_mean * 1000)
        print(f"SciPy Halton: {scipy_halton_mean*1000:.3f} ± {scipy_halton_std*1000:.3f} ms")
        
        # SciPy Sobol
        def scipy_sobol(n, d, scramble):
            sampler = qmc.Sobol(d=d, scramble=scramble, seed=42)
            return sampler.random(n)
        
        scipy_sobol_mean, scipy_sobol_std = time_implementation(
            scipy_sobol, n, d, scramble, num_runs=10
        )
        scipy_sobol_samples = scipy_sobol(n, d, scramble)
        samples_dict['SciPy Sobol'] = scipy_sobol_samples
        results['scipy_sobol'].append(scipy_sobol_mean * 1000)
        print(f"SciPy Sobol:  {scipy_sobol_mean*1000:.3f} ± {scipy_sobol_std*1000:.3f} ms")
        
        # PyTorch Sobol (if available)
        if PYTORCH_AVAILABLE:
            def pytorch_sobol(n, d, scramble):
                soboleng = torch.quasirandom.SobolEngine(dimension=d, scramble=scramble, seed=42)
                return soboleng.draw(n).numpy()
            
            pytorch_sobol_mean, pytorch_sobol_std = time_implementation(
                pytorch_sobol, n, d, scramble, num_runs=10
            )
            pytorch_sobol_samples = pytorch_sobol(n, d, scramble)
            samples_dict['PyTorch Sobol'] = pytorch_sobol_samples
            results['pytorch_sobol'].append(pytorch_sobol_mean * 1000)
            print(f"PyTorch Sobol: {pytorch_sobol_mean*1000:.3f} ± {pytorch_sobol_std*1000:.3f} ms")
        
        # Plot for largest n
        if n == max(n_values):
            fig = plot_comparison(samples_dict, f"(n={n}, d={d}, scramble={scramble})")
            plot_path = os.path.join(OUTPUT_DIR, f'qmc_comparison_n{n}_scramble{scramble}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"\nPlot saved to: {plot_path}")
        
        results['n'].append(n)
    
    # Summary plot with different bar types (patterns)
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    
    x = np.arange(len(results['n']))
    width = 0.12
    
    # Define implementations with colors and hatch patterns
    # Same colors for same library, different patterns for different algorithms
    implementations = [
        ('JAX Halton', results['jax_halton'], '#1f77b4', ''),      # Solid - JAX Halton
        ('JAX Sobol', results['jax_sobol'], '#1f77b4', '///'),     # Diagonal stripes - JAX Sobol
        ('SciPy Halton', results['scipy_halton'], '#ff7f0e', ''),  # Solid - SciPy Halton
        ('SciPy Sobol', results['scipy_sobol'], '#ff7f0e', 'xxx'), # Cross hatch - SciPy Sobol
    ]
    
    if PYTORCH_AVAILABLE:
        implementations.append(('PyTorch Sobol', results['pytorch_sobol'], '#2ca02c', '...'))  # Dots - PyTorch
    
    for i, (name, times, color, hatch) in enumerate(implementations):
        bars = ax.bar(x + i*width, times, width, label=name, color=color, 
                      alpha=0.85, hatch=hatch, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'QMC Implementation Performance Comparison [{BACKEND_LABEL}]\n(Same colors = Same library, Patterns = Different algorithms)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * (len(implementations) - 1) / 2)
    ax.set_xticklabels(results['n'])
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, 'qmc_benchmark_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("\n" + "=" * 80)
    print(f"Summary plot saved to: {summary_path}")
    print("=" * 80)
    
    return results


def main():
    """Run the comparison."""
    print("=" * 80)
    print(f"QMC COMPARISON - {BACKEND_LABEL}")
    print("=" * 80)

    # Compare with scrambling
    print("\n" + "=" * 80)
    print("COMPARISON WITH SCRAMBLING ENABLED")
    print("=" * 80)
    results_scrambled = run_comparison(n_values=[100, 1000, 5000, 10000], d=2, scramble=True)
    
    # Compare without scrambling
    print("\n" + "=" * 80)
    print("COMPARISON WITHOUT SCRAMBLING")
    print("=" * 80)
    results_unscrambled = run_comparison(n_values=[100, 1000, 5000, 10000], d=2, scramble=False)


if __name__ == "__main__":
    main()
