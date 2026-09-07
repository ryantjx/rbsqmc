"""Compare shared Halton/Sobol generation on CPU and GPU."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
from sqmc.comparison import common
from sqmc.qmc.qmc import QMCState


def make_runner(sequence, n, dimension, scramble, mode, device, seed):
    with jax.default_device(device):
        if mode == "sample":
            generator = common.engine(sequence, dimension, jax.random.key(seed), scramble)

            @jax.jit
            def run(index, key):
                points, _ = generator.sample(n, state=QMCState(index))
                return points
        else:
            @jax.jit
            def run(index, key):
                generator = common.engine(sequence, dimension, key, scramble)
                return generator.sample(n)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser)
    parser.add_argument("--sequences", nargs="+", choices=["sobol", "halton"], default=["sobol", "halton"])
    parser.add_argument("--modes", nargs="+", choices=["sample", "fresh"], default=["sample", "fresh"])
    parser.add_argument("--scramble", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    with common.run_directory("qmc", args) as output:
        common.validate_grid(args.dimensions, args.n_values, max_dimension=10000,
                             power_two="sobol" in args.sequences)
        selected = common.devices(args.platforms)
        common.provenance(output, selected)
        rows = []
        case = 0
        for sequence in dict.fromkeys(args.sequences):
            for dimension in args.dimensions:
                for n in args.n_values:
                    for mode in dict.fromkeys(args.modes):
                        order = list(selected)
                        if case % 2:
                            order.reverse()
                        case += 1
                        for position, backend in enumerate(order):
                            device = selected[backend]
                            start = time.perf_counter()
                            runner = make_runner(sequence, n, dimension, args.scramble, mode, device, args.seed)
                            setup = time.perf_counter() - start
                            with jax.default_device(device):
                                keys = jax.random.split(jax.random.key(args.seed), args.repeats)
                                inputs = common.prepare_inputs([(jnp.uint32(0), key) for key in keys], device)
                                timing, points = common.timed(runner, inputs, args.warmups, args.repeats)
                            common.assert_device(points, device)
                            host = np.asarray(points)
                            if host.shape != (n, dimension) or not np.isfinite(host).all() or not ((host >= 0) & (host < 1)).all():
                                raise RuntimeError("QMC output failed shape/range validation.")
                            row = dict(sequence=sequence, dimension=dimension, n=n, mode=mode,
                                       backend=backend, order=position, setup_seconds=setup, **timing)
                            rows.append(row)
                            common.write_json(output / "results.json", rows)
                            common.write_csv(output / "results.csv", rows)
                            print(f"{sequence} d={dimension} N={n} {mode} {backend}: {timing['median_seconds']:.6f}s")
        comparisons = []
        for row in rows:
            if row["backend"] != "cpu":
                continue
            gpu = next((other for other in rows if other["backend"] == "gpu" and all(other[k] == row[k] for k in ("sequence", "dimension", "n", "mode"))), None)
            if gpu:
                comparisons.append({k: row[k] for k in ("sequence", "dimension", "n", "mode")} |
                                   {"cpu_seconds": row["median_seconds"], "gpu_seconds": gpu["median_seconds"],
                                    "cpu_over_gpu": row["median_seconds"] / gpu["median_seconds"]})
        common.write_json(output / "cpu_gpu_comparison.json", comparisons)
        common.plot_lines(rows, output, "runtime.png", group_fields=["sequence", "mode", "dimension", "backend"],
                          x="n", y="median_seconds", ylabel="Median generation time (seconds)")


if __name__ == "__main__":
    main()
