from __future__ import annotations

import os
import time
from typing import Any, NamedTuple

import jax

# Global file-based progress log. Set RBSQMC_PROGRESS_LOG to a path to persist
# progress lines to a file (works even when the colab CLI does not stream the
# nested training subprocess's stdout). Falls back to flushed stdout.
_PROGRESS_LOG = os.environ.get("RBSQMC_PROGRESS_LOG", "")


def progress(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if _PROGRESS_LOG:
        try:
            with open(_PROGRESS_LOG, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


class Matches(NamedTuple):
    home_id: jax.Array
    away_id: jax.Array
    home_score: jax.Array
    away_score: jax.Array


class FootballResults(NamedTuple):
    date: jax.Array
    timestamp: jax.Array
    timestamp_prev: jax.Array
    matches: Matches
    match_mask: jax.Array


class RBPFFootballResults(NamedTuple):
    date: jax.Array
    timestamp: jax.Array
    timestamp_prev: jax.Array
    matches: Matches
    match_mask: jax.Array
    gamma: jax.Array
    gamma_pred: jax.Array
    gamma_observed: jax.Array
    kalman_gain: jax.Array


class EMParams(NamedTuple):
    mean_0: jax.Array
    gamma_0: jax.Array
    B: jax.Array
    kappa: jax.Array
    alpha: jax.Array
    beta: jax.Array


class RawEMParams(NamedTuple):
    gamma_0_chol: jax.Array
    B_ratio_raw: jax.Array
    kappa_raw: jax.Array
    alpha: jax.Array
    beta: jax.Array


class ParticleMeans(NamedTuple):
    x: jax.Array


class FilterStates(NamedTuple):
    particles: ParticleMeans
    log_weights: jax.Array
    ancestor_indices: jax.Array
    log_normalizing_constant: jax.Array


class RBSmootherParticle(NamedTuple):
    x: jax.Array
    gamma_filtered: jax.Array
    gamma_pred_next: jax.Array
    phi_next: jax.Array


class SmoothedStates(NamedTuple):
    particles: ParticleMeans
    component_indices: jax.Array
    backward_probabilities: jax.Array

    @property
    def ancestor_indices(self):
        return self.component_indices


def tree_to_python(value: Any) -> Any:
    """Convert a nested JAX/NumPy value to strict JSON-compatible values."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): tree_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
        return [tree_to_python(v) for v in value]
    if hasattr(value, "_asdict"):
        return {k: tree_to_python(v) for k, v in value._asdict().items()}
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        a = np.asarray(value)
        return a.item() if a.ndim == 0 else a.tolist()
    return value
