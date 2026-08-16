from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from .utils import EMParams


def params_to_dict(params):
    return {name: (float(value) if jnp.asarray(value).ndim == 0 else jnp.asarray(value).tolist())
            for name, value in params._asdict().items()}


def save_params(params: EMParams, path):
    Path(path).write_text(json.dumps(params_to_dict(params), indent=2))


def load_params(path):
    values = json.loads(Path(path).read_text())
    return EMParams(*(jnp.asarray(values[name]) for name in EMParams._fields))
