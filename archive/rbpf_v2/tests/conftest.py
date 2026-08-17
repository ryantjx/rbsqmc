import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_v2_matplotlib")

import jax
import pytest

from rbpf_v2.src.data import synthetic_results
from rbpf_v2.src.helpers import default_init_params


@pytest.fixture
def small_model():
    frame, data, names = synthetic_results()
    return frame, data, default_init_params(len(names))


@pytest.fixture
def key():
    return jax.random.key(7)
