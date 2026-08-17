from pathlib import Path
import re

import jax
import numpy as np


FILES = (
    "bivariate_poisson.py",
    "data.py",
    "graphic.py",
    "helpers.py",
    "model.py",
    "model_trained.py",
    "utils.py",
)


def test_wholesale_source_copy_has_only_namespace_changes():
    root = Path(__file__).resolve().parents[3]
    for name in FILES:
        original = (root / "rbpf/src" / name).read_text()
        copied = (root / "rbpf_v3/src" / name).read_text()
        assert copied.replace("rbpf_v3.src", "rbpf.src") == original
    source = "\n".join((root / "rbpf_v3/src" / name).read_text() for name in FILES)
    assert re.search(r"(?:from|import) rbpf\.src", source) is None


def test_fixed_key_filter_copy_equivalence():
    from rbpf.src.data import get_results as old_get_results
    from rbpf.src.helpers import default_init_params as old_default
    from rbpf.src.model import run_filter as old_run_filter
    from rbpf_v3.src.data import get_results as new_get_results
    from rbpf_v3.src.helpers import default_init_params as new_default
    from rbpf_v3.src.model import run_filter as new_run_filter

    kwargs = dict(start_date="2024-01-01", end_date="2024-01-15", max_goals=8)
    _, old_data, old_names = old_get_results(**kwargs)
    _, new_data, new_names = new_get_results(**kwargs)
    assert old_names == new_names
    old = old_run_filter(
        jax.random.key(3), old_data, old_default(len(old_names), team_id_to_name=old_names), 4, 8
    )
    new = new_run_filter(
        jax.random.key(3), new_data, new_default(len(new_names), team_id_to_name=new_names), 4, 8
    )
    old_paths = [str(path) for path, _ in jax.tree_util.tree_flatten_with_path(old)[0]]
    new_paths = [str(path) for path, _ in jax.tree_util.tree_flatten_with_path(new)[0]]
    assert old_paths == new_paths
    for left, right in zip(jax.tree.leaves(old), jax.tree.leaves(new)):
        if jax.dtypes.issubdtype(left.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(jax.random.key_data(left), jax.random.key_data(right))
        else:
            assert left.shape == right.shape and left.dtype == right.dtype
            np.testing.assert_allclose(left, right, rtol=0, atol=0)
