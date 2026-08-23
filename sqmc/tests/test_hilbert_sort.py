import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqmc.hilbert_sort.hilbert_sort import (
    Hilbert_to_int,
    gray_decode,
    gray_decode_travel,
    gray_encode,
    gray_encode_travel,
    hilbert_sort,
)


def test_x64_is_enabled():
    assert jax.config.jax_enable_x64


def test_gray_code_round_trip():
    values = jnp.asarray(
        [0, 1, 2, 3, 17, 2**31, 2**62 - 1],
        dtype=jnp.uint64,
    )
    np.testing.assert_array_equal(
        np.asarray(gray_decode(gray_encode(values))),
        np.asarray(values),
    )


@pytest.mark.parametrize("dimension", [2, 7, 31, 62])
def test_traveling_gray_code_round_trip_at_wide_dimensions(dimension):
    mask = jnp.uint64((1 << dimension) - 1)
    start = jnp.uint64((1 << (dimension - 1)) - 1)
    end = start ^ jnp.uint64(1 << (dimension - 1))
    steps = jnp.asarray(
        [0, 1, 2, 3, (1 << dimension) - 1],
        dtype=jnp.uint64,
    )

    encoded = jax.vmap(
        lambda step: gray_encode_travel(start, end, mask, step)
    )(steps)
    decoded = jax.vmap(
        lambda value: gray_decode_travel(start, end, mask, value)
    )(encoded)

    np.testing.assert_array_equal(np.asarray(decoded), np.asarray(steps))


def test_known_two_dimensional_hilbert_indices():
    coordinates = jnp.asarray(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
        ],
        dtype=jnp.uint64,
    )
    expected = np.asarray(
        [0, 3, 4, 5, 1, 2, 7, 6, 14, 13, 8, 9, 15, 12, 11, 10],
        dtype=np.uint64,
    )
    actual = jax.vmap(lambda point: Hilbert_to_int(point, 4))(coordinates)

    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_one_dimensional_sort_matches_argsort():
    values = jnp.asarray([3.0, 1.0, 4.0, 1.0, 5.0])
    expected = jnp.argsort(values, stable=True)

    np.testing.assert_array_equal(
        np.asarray(hilbert_sort(values)),
        np.asarray(expected),
    )
    np.testing.assert_array_equal(
        np.asarray(hilbert_sort(values[:, None])),
        np.asarray(expected),
    )


def test_constant_columns_are_handled_without_nan_dependent_ordering():
    points = jnp.ones((16, 3), dtype=jnp.float64)
    np.testing.assert_array_equal(
        np.asarray(hilbert_sort(points)),
        np.arange(16),
    )


@pytest.mark.parametrize("dimension", [2, 4, 7, 31, 62])
def test_sort_returns_a_permutation_for_supported_dimensions(dimension):
    points = jax.random.normal(
        jax.random.PRNGKey(dimension),
        shape=(128, dimension),
        dtype=jnp.float64,
    )
    order = np.asarray(hilbert_sort(points))

    np.testing.assert_array_equal(
        np.sort(order),
        np.arange(points.shape[0]),
    )


def test_empty_input_returns_empty_permutation():
    order = hilbert_sort(jnp.empty((0, 2), dtype=jnp.float64))
    assert order.shape == (0,)


def test_more_than_62_dimensions_is_rejected():
    with pytest.raises(ValueError, match="At most 62 dimensions"):
        hilbert_sort(jnp.ones((4, 63), dtype=jnp.float64))


@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 0)])
def test_invalid_shape_is_rejected(shape):
    with pytest.raises(ValueError):
        hilbert_sort(jnp.ones(shape, dtype=jnp.float64))
