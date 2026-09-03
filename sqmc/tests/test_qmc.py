"""Tests for the QMC point-set generators in ``sqmc.qmc.qmc``.

Test conventions follow ``state-space-models/cuthbert``: ``chex.TestCase``
classes, absl ``parameterized`` markers, ``chex.assert_trees_all_close`` and
``chex.assert_shape``, and a module-autouse x64 fixture that restores the
flag on teardown.

``@chex.variants(with_jit, without_jit)`` is applied only to pure,
array-valued transforms (e.g. ``_apply_lms``). The engine ``sample()``
methods are stateful (they mutate ``_num_generated``) and are therefore run
as plain ``chex.TestCase`` methods.
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from scipy.stats import qmc

import chex

from sqmc.qmc.qmc import Halton, Sobol, _MAXBITS, _apply_lms


@pytest.fixture(scope="module", autouse=True)
def config():
    """Enable double precision for the module and restore it on teardown."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


class HaltonTest(chex.TestCase):
    def test_known_first_points(self):
        actual = np.asarray(
            Halton(
                d=2,
                scramble=False,
                start_index=0,
                dtype=jnp.float64,
            ).sample(8)
        )
        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0 / 3.0],
                [0.25, 2.0 / 3.0],
                [0.75, 1.0 / 9.0],
                [0.125, 4.0 / 9.0],
                [0.625, 7.0 / 9.0],
                [0.375, 2.0 / 9.0],
                [0.875, 5.0 / 9.0],
            ],
            dtype=np.float64,
        )

        self.assert_trees_all_close(actual, expected, rtol=0.0, atol=1e-15)

    @parameterized.product(d=[1, 2, 5], n=[1, 100])
    def test_unscrambled_matches_scipy(self, d, n):
        actual = np.asarray(
            Halton(
                d=d,
                scramble=False,
                start_index=0,
                dtype=jnp.float64,
            ).sample(n)
        )
        expected = qmc.Halton(d=d, scramble=False).random(n)
        self.assert_trees_all_close(actual, expected, rtol=0.0, atol=1e-15)

    @parameterized.product(scramble=[False, True])
    def test_chunked_sampling_matches_single_batch(self, scramble):
        whole = np.asarray(
            Halton(
                d=5,
                scramble=scramble,
                key=jax.random.PRNGKey(42),
                start_index=0,
                dtype=jnp.float64,
            ).sample(100)
        )

        engine = Halton(
            d=5,
            scramble=scramble,
            key=jax.random.PRNGKey(42),
            start_index=0,
            dtype=jnp.float64,
        )
        chunked = np.concatenate(
            [
                np.asarray(engine.sample(17)),
                np.asarray(engine.sample(33)),
                np.asarray(engine.sample(50)),
            ],
            axis=0,
        )

        self.assert_trees_all_close(whole, chunked, rtol=0.0, atol=0.0)

    def test_scramble_key_controls_reproducibility(self):
        first = np.asarray(
            Halton(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(42),
                dtype=jnp.float64,
            ).sample(100)
        )
        repeated = np.asarray(
            Halton(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(42),
                dtype=jnp.float64,
            ).sample(100)
        )
        different = np.asarray(
            Halton(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(43),
                dtype=jnp.float64,
            ).sample(100)
        )

        self.assert_trees_all_close(first, repeated, rtol=0.0, atol=0.0)
        self.assertFalse(np.array_equal(first, different))

    def test_owen_permutations_are_valid(self):
        engine = Halton(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
            dtype=jnp.float64,
        )

        for base, num_digits, permutations in zip(
            engine._bases,
            engine._digits_per_dim,
            engine._permutations,
        ):
            permutations = np.asarray(permutations)
            self.assert_shape(permutations, (num_digits, base))

            expected_digits = np.arange(base)
            for permutation in permutations:
                self.assert_trees_all_close(
                    np.sort(permutation),
                    expected_digits,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_scrambled_kernel_matches_scipy_with_same_permutations(self):
        scipy_engine = qmc.Halton(
            d=5,
            scramble=True,
            rng=np.random.default_rng(42),
        )
        custom_engine = Halton(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
            start_index=0,
            dtype=jnp.float64,
        )

        # This private SciPy state is used only to isolate and validate the
        # custom radical-inverse kernel independently of RNG differences.
        custom_engine._permutations = tuple(
            jnp.asarray(permutation)
            for permutation in scipy_engine._permutations
        )

        actual = np.asarray(custom_engine.sample(100))
        expected = scipy_engine.random(100)
        self.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    @parameterized.parameters(0, -1, 10_001)
    def test_invalid_dimension_raises(self, d):
        with pytest.raises(ValueError):
            Halton(d=d)

    @parameterized.parameters(-1, -10)
    def test_negative_sample_size_raises(self, n):
        with pytest.raises(ValueError):
            Halton(d=2).sample(n)

    @parameterized.parameters(1.5, "10", None)
    def test_noninteger_sample_size_raises(self, n):
        with pytest.raises(TypeError):
            Halton(d=2).sample(n)


class SobolTest(chex.TestCase):
    def test_known_first_points(self):
        actual = np.asarray(
            Sobol(d=2, scramble=False, dtype=jnp.float64).sample(8)
        )
        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.375, 0.375],
                [0.875, 0.875],
                [0.625, 0.125],
                [0.125, 0.625],
            ],
            dtype=np.float64,
        )

        self.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    @parameterized.product(d=[1, 2, 5], m=[0, 3, 7])
    def test_unscrambled_matches_scipy(self, d, m):
        n = 2**m
        actual = np.asarray(
            Sobol(d=d, scramble=False, dtype=jnp.float64).sample(n)
        )
        expected = qmc.Sobol(d=d, scramble=False, bits=_MAXBITS).random_base2(m)
        self.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    @parameterized.product(scramble=[False, True])
    def test_chunked_sampling_matches_single_batch(self, scramble):
        key = jax.random.PRNGKey(42)
        whole = np.asarray(
            Sobol(
                d=5,
                scramble=scramble,
                key=key,
                dtype=jnp.float64,
            ).sample(64)
        )

        engine = Sobol(
            d=5,
            scramble=scramble,
            key=key,
            dtype=jnp.float64,
        )
        chunked = np.concatenate(
            [
                np.asarray(engine.sample(7)),
                np.asarray(engine.sample(19)),
                np.asarray(engine.sample(38)),
            ],
            axis=0,
        )

        self.assert_trees_all_close(whole, chunked, rtol=0.0, atol=0.0)

    def test_scramble_key_controls_reproducibility(self):
        first = np.asarray(
            Sobol(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(42),
                dtype=jnp.float64,
            ).sample(64)
        )
        repeated = np.asarray(
            Sobol(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(42),
                dtype=jnp.float64,
            ).sample(64)
        )
        different = np.asarray(
            Sobol(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(43),
                dtype=jnp.float64,
            ).sample(64)
        )

        self.assert_trees_all_close(first, repeated, rtol=0.0, atol=0.0)
        self.assertFalse(np.array_equal(first, different))

    def test_lms_row_masks_are_unit_lower_triangular(self):
        engine = Sobol(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
        )
        row_masks = np.asarray(engine._lms_matrices, dtype=np.uint32)

        rows = np.arange(_MAXBITS, dtype=np.uint32)
        bit_positions = np.uint32(_MAXBITS - 1) - rows
        diagonal_masks = np.left_shift(np.uint32(1), bit_positions)
        allowed_masks = np.left_shift(
            np.left_shift(np.uint32(1), rows + np.uint32(1)) - np.uint32(1),
            bit_positions,
        )

        self.assert_shape(row_masks, (engine.d, _MAXBITS))
        self.assert_trees_all_close(
            row_masks & diagonal_masks[None, :],
            np.broadcast_to(diagonal_masks, row_masks.shape),
            rtol=0.0,
            atol=0.0,
        )
        self.assert_trees_all_close(
            row_masks & ~allowed_masks[None, :],
            np.zeros_like(row_masks),
            rtol=0.0,
            atol=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_identity_lms_preserves_direction_integers(self):
        engine = Sobol(d=5, scramble=False)
        bit_positions = jnp.arange(
            _MAXBITS - 1,
            -1,
            -1,
            dtype=jnp.uint32,
        )
        identity_row_masks = jnp.broadcast_to(
            (jnp.uint32(1) << bit_positions)[None, :],
            (engine.d, _MAXBITS),
        )

        actual = self.variant(_apply_lms)(
            engine._direction_integers,
            identity_row_masks,
        )
        self.assert_trees_all_close(
            actual,
            engine._direction_integers,
            rtol=0.0,
            atol=0.0,
        )

    def test_first_scrambled_point_is_digital_shift(self):
        engine = Sobol(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
            dtype=jnp.float64,
        )

        actual = np.asarray(engine.sample(1)[0])
        expected = (
            np.asarray(engine._digital_shift, dtype=np.float64) * 2.0 ** -_MAXBITS
        )
        self.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    def test_reset_reproduces_points(self):
        engine = Sobol(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
            dtype=jnp.float64,
        )
        first = np.asarray(engine.sample(32))
        returned = engine.reset()
        repeated = np.asarray(engine.sample(32))

        self.assertIs(returned, engine)
        self.assert_trees_all_close(first, repeated, rtol=0.0, atol=0.0)

    @parameterized.parameters(0, -1, 21_202)
    def test_invalid_dimension_raises(self, d):
        with pytest.raises(ValueError):
            Sobol(d=d)

    @parameterized.parameters(0, -1, -10)
    def test_nonpositive_sample_size_raises(self, n):
        with pytest.raises(ValueError):
            Sobol(d=2).sample(n)

    @parameterized.parameters(1.5, "10", None)
    def test_noninteger_sample_size_raises(self, n):
        with pytest.raises(TypeError):
            Sobol(d=2).sample(n)
