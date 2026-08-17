from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.nn import softplus

from .utils import EMParams, RawEMParams


def _inverse_softplus(x):
    return jnp.log(jnp.expm1(x))


def default_init_params(num_teams: int, mean_0=None) -> EMParams:
    mean = jnp.zeros((num_teams, 2)) if mean_0 is None else jnp.asarray(mean_0)
    return EMParams(mean, jnp.eye(num_teams), jnp.eye(2), jnp.asarray(0.03),
                    jnp.asarray(0.15), jnp.asarray(-1.5))


def encode_EM_params(params: EMParams) -> RawEMParams:
    chol = np.linalg.cholesky(np.asarray(params.gamma_0))
    raw = np.tril(chol)
    diag = np.diag_indices_from(raw)
    raw[diag] = np.asarray(_inverse_softplus(jnp.asarray(raw[diag]) - 1e-4))
    ratio = 0.5 * jnp.log(params.B[0, 0] / params.B[1, 1])
    ratio_raw = jnp.arctanh(jnp.clip(ratio / 5.0, -0.999999, 0.999999))
    return RawEMParams(jnp.asarray(raw), ratio_raw,
                       _inverse_softplus(params.kappa - 1e-6),
                       params.alpha, params.beta)


def decode_EM_params(raw: RawEMParams, mean_0) -> EMParams:
    lower = jnp.tril(raw.gamma_0_chol, -1)
    diag = softplus(jnp.diag(raw.gamma_0_chol)) + 1e-4
    L = lower + jnp.diag(diag)
    gamma = L @ L.T
    r = 5.0 * jnp.tanh(raw.B_ratio_raw)
    B = jnp.diag(jnp.array([jnp.exp(r), jnp.exp(-r)]))
    return EMParams(jnp.asarray(mean_0), gamma, B,
                    softplus(raw.kappa_raw) + 1e-6, raw.alpha, raw.beta)


def log_inverse_wishart_kernel(gamma_0, nu, S_gamma):
    sign, logdet = jnp.linalg.slogdet(gamma_0)
    solved = jnp.linalg.solve(gamma_0, S_gamma)
    value = -0.5 * (nu + gamma_0.shape[0] + 1) * logdet - 0.5 * jnp.trace(solved)
    return jnp.where(sign > 0, value, -jnp.inf)


encode_em_params = encode_EM_params
decode_em_params = decode_EM_params
