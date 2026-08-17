"""Kronecker Gaussian operations which avoid dense 2M by 2M matrices."""

import jax
import jax.numpy as jnp


def symmetrize(a):
    return 0.5 * (a + a.T)


def kron_logdet(A, B):
    sign_a, ld_a = jnp.linalg.slogdet(A)
    sign_b, ld_b = jnp.linalg.slogdet(B)
    value = B.shape[0] * ld_a + A.shape[0] * ld_b
    return jnp.where((sign_a > 0) & (sign_b > 0), value, jnp.nan)


def kron_quad(A, B, residuals):
    """Quadratic forms for arrays shaped ``(..., M, K)`` or ``(..., M*K)``."""
    r = jnp.asarray(residuals)
    M, K = A.shape[0], B.shape[0]
    leading = r.shape[:-2] if r.shape[-2:] == (M, K) else r.shape[:-1]
    s = r.reshape((-1, M, K))
    rhs = s.transpose(1, 0, 2).reshape(M, -1)
    solved = jnp.linalg.solve(A, rhs).reshape(M, s.shape[0], K).transpose(1, 0, 2)
    cross = jnp.matmul(s.transpose(0, 2, 1), solved)
    b_inv = jnp.linalg.solve(B, jnp.eye(K, dtype=B.dtype))
    out = jnp.sum(cross * b_inv.T[None], axis=(-2, -1))
    return out.reshape(leading)


def psd_sqrt(A, *, negative_tolerance=1e-7):
    """Symmetric PSD square root; eager calls reject substantive negatives."""
    values, vectors = jnp.linalg.eigh(symmetrize(A))
    if not isinstance(values, jax.core.Tracer):
        import numpy as np
        scale = max(1.0, float(np.max(np.abs(np.asarray(values)))))
        if float(np.min(np.asarray(values))) < -negative_tolerance * scale:
            raise ValueError("matrix is not positive semidefinite")
    values = jnp.maximum(values, 0.0)
    return (vectors * jnp.sqrt(values)[None, :]) @ vectors.T


def sample_kron_psd(key, mean, gamma, B):
    lg = psd_sqrt(gamma)
    lb = psd_sqrt(B)
    z = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + lg @ z @ lb.T


def rts_kron_terms(gamma_filtered, gamma_pred_next, phi):
    """Team-axis RTS gain and conditional covariance."""
    J = phi * jnp.linalg.solve(gamma_pred_next, gamma_filtered.T).T
    conditional = symmetrize(
        gamma_filtered - J @ gamma_pred_next @ J.T
    )
    return J, conditional
