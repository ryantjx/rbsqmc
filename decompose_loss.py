import jax
import jax.numpy as jnp
import numpy as np
from rbpf.src.data import get_results, WORLDCUP_2026_TEAMS
from rbpf.src.helpers import default_init_params, generate_augmented_data
from rbpf.src.model import compute_gamma_trajectory
from rbpf.src.smoothing_1traj_old import E_step, loss_fn, _constrain, MAX_GOALS
from rbpf.src.bivariate_poisson import loglik

def decompose_loss(params, smoothed, mi):
    """Return (obs_loss, transition_loss) separately."""
    n_obs = smoothed.shape[0]
    num_teams = smoothed.shape[1]
    K = smoothed.shape[2]
    dim = num_teams * K
    obs_idx = jnp.arange(n_obs)
    tr_idx = jnp.arange(1, n_obs)

    def obs_step(i):
        return loglik(
            y=jnp.array([mi.home_score[i], mi.away_score[i]]),
            x_i=smoothed[i, mi.home_team_id[i]],
            x_j=smoothed[i, mi.away_team_id[i]],
            alpha=params.alpha, beta=params.beta,
            max_goals=MAX_GOALS, scale=1.0,
        )
    obs_loss = -jnp.sum(jax.vmap(obs_step)(obs_idx))

    dts = mi.timestamp[tr_idx] - mi.timestamp_prev[tr_idx]
    phis = jnp.exp(-params.kappa * dts)

    def tr_step(i, phi):
        pred_mean = params.mean_0 + phi * (smoothed[i-1] - params.mean_0)
        diff = smoothed[i] - pred_mean
        scale = 1.0 - phi**2
        det = scale <= 1e-8
        Q_gamma = jnp.maximum(scale, 1e-8) * params.gamma_0
        Q_gamma = 0.5*(Q_gamma+Q_gamma.T)
        Q = jnp.kron(Q_gamma, params.B)
        diff_flat = diff.reshape(-1)
        quad = diff_flat @ jnp.linalg.solve(Q, diff_flat)
        s, ld = jnp.linalg.slogdet(Q)
        ld_ = -0.5*quad - 0.5*ld - 0.5*dim*jnp.log(2*jnp.pi)
        return jnp.where(det, 0.0, ld_)
    tr_loss = -jnp.sum(jax.vmap(tr_step)(tr_idx, phis))
    return float(obs_loss), float(tr_loss)

def main():
    data, mi, t2n = get_results(start_date="1900-01-01", end_date="2025-12-31",
                                max_goals=8, teams_only=WORLDCUP_2026_TEAMS)
    M = len(t2n)
    params = default_init_params(M, team_id_to_name=t2n)
    # simulate a "returned" params after an aggressive M-step: kappa at floor,
    # gamma_0/B projected but maybe with near-floor eigenvalues.
    p = _constrain(params)
    g_upd, g_pred, K = compute_gamma_trajectory(mi, p.gamma_0, p.kappa, M)
    aug = generate_augmented_data(mi, g_upd, g_pred, K)
    _, smoothed, lm = E_step(p, aug, M, n_particles=10, key=jax.random.PRNGKey(0))
    total = loss_fn(p, smoothed, aug)
    obs, tr = decompose_loss(p, smoothed, aug)
    print("total loss:", float(total))
    print("  obs_loss:", obs)
    print("  transition_loss:", tr)
    g = np.array(p.gamma_0)
    b = np.array(p.B)
    print("gamma_0 eig (min, max):", np.linalg.eigvalsh(g).min(), np.linalg.eigvalsh(g).max())
    print("B eig:", np.linalg.eigvalsh(b))
    # what are the smallest eigenvalues of Q = kron(gamma_0, B)?
    q_min = np.linalg.eigvalsh(g).min() * np.linalg.eigvalsh(b).min()
    print("smallest eigenvalue of Q = kron(gamma_0,B) ~", q_min)

if __name__ == "__main__":
    main()
