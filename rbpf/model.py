import jax
import jax.numpy as jnp
from bivariate_poisson import loglik
from data import download_results

def init_sample(model_inputs, init_mean, init_cov, num_teams):
    pass
def propagate_sample():
    pass

def main():
    data, results, team_id_to_name = download_results(start_date="2020-01-01", max_goals=8)

    NUM_TEAMS = len(team_id_to_name)
    INIT_MEAN = jnp.zeros(NUM_TEAMS * 2)
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (NUM_TEAMS, 2))
    INIT_GAMMA = A @ A.T + 1.0 * jnp.eye(2)
    B = jax.random.normal(key, (2, 2))
    INIT_B = B @ B.T + 1.0 * jnp.eye(2)
    # INIT_COV = jnp.kron(INIT_GAMMA, INIT_BETA)
    INIT_KAPPA = jnp.eye(2)
    print("Initial mean:", INIT_MEAN)
    print("Initial gamma:", INIT_GAMMA)
    print("Initial beta:", INIT_B)
    print("Initial kappa:", INIT_KAPPA)

if __name__ == "__main__":
    main()