import jax
import os
from rbpf_v3.src.helpers import default_init_params, load_params, generate_rbpf_trajectory
from rbpf_v3.src.model import compute_gamma_trajectory, run_filter
from rbpf_v3.src.graphic import plot_all

MAX_GOALS = 8
N = 100

def main():

    from rbpf_v3.src.data import get_results, WORLDCUP_2026_TEAMS
    data, model_inputs, team_id_to_name = get_results(
        start_date="2000-01-01",
        end_date="2026-01-01",
        max_goals=MAX_GOALS,
        include_friendly=False,
        teams_only=WORLDCUP_2026_TEAMS,
    )
    print("DataFrame head:")
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(3))
    print(data[['date', 'home_team', 'away_team', 'home_score', 'away_score']].tail(3))
    num_teams = len(team_id_to_name)
    key = jax.random.PRNGKey(42)
    key, filter_key = jax.random.split(key)

    # Resolve the params file to an absolute path and check it exists.
    params_path = os.path.abspath("rbpf/outputs/smoothing/em_best_params.json")
    if not os.path.exists(params_path):
        print(f"ERROR: EM params file not found: {params_path}")
        print("Run `uv run -m rbpf_v3.src.smoothing` first to generate it.")
        return
    print(f"Loading EM parameters from {params_path}...")
    params = load_params(params_path)
    gamma_updated, gamma_pred, kalman_gain = compute_gamma_trajectory(
        model_inputs=model_inputs,
        gamma_0=params.gamma_0,
        kappa=params.kappa
    )
    model_inputs_rbpf = generate_rbpf_trajectory(
        model_inputs=model_inputs,
        gamma_updated=gamma_updated,
        gamma_pred=gamma_pred,
        kalman_gain=kalman_gain
    )
    print("Running filter (OU)...")
    try:
        filtered_states, model_inputs_rbpf = run_filter(
            key=filter_key,
            model_inputs=model_inputs_rbpf,
            params=params,
            num_teams=num_teams,
            n_particles=N,
        )
    except Exception as e:
        print("Error during filtering:", e)
        return

    print("Final log-normalizing constant:", filtered_states.log_normalizing_constant[-1])

    # Resolve the output directory to an absolute path and print it clearly.
    save_path = os.path.abspath("./rbpf/outputs/trained")
    print(f"Output directory (absolute): {save_path}")
    # Plot the results
    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=save_path,
        timestamps=data["date"].to_numpy(),
    )

if __name__ == "__main__":
    main()