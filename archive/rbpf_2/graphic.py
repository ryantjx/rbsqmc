"""Visualization utilities for RBPF filter results."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from data import FootballResults


def _normalize_weights(log_weights: jax.Array) -> jax.Array:
    """Convert log weights to normalized weights."""
    return jnp.exp(log_weights - jax.nn.logsumexp(log_weights))


def weighted_mean(particles: jax.Array, log_weights: jax.Array) -> jax.Array:
    """Compute weighted mean of particles at a single time step.

    Args:
        particles: shape (N, ...) — particle values
        log_weights: shape (N,) — log weights

    Returns:
        Weighted mean with shape (...) — same as a single particle minus the N axis.
    """
    weights = _normalize_weights(log_weights)
    # Expand weights to broadcast over all non-N axes
    njax = jnp.ndim(particles)
    if njax > 1:
        weights_expanded = weights.reshape((-1,) + (1,) * (njax - 1))
        return jnp.sum(particles * weights_expanded, axis=0)
    return jnp.sum(particles * weights, axis=0)


def plot_team_strengths(
    result,
    team_id_to_name: dict[int, str],
    team_indices: list[int] | None = None,
    max_teams: int = 10,
    save_path: str | None = None,
):
    """Plot attack/defense strength trajectories for selected teams.

    Args:
        result: ParticleFilterState from cuthbert.filtering.filter.
            result.particles.x has shape (T+1, N, NUM_TEAMS, 2)
            result.log_weights has shape (T+1, N)
        team_id_to_name: mapping from team ID to team name.
        team_indices: specific team IDs to plot. If None, picks the
            most frequently observed teams (up to max_teams).
        max_teams: max number of teams to plot if team_indices is None.
        save_path: if provided, save figure to this path.
    """
    # shapes: (T+1, N, NUM_TEAMS, 2), (T+1, N)
    all_x = result.particles.x
    all_log_w = result.log_weights
    T_plus_1, N_particles, num_teams, _ = all_x.shape

    # Compute weighted mean at each time step: (T+1, NUM_TEAMS, 2)
    weights = jax.vmap(_normalize_weights)(all_log_w)  # (T+1, N)
    x_mean = jnp.einsum("tn,tndp->tdp", weights, all_x)  # (T+1, NUM_TEAMS, 2)

    # Compute weighted std for confidence bands
    x_sq_mean = jnp.einsum("tn,tndp->tdp", weights, all_x ** 2)
    x_std = jnp.sqrt(jnp.maximum(x_sq_mean - x_mean ** 2, 0.0))

    x_mean = np.array(x_mean)
    x_std = np.array(x_std)

    # Select teams to plot
    if team_indices is None:
        team_indices = list(range(min(max_teams, num_teams)))

    n_plot = len(team_indices)
    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = axes[None, :]

    t_range = np.arange(T_plus_1)

    for row, tid in enumerate(team_indices):
        name = team_id_to_name.get(tid, f"Team {tid}")
        att_mean = x_mean[:, tid, 0]
        att_std = x_std[:, tid, 0]
        def_mean = x_mean[:, tid, 1]
        def_std = x_std[:, tid, 1]

        # Attack
        ax = axes[row, 0]
        ax.plot(t_range, att_mean, "b-", alpha=0.8, label="Attack")
        ax.fill_between(
            t_range, att_mean - att_std, att_mean + att_std, alpha=0.2, color="blue"
        )
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Attack Strength")
        if row == n_plot - 1:
            ax.set_xlabel("Time step")

        # Defense
        ax = axes[row, 1]
        ax.plot(t_range, def_mean, "r-", alpha=0.8, label="Defense")
        ax.fill_between(
            t_range, def_mean - def_std, def_mean + def_std, alpha=0.2, color="red"
        )
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Defense Strength")
        if row == n_plot - 1:
            ax.set_xlabel("Time step")

    plt.suptitle("RBPF Team Strength Trajectories (mean ± 1 std)", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved team strengths plot to {save_path}")
    plt.close()


def plot_gamma_heatmap(
    gamma_trajectory: jax.Array,
    time_index: int = -1,
    team_id_to_name: dict[int, str] | None = None,
    max_teams: int = 20,
    save_path: str | None = None,
):
    """Plot the Gamma (inter-team covariance) matrix at a given time step.

    Gamma is precomputed deterministically (one copy per time step, not per particle).

    Args:
        gamma_trajectory: shape (T+1, NUM_TEAMS, NUM_TEAMS) from compute_gamma_trajectory.
        time_index: which time step to visualize (default: last).
        team_id_to_name: optional team name mapping for axis labels.
        max_teams: max number of teams to show (truncates large matrices).
        save_path: if provided, save figure to this path.
    """
    if gamma_trajectory is None:
        print("Skipping gamma heatmap — no gamma trajectory provided")
        return

    # gamma_trajectory: (T+1, NUM_TEAMS, NUM_TEAMS) — one copy per step, not per particle
    t = time_index
    gamma_mean = np.array(gamma_trajectory[t])  # (NUM_TEAMS, NUM_TEAMS)

    num_teams = gamma_mean.shape[0]
    n_show = min(max_teams, num_teams)
    gamma_show = gamma_mean[:n_show, :n_show]

    fig, ax = plt.subplots(figsize=(max(8, n_show * 0.6), max(6, n_show * 0.6)))
    im = ax.imshow(gamma_show, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Covariance")

    # Labels
    if team_id_to_name is not None:
        labels = [team_id_to_name.get(i, str(i)) for i in range(n_show)]
        ax.set_xticks(range(n_show))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks(range(n_show))
        ax.set_yticks(range(n_show))

    ax.set_title(f"Gamma matrix (inter-team covariance) at t={t}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved gamma heatmap to {save_path}")
    plt.close()


def plot_ess(
    result,
    save_path: str | None = None,
):
    """Plot Effective Sample Size (ESS) over time.

    Args:
        result: ParticleFilterState from cuthbert.filtering.filter.
        save_path: if provided, save figure to this path.
    """
    all_log_w = result.log_weights  # (T+1, N)
    T_plus_1, N_particles = all_log_w.shape

    # ESS = 1 / sum(w^2) at each time step
    weights = jax.vmap(_normalize_weights)(all_log_w)  # (T+1, N)
    ess = 1.0 / jnp.sum(weights ** 2, axis=1)  # (T+1,)
    ess = np.array(ess)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ess, "g-", alpha=0.8)
    ax.axhline(N_particles, color="gray", linestyle="--", alpha=0.5, label=f"N={N_particles}")
    ax.axhline(N_particles / 2, color="orange", linestyle="--", alpha=0.5, label=f"N/2={N_particles // 2}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("ESS")
    ax.set_title("Effective Sample Size over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ESS plot to {save_path}")
    plt.close()


def plot_log_normalizing_constant(
    result,
    save_path: str | None = None,
):
    """Plot the log normalizing constant (log marginal likelihood) over time.

    Args:
        result: ParticleFilterState from cuthbert.filtering.filter.
        save_path: if provided, save figure to this path.
    """
    log_z = np.array(result.log_normalizing_constant)  # scalar or (T+1,)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(log_z, "b-", alpha=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("log Z")
    ax.set_title("Log Normalizing Constant (log marginal likelihood)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved log normalizing constant plot to {save_path}")
    plt.close()


def plot_em_convergence(
    log_marginals: list[float],
    save_path: str | None = None,
):
    """Plot EM log marginal likelihood convergence across epochs.

    Args:
        log_marginals: list of log marginal likelihoods, one per EM epoch.
        save_path: if provided, save figure to this path.
    """
    epochs = np.arange(1, len(log_marginals) + 1)
    log_marginals = np.array(log_marginals)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, log_marginals, "o-", color="darkgreen", alpha=0.8, linewidth=2)
    best_idx = int(np.argmax(log_marginals))
    ax.plot(epochs[best_idx], log_marginals[best_idx], "r*", markersize=15,
            label=f"Best: epoch {epochs[best_idx]} = {log_marginals[best_idx]:.2f}")
    ax.axhline(log_marginals[0], color="gray", linestyle="--", alpha=0.4,
               label=f"Initial: {log_marginals[0]:.2f}")
    ax.set_xlabel("EM Epoch")
    ax.set_ylabel("Log Marginal Likelihood")
    ax.set_title("EM Convergence (log marginal likelihood vs epoch)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved EM convergence plot to {save_path}")
    plt.close()


def plot_final_strengths_bar(
    result,
    team_id_to_name: dict[int, str],
    max_teams: int = 20,
    save_path: str | None = None,
):
    """Plot a bar chart of final team strengths (attack & defense) sorted by attack.

    Args:
        result: ParticleFilterState from cuthbert.filtering.filter.
        team_id_to_name: mapping from team ID to team name.
        max_teams: max number of teams to show.
        save_path: if provided, save figure to this path.
    """
    all_x = result.particles.x  # (T+1, N, NUM_TEAMS, 2)
    all_log_w = result.log_weights  # (T+1, N)

    # Final time step weighted mean
    log_w = all_log_w[-1]
    weights = _normalize_weights(log_w)
    x_final = jnp.einsum("n,ndp->dp", weights, all_x[-1])  # (NUM_TEAMS, 2)
    x_final = np.array(x_final)

    num_teams = x_final.shape[0]
    n_show = min(max_teams, num_teams)

    # Sort by attack strength (descending)
    order = np.argsort(x_final[:n_show, 0])[::-1]

    names = [team_id_to_name.get(i, f"Team {i}") for i in order]
    att = x_final[order, 0]
    defn = x_final[order, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, n_show * 0.4)))
    y_pos = np.arange(n_show)

    ax1.barh(y_pos, att, color="steelblue", alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Attack Strength")
    ax1.set_title("Final Attack Strength (top teams)")
    ax1.grid(True, axis="x", alpha=0.3)

    ax2.barh(y_pos, defn, color="salmon", alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Defense Strength")
    ax2.set_title("Final Defense Strength")
    ax2.grid(True, axis="x", alpha=0.3)

    plt.suptitle("RBPF Final Team Strengths", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved final strengths bar chart to {save_path}")
    plt.close()


def plot_correlation_matrix(
    result,
    time_index: int = -1,
    feature: int = 0,
    team_id_to_name: dict[int, str] | None = None,
    max_teams: int = 20,
    save_path: str | None = None,
):
    """Plot Pearson correlation matrix of team strengths across particles at a given time step.

    Computes the correlation between each pair of teams across the N particles,
    showing how team strengths co-vary in the particle distribution.

    Args:
        result: ParticleFilterState from cuthbert.filtering.filter.
            result.particles.x has shape (T+1, N, NUM_TEAMS, 2)
            result.log_weights has shape (T+1, N)
        time_index: which time step to visualize (0=initial, -1=final).
        feature: 0 for attack strength, 1 for defense strength.
        team_id_to_name: optional team name mapping for axis labels.
        max_teams: max number of teams to show (truncates large matrices).
        save_path: if provided, save figure to this path.
    """
    all_x = result.particles.x  # (T+1, N, NUM_TEAMS, 2)
    x_t = np.array(all_x[time_index])  # (N, NUM_TEAMS, 2)

    # Extract feature values: (N, NUM_TEAMS)
    values = x_t[:, :, feature]

    # Compute Pearson correlation matrix across teams (variables in columns)
    corr = np.corrcoef(values, rowvar=False)  # (NUM_TEAMS, NUM_TEAMS)

    num_teams = corr.shape[0]
    n_show = min(max_teams, num_teams)
    corr_show = corr[:n_show, :n_show]

    fig, ax = plt.subplots(figsize=(max(8, n_show * 0.6), max(6, n_show * 0.6)))
    im = ax.imshow(corr_show, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson correlation")

    # Labels
    if team_id_to_name is not None:
        labels = [team_id_to_name.get(i, str(i)) for i in range(n_show)]
        ax.set_xticks(range(n_show))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_xticks(range(n_show))
        ax.set_yticks(range(n_show))

    feature_name = "Attack" if feature == 0 else "Defense"
    time_label = "initial" if time_index == 0 else "final"
    ax.set_title(f"Team {feature_name} Correlation Matrix ({time_label} state, t={time_index})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved correlation matrix to {save_path}")
    plt.close()


def generate_all_plots(
    result,
    gamma_trajectory: jax.Array | None,
    team_id_to_name: dict[int, str],
    output_dir: str = "./outputs",
    team_indices: list[int] | None = None,
    max_teams: int = 10,
):
    """Generate all diagnostic plots from the filter result.

    Args:
        result: ParticleFilterState from cuthbert.filtering.filter.
            result.particles.x has shape (T+1, N, NUM_TEAMS, 2)
            result.log_weights has shape (T+1, N)
        gamma_trajectory: (T+1, NUM_TEAMS, NUM_TEAMS) from compute_gamma_trajectory,
            or None if not available.
        team_id_to_name: mapping from team ID to team name.
        output_dir: directory to save plots.
        team_indices: specific team IDs to plot trajectories for.
        max_teams: max teams for trajectory and bar plots.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_team_strengths(
        result, team_id_to_name,
        team_indices=team_indices,
        max_teams=max_teams,
        save_path=f"{output_dir}/team_strengths.png",
    )

    plot_gamma_heatmap(
        gamma_trajectory,
        time_index=-1,
        team_id_to_name=team_id_to_name,
        max_teams=20,
        save_path=f"{output_dir}/gamma_heatmap.png",
    )

    plot_ess(
        result,
        save_path=f"{output_dir}/ess.png",
    )

    plot_log_normalizing_constant(
        result,
        save_path=f"{output_dir}/log_normalizing_constant.png",
    )

    plot_final_strengths_bar(
        result, team_id_to_name,
        max_teams=max_teams,
        save_path=f"{output_dir}/final_strengths.png",
    )

    # Initial and final correlation matrices (attack and defense)
    plot_correlation_matrix(
        result,
        time_index=0,
        feature=0,
        team_id_to_name=team_id_to_name,
        max_teams=max_teams,
        save_path=f"{output_dir}/correlation_initial_attack.png",
    )
    plot_correlation_matrix(
        result,
        time_index=0,
        feature=1,
        team_id_to_name=team_id_to_name,
        max_teams=max_teams,
        save_path=f"{output_dir}/correlation_initial_defense.png",
    )
    plot_correlation_matrix(
        result,
        time_index=-1,
        feature=0,
        team_id_to_name=team_id_to_name,
        max_teams=max_teams,
        save_path=f"{output_dir}/correlation_final_attack.png",
    )
    plot_correlation_matrix(
        result,
        time_index=-1,
        feature=1,
        team_id_to_name=team_id_to_name,
        max_teams=max_teams,
        save_path=f"{output_dir}/correlation_final_defense.png",
    )

    print(f"\nAll plots saved to {output_dir}/")


def save_results(
    result,
    gamma_trajectory: jax.Array,
    team_id_to_name: dict[int, str],
    output_dir: str = "./outputs",
):
    """Save filter results to Parquet files for analysis and parameter estimation.

    Saves:
        - states.parquet: weighted mean/std of team strengths per time step
        - gamma_trajectory.parquet: gamma matrix per time step (flattened)
        - log_weights.parquet: particle log weights per time step
        - log_z.parquet: log normalizing constant per time step

    Args:
        result: ParticleFilterState from cuthbert.filtering.filter.
            result.particles.x has shape (T+1, N, NUM_TEAMS, 2)
            result.log_weights has shape (T+1, N)
            result.log_normalizing_constant has shape (T+1,)
        gamma_trajectory: (T+1, NUM_TEAMS, NUM_TEAMS) from compute_gamma_trajectory.
        team_id_to_name: mapping from team ID to team name.
        output_dir: directory to save parquet files.
    """
    import os
    import polars as pl

    os.makedirs(output_dir, exist_ok=True)

    all_x = result.particles.x          # (T+1, N, NUM_TEAMS, 2)
    all_log_w = result.log_weights       # (T+1, N)
    log_z = result.log_normalizing_constant  # (T+1,)
    T_plus_1, N_particles, num_teams, _ = all_x.shape

    # --- 1. States: weighted mean and std per time step ---
    weights = jax.vmap(_normalize_weights)(all_log_w)  # (T+1, N)
    x_mean = jnp.einsum("tn,tndp->tdp", weights, all_x)     # (T+1, NUM_TEAMS, 2)
    x_sq_mean = jnp.einsum("tn,tndp->tdp", weights, all_x ** 2)
    x_std = jnp.sqrt(jnp.maximum(x_sq_mean - x_mean ** 2, 0.0))

    x_mean = np.array(x_mean)
    x_std = np.array(x_std)

    states_data = []
    for t in range(T_plus_1):
        row: dict[str, float | int] = {"timestep": t}
        for d in range(num_teams):
            name = team_id_to_name.get(d, f"Team {d}")
            row[f"{name}_att_mean"] = float(x_mean[t, d, 0])
            row[f"{name}_att_std"] = float(x_std[t, d, 0])
            row[f"{name}_def_mean"] = float(x_mean[t, d, 1])
            row[f"{name}_def_std"] = float(x_std[t, d, 1])
        states_data.append(row)

    pl.DataFrame(states_data).write_parquet(f"{output_dir}/states.parquet")
    print(f"Saved states to {output_dir}/states.parquet")

    # --- 2. Gamma trajectory (flattened per time step) ---
    gamma_np = np.array(gamma_trajectory)  # (T+1, NUM_TEAMS, NUM_TEAMS)
    gamma_data = []
    for t in range(T_plus_1):
        row: dict[str, float | int] = {"timestep": t}
        for i in range(num_teams):
            for j in range(num_teams):
                row[f"gamma_{i}_{j}"] = float(gamma_np[t, i, j])
        gamma_data.append(row)

    pl.DataFrame(gamma_data).write_parquet(f"{output_dir}/gamma_trajectory.parquet")
    print(f"Saved gamma trajectory to {output_dir}/gamma_trajectory.parquet")

    # --- 3. Log weights ---
    log_w_np = np.array(all_log_w)  # (T+1, N)
    log_w_data = []
    for t in range(T_plus_1):
        row: dict[str, float | int] = {"timestep": t}
        for n in range(N_particles):
            row[f"particle_{n}"] = float(log_w_np[t, n])
        log_w_data.append(row)

    pl.DataFrame(log_w_data).write_parquet(f"{output_dir}/log_weights.parquet")
    print(f"Saved log weights to {output_dir}/log_weights.parquet")

    # --- 4. Log normalizing constant ---
    log_z_np = np.array(log_z)  # (T+1,)
    pl.DataFrame({
        "timestep": list(range(T_plus_1)),
        "log_z": [float(log_z_np[t]) for t in range(T_plus_1)],
    }).write_parquet(f"{output_dir}/log_z.parquet")
    print(f"Saved log normalizing constant to {output_dir}/log_z.parquet")