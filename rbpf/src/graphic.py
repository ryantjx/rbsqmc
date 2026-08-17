"""Visualization helpers for the RBPF filter results.

Functions:
  - top_teams_by_strength: top-N team indices by attack / defense / total
  - plot_top_strengths: bar chart of top-N attack/defense/total strengths
  - plot_timeseries_states: attack and defense filter trajectories over time
  - covariance_matrix: final-state team covariance Gamma_T
  - correlation_matrix: final-state team correlation matrix
  - plot_correlation_matrix: heatmap of between-team correlation at final state
  - plot_log_normalizing_constant: line plot of filter log marginal likelihood
  - plot_all: generate all plots and save them
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphic")


# ---------------------------------------------------------------------------
# 1. Top teams by attack / defense / total strength
# ---------------------------------------------------------------------------
def top_teams_by_strength(filtered_states, top_n=5, rank_by="attack"):
    """Return the indices of the top-N teams by a final-state strength component.

    Args:
        filtered_states: cuthbert FilterStates with particles.x shape
            ``(T, N, M, 2)`` (attack, defense).
        top_n: number of teams to return.
        rank_by: ``"attack"``, ``"defense"``, or ``"total"`` (attack + defense).

    Returns:
        ``np.ndarray`` of team indices, best first.
    """
    x_final = np.asarray(filtered_states.particles.x[-1])  # (N, M, 2)
    mean_strengths = x_final.mean(axis=0)  # (M, 2)

    if rank_by == "attack":
        scores = mean_strengths[:, 0]
    elif rank_by == "defense":
        scores = mean_strengths[:, 1]
    elif rank_by == "total":
        scores = mean_strengths[:, 0] + mean_strengths[:, 1]
    else:
        raise ValueError("rank_by must be 'attack', 'defense', or 'total'")

    top_n = min(int(top_n), scores.size)
    return np.argsort(scores)[-top_n:][::-1]


def plot_top_strengths(filtered_states, team_id_to_name, top_n=5,
                       save_path=os.path.join(OUTPUT_DIR, "top_strengths.png")):
    """Bar chart of top-N attack, defense, and total strengths at final timestep.

    Args:
        filtered_states: cuthbert FilterStates with particles.x shape (T, N, M, 2)
        team_id_to_name: dict mapping team_id -> team name
        top_n: number of teams to show
        save_path: if given, save figure to this path
    """
    x_final = np.asarray(filtered_states.particles.x[-1])  # (N, M, 2)
    mean_strengths = x_final.mean(axis=0)  # (M, 2)

    attack = mean_strengths[:, 0]
    defense = mean_strengths[:, 1]
    total = attack + defense

    top_attack_idx = np.argsort(attack)[-top_n:][::-1]
    top_defense_idx = np.argsort(defense)[-top_n:][::-1]
    top_total_idx = np.argsort(total)[-top_n:][::-1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.barh([team_id_to_name[i] for i in top_attack_idx],
             attack[top_attack_idx], color="steelblue")
    ax1.set_xlabel("Attack Strength")
    ax1.set_title(f"Top {top_n} Attack Strengths")
    ax1.invert_yaxis()

    ax2.barh([team_id_to_name[i] for i in top_defense_idx],
             defense[top_defense_idx], color="firebrick")
    ax2.set_xlabel("Defense Strength")
    ax2.set_title(f"Top {top_n} Defense Strengths")
    ax2.invert_yaxis()

    ax3.barh([team_id_to_name[i] for i in top_total_idx],
             total[top_total_idx], color="darkgreen")
    ax3.set_xlabel("Total Strength (Attack + Defense)")
    ax3.set_title(f"Top {top_n} Total Strengths")
    ax3.invert_yaxis()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {os.path.abspath(save_path)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Filter states over time for top teams
# ---------------------------------------------------------------------------
def plot_timeseries_states(
    filtered_states,
    team_id_to_name,
    top_n=5,
    rank_by="attack",
    timestamps=None,
    save_path=os.path.join(OUTPUT_DIR, "timeseries_states.png"),
):
    """Plot attack and defense filtered states over time for the top teams.

    Teams are selected using their final filtered posterior mean. Particle
    log weights are used when available; otherwise the particle mean is used.

    The x-axis uses real observation timestamps (e.g. match dates) when
    ``timestamps`` is supplied. ``filtered_states`` carries a prepended
    initial (prior) state, so if ``timestamps`` has one fewer entry than the
    number of states, a reference point is prepended to keep the axis aligned.

    Args:
        filtered_states: cuthbert FilterStates with particles.x shaped
            ``(T, N, M, 2)``.
        team_id_to_name: dict mapping team index -> team name.
        top_n: number of teams to plot.
        rank_by: final state component used to select teams: ``"attack"``,
            ``"defense"``, or ``"total"``.
        timestamps: optional x-axis values (e.g. match dates).
        save_path: if given, save the figure to this path.

    Returns:
        ``(figure, (attack_axis, defense_axis))``.
    """
    x_history = np.asarray(filtered_states.particles.x)
    if x_history.ndim != 4 or x_history.shape[-1] < 2:
        raise ValueError(
            "filtered_states.particles.x must have shape (T, N, M, K) with K >= 2"
        )

    n_steps, n_particles, n_teams, _ = x_history.shape
    if n_steps == 0 or n_teams == 0:
        raise ValueError("filtered_states must contain at least one state and team")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    top_n = min(int(top_n), n_teams)
    if rank_by not in {"attack", "defense", "total"}:
        raise ValueError("rank_by must be 'attack', 'defense', or 'total'")

    log_weights = np.asarray(getattr(filtered_states, "log_weights", None))
    if log_weights.shape == (n_steps, n_particles):
        finite_log_weights = np.where(np.isfinite(log_weights), log_weights, -np.inf)
        max_log_weight = np.max(finite_log_weights, axis=1, keepdims=True)
        shifted_weights = np.exp(log_weights - max_log_weight)
        shifted_weights[~np.isfinite(shifted_weights)] = 0.0
        weight_sum = shifted_weights.sum(axis=1, keepdims=True)
        uniform_weights = np.full_like(shifted_weights, 1.0 / n_particles)
        weights = np.divide(shifted_weights, weight_sum,
                            out=uniform_weights, where=weight_sum > 0)
        filter_means = np.sum(x_history[..., :2] * weights[:, :, None, None], axis=1)
    else:
        filter_means = x_history[..., :2].mean(axis=1)

    final_means = filter_means[-1]
    if rank_by == "attack":
        ranking_scores = final_means[:, 0]
    elif rank_by == "defense":
        ranking_scores = final_means[:, 1]
    else:
        ranking_scores = final_means[:, 0] + final_means[:, 1]
    top_indices = np.argsort(ranking_scores)[-top_n:][::-1]

    if timestamps is None:
        model_inputs = getattr(filtered_states, "model_inputs", None)
        timestamps = getattr(model_inputs, "timestamp", None)
    if timestamps is None:
        timestamps = np.arange(n_steps)
    else:
        timestamps = np.asarray(timestamps).reshape(-1)
        if timestamps.size == n_steps - 1:
            # filtered_states prepends the initial (prior) state at index 0,
            # so the match dates are one short; prepend a reference point to
            # keep the date x-axis aligned with the states.
            timestamps = np.concatenate([timestamps[:1], timestamps])
        elif timestamps.size != n_steps:
            timestamps = np.arange(n_steps)

    names = [team_id_to_name.get(int(i), str(i)) for i in top_indices]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, top_n))
    fig, (attack_ax, defense_ax) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    is_datetime = np.asarray(timestamps).dtype.kind in {"M", "O"}
    if is_datetime:
        import matplotlib.dates as mdates
        attack_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        attack_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

    for team_idx, name, color in zip(top_indices, names, colors):
        attack_ax.plot(timestamps, filter_means[:, team_idx, 0],
                       color=color, linewidth=1.8, label=name)
        defense_ax.plot(timestamps, filter_means[:, team_idx, 1],
                       color=color, linewidth=1.8, label=name)

    rank_label = {"attack": "attack", "defense": "defense",
                  "total": "attack + defense"}[rank_by]
    attack_ax.set_title(f"Top {top_n} Teams by Final Filtered {rank_label}")
    attack_ax.set_ylabel("Attack state")
    defense_ax.set_ylabel("Defense state")
    defense_ax.set_xlabel("Date" if is_datetime else "Time")
    for axis in (attack_ax, defense_ax):
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {os.path.abspath(save_path)}")

    return fig, (attack_ax, defense_ax)


# ---------------------------------------------------------------------------
# 2. Covariance matrix of the final filtered state (Gamma_T)
# ---------------------------------------------------------------------------
def covariance_matrix(augmented_results):
    """Return the final-state team covariance matrix Gamma_T.

    Args:
        augmented_results: RBPFFootballResults with gamma shape (T, M, M).

    Returns:
        ``np.ndarray`` of shape ``(M, M)``.
    """
    return np.asarray(augmented_results.gamma[-1])


# ---------------------------------------------------------------------------
# 3. Correlation matrix of the final filtered state
# ---------------------------------------------------------------------------
def correlation_matrix(augmented_results):
    """Return the final-state team correlation matrix.

    With the Kronecker structure, ``gamma`` is already the ``M x M`` team
    covariance (the attack/defence factor ``B`` is shared and does not affect
    between-team correlation). We normalize its diagonal to a correlation.

    Args:
        augmented_results: RBPFFootballResults with gamma shape (T, M, M).

    Returns:
        ``(corr, std)`` where ``corr`` is the ``(M, M)`` correlation matrix
        and ``std`` is the per-team standard deviation.
    """
    gamma_final = covariance_matrix(augmented_results)
    std = np.sqrt(np.diag(gamma_final))
    std_safe = np.where(std > 1e-10, std, 1.0)
    corr = gamma_final / np.outer(std_safe, std_safe)
    corr = np.clip(corr, -1, 1)
    return corr, std


def plot_correlation_matrix(augmented_results, team_id_to_name, num_teams=None,
                            save_path=os.path.join(OUTPUT_DIR, "correlation_matrix.png")):
    """Heatmap of between-team correlation matrix at final timestep.

    Args:
        augmented_results: RBPFFootballResults with gamma shape (T, M, M)
        team_id_to_name: dict mapping team_id -> team name
        num_teams: number of teams (inferred from team_id_to_name if None)
        save_path: if given, save figure to this path
    """
    if num_teams is None:
        num_teams = len(team_id_to_name)
    corr, std = correlation_matrix(augmented_results)

    active = std > 1e-10
    active_idx = np.where(active)[0]
    corr_sub = corr[np.ix_(active_idx, active_idx)]
    names = [team_id_to_name[i] for i in active_idx]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_sub, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title("Team Correlation Matrix (Final State)")
    fig.colorbar(im, ax=ax, label="Correlation")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {os.path.abspath(save_path)}")
    plt.close(fig)


def plot_log_normalizing_constant(filtered_states,
                                  save_path=os.path.join(OUTPUT_DIR, "log_normalizing_constant.png")):
    """Line plot of the log normalizing constant over time.

    Args:
        filtered_states: cuthbert FilterStates with log_normalizing_constant shape (T+1,)
        save_path: if given, save figure to this path
    """
    log_z = np.asarray(filtered_states.log_normalizing_constant)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(log_z, color="darkgreen", linewidth=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Log Normalizing Constant")
    ax.set_title("Filter Log Marginal Likelihood")
    ax.grid(True, alpha=0.3)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {os.path.abspath(save_path)}")
    plt.close(fig)


def plot_log_marginal_likelihood_curve(
    log_marginal_likelihoods,
    save_path=os.path.join(OUTPUT_DIR, "em_log_marginal_likelihood_curve.png"),
):
    """Plot the per-epoch log marginal likelihood curve of an EM run.

    Args:
        log_marginal_likelihoods: array/list for ``theta_0`` through the final
            ``theta_K`` (length = n_epochs + 1 for current ``run_EM`` output).
        save_path: if given, save the figure to this path.

    Returns:
        ``(figure, axis)``.
    """
    log_mll = np.asarray(log_marginal_likelihoods).reshape(-1)
    # run_EM stores theta_0 before the first M-step and appends the final
    # theta_K likelihood after the last M-step.
    epochs = np.arange(log_mll.size)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, log_mll, marker="o", linewidth=1.8, color="darkgreen")
    ax.set_xlabel("Parameter index (theta_k)")
    ax.set_ylabel("Log Marginal Likelihood")
    ax.set_title("EM Log Marginal Likelihood Curve")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {os.path.abspath(save_path)}")

    return fig, ax


def plot_em_covariance_diagnostics(diagnostics_history, save_path):
    """Plot Gamma_0 eigenvalue, conditioning and scale diagnostics."""
    if not diagnostics_history:
        return None

    parameter_index = np.asarray(
        [diagnostics_history[0]["e_step_parameter_index"]]
        + [entry["updated_parameter_index"] for entry in diagnostics_history]
    )
    parameter = [diagnostics_history[0]["e_step_parameters"]] + [
        entry["updated_parameters"] for entry in diagnostics_history
    ]

    minimum = np.asarray([entry["gamma_min_eigenvalue"] for entry in parameter])
    maximum = np.asarray([entry["gamma_max_eigenvalue"] for entry in parameter])
    condition = np.asarray([entry["gamma_condition_number"] for entry in parameter])
    trace = np.asarray([entry["gamma_trace"] for entry in parameter])
    effective_rank = np.asarray([entry["gamma_effective_rank"] for entry in parameter])
    logdet = np.asarray([entry["gamma_logdet"] for entry in parameter])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].plot(parameter_index, minimum, marker="o", label="minimum")
    axes[0, 0].plot(parameter_index, maximum, marker="o", label="maximum")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("Eigenvalue (log scale)")
    axes[0, 0].set_title("Gamma_0 eigenvalue range")
    axes[0, 0].legend()

    axes[0, 1].plot(parameter_index, condition, marker="o", color="#d62728")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("Condition number (log scale)")
    axes[0, 1].set_title("Gamma_0 conditioning")

    axes[1, 0].plot(parameter_index, trace, marker="o", label="trace")
    axes[1, 0].plot(
        parameter_index, effective_rank, marker="o", label="effective rank"
    )
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].set_title("Scale and effective rank")
    axes[1, 0].legend()

    axes[1, 1].plot(parameter_index, logdet, marker="o", color="#9467bd")
    axes[1, 1].set_ylabel("log det(Gamma_0)")
    axes[1, 1].set_title("Gamma_0 log determinant")

    for axis in axes.ravel():
        axis.set_xlabel("Parameter index (theta_k)")
        axis.grid(True, alpha=0.3)
    fig.suptitle("EM covariance diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_em_path_diagnostics(diagnostics_history, save_path):
    """Plot full-state representation and smoother-diversity checks."""
    if not diagnostics_history:
        return None

    epochs = np.asarray([entry["epoch"] + 1 for entry in diagnostics_history])
    cloud = [entry["materialized_cloud"] for entry in diagnostics_history]
    path = [entry["smoother_paths"] for entry in diagnostics_history]
    cloud_initial_ratio = np.asarray(
        [entry["initial_mahalanobis_ratio"] for entry in cloud]
    )
    initial_ratio = np.asarray(
        [entry["initial_mahalanobis_ratio"] for entry in path]
    )
    transition_ratio = np.asarray(
        [entry["transition_mahalanobis_ratio"] for entry in path]
    )
    unique_min = np.asarray(
        [entry["unique_smoother_particles_min"] for entry in path]
    )
    unique_mean = np.asarray(
        [entry["unique_smoother_particles_mean"] for entry in path]
    )
    unique_final = np.asarray(
        [entry["unique_smoother_particles_final"] for entry in path]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        epochs,
        cloud_initial_ratio,
        marker="o",
        label="materialized cloud initial q / dimension",
    )
    axes[0].plot(
        epochs,
        initial_ratio,
        marker="o",
        label="smoothed initial q / dimension",
    )
    axes[0].plot(
        epochs,
        transition_ratio,
        marker="o",
        label="transition q / dimension",
    )
    axes[0].axhline(
        1.0, color="black", linestyle="--", label="unconditional reference"
    )
    axes[0].axhline(
        0.1,
        color="#d62728",
        linestyle=":",
        label="RB-mean warning threshold",
    )
    axes[0].set_ylabel("Mahalanobis ratio")
    axes[0].set_title("Full-state materialization check")
    axes[0].legend()

    axes[1].plot(epochs, unique_min, marker="o", label="minimum over time")
    axes[1].plot(epochs, unique_mean, marker="o", label="mean over time")
    axes[1].plot(epochs, unique_final, marker="o", label="terminal")
    axes[1].set_ylabel("Unique smoother particle indices")
    axes[1].set_title("FFBSi path diversity")
    axes[1].legend()

    for axis in axes:
        axis.set_xlabel("EM epoch")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_em_transition_decomposition(diagnostics_history, save_path):
    """Plot transition normalization and quadratic-penalty contributions."""
    if not diagnostics_history:
        return None

    epochs = np.asarray([entry["epoch"] + 1 for entry in diagnostics_history])
    density = [entry["m_step_density"] for entry in diagnostics_history]
    normalization = np.asarray(
        [entry["transition_normalization_sum"] for entry in density]
    )
    penalty = np.asarray([entry["transition_quadratic_penalty"] for entry in density])
    transition = normalization + penalty
    ratio = np.asarray([entry["transition_mahalanobis_ratio"] for entry in density])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, normalization, marker="o", label="normalization")
    axes[0].plot(epochs, penalty, marker="o", label="quadratic penalty")
    axes[0].plot(epochs, transition, marker="o", label="transition total")
    axes[0].set_ylabel("Summed log-density contribution")
    axes[0].set_title("Transition-density decomposition")
    axes[0].legend()

    axes[1].plot(epochs, ratio, marker="o", color="#d62728")
    axes[1].axhline(
        1.0, color="black", linestyle="--", label="unconditional reference"
    )
    axes[1].set_ylabel("Mean Mahalanobis / dimension")
    axes[1].set_title("M-step transition residual fit")
    axes[1].legend()

    for axis in axes:
        axis.set_xlabel("EM epoch")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_em_parameter_history(params_history, save_path):
    """Plot scalar and identified diagonal-B parameter trajectories."""
    if not params_history:
        return None

    parameter_index = np.arange(len(params_history))
    kappa = np.asarray([float(params.kappa) for params in params_history])
    alpha = np.asarray([float(params.alpha) for params in params_history])
    beta = np.asarray([float(params.beta) for params in params_history])
    B_attack = np.asarray([float(params.B[0, 0]) for params in params_history])
    B_defence = np.asarray([float(params.B[1, 1]) for params in params_history])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(parameter_index, kappa, marker="o")
    axes[0].set_title("OU kappa")
    axes[0].set_ylabel("kappa")

    axes[1].plot(parameter_index, alpha, marker="o", label="alpha")
    axes[1].plot(parameter_index, beta, marker="o", label="beta")
    axes[1].set_title("Observation parameters")
    axes[1].legend()

    axes[2].plot(parameter_index, B_attack, marker="o", label="attack")
    axes[2].plot(parameter_index, B_defence, marker="o", label="defence")
    axes[2].axhline(1.0, color="black", linestyle="--", alpha=0.6)
    axes[2].set_title("Diagonal B (determinant one)")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("Parameter index (theta_k)")
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_loss_components(
    loss_traces,
    save_path=os.path.join(OUTPUT_DIR, "em_loss_components_curve.png"),
):
    """Plot the per-gradient-step loss components of each EM epoch.

    Each epoch's M-step runs ``n_gradient_steps`` gradient updates; this plots
    the complete-data loss (and its init / transition / observation terms) as a
    function of the gradient step, with one line per epoch so the epochs can be
    told apart.

    Args:
        loss_traces: array of shape ``(n_epochs, n_gradient_steps, 5)`` where
            the last axis is ``[total, init, transition, observation, prior]``.
        save_path: if given, save the figure to this path.

    Returns:
        ``(figure, axis)``.
    """
    loss_traces = np.asarray(loss_traces)
    n_epochs, n_gradient_steps, _ = loss_traces.shape
    steps = np.arange(1, n_gradient_steps + 1)

    component_names = ["total", "init", "transition", "observation", "prior"]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    n_components = len(component_names)
    n_cols = 3
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes = axes.ravel()

    for ax, name, color in zip(axes, component_names, colors):
        for epoch in range(n_epochs):
            ax.plot(
                steps,
                loss_traces[epoch, :, component_names.index(name)],
                marker="o",
                linewidth=1.5,
                color=color,
                alpha=0.55 + 0.45 * (epoch / max(n_epochs - 1, 1)),
                label=f"Epoch {epoch + 1}",
            )
        ax.set_title(f"{name} loss")
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # hide any unused subplots
    for ax in axes[n_components:]:
        ax.set_visible(False)

    fig.suptitle("EM M-step loss components per gradient step")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {os.path.abspath(save_path)}")

    return fig, axes


def plot_all(filtered_states, augmented_results, team_id_to_name, top_n, save_path,
             timestamps=None):
    """Generate all plots and save them to the outputs/graphic directory.

    Args:
        timestamps: optional x-axis values (e.g. match dates) for the
            ``plot_timeseries_states`` plot. If None, falls back to the
            filter's numeric timestamps.
    """
    os.makedirs(save_path, exist_ok=True)
    abs_save_path = os.path.abspath(save_path)
    print(f"Output directory (absolute): {abs_save_path}")
    num_teams = len(team_id_to_name)
    plot_top_strengths(filtered_states, team_id_to_name,
                       top_n=top_n,
                       save_path=os.path.join(save_path, "top_strengths.png"))
    plot_timeseries_states(filtered_states, team_id_to_name,
                           top_n=top_n,
                           timestamps=timestamps,
                           save_path=os.path.join(save_path, "timeseries_states.png"))
    plot_correlation_matrix(augmented_results, team_id_to_name, num_teams=num_teams,
                            save_path=os.path.join(save_path, "correlation_matrix.png"))
    plot_log_normalizing_constant(filtered_states,
                                  save_path=os.path.join(save_path, "log_normalizing_constant.png"))


# ---------------------------------------------------------------------------
# Smoothed trajectory plots (from plot_smoothing.py)
# ---------------------------------------------------------------------------
def plot_smoothed_trajectories(
    smoothed_trajectories,
    team_id_to_name,
    df,
    top_n=8,
    n_sample_trajs=5,
    save_path=os.path.join(OUTPUT_DIR, "smoothed_trajectories.png"),
):
    """Plot smoothed attack/defense trajectories with uncertainty bands.

    Args:
        smoothed_trajectories: array of shape (N_traj, T+1, M, 2) from
            rbpf_backward_smoothing.
        team_id_to_name: dict mapping team index -> team name.
        df: pandas DataFrame from get_results with a 'date' column.
        top_n: number of top teams to plot (ranked by final smoothed attack).
        n_sample_trajs: number of individual trajectories to overlay.
        save_path: where to save the figure.
    """
    trajs = np.asarray(smoothed_trajectories)  # (N, T+1, M, 2)
    n_traj, n_steps, n_teams, _ = trajs.shape

    # Smoothed mean and std across trajectories
    smooth_mean = trajs.mean(axis=0)   # (T+1, M, 2)
    smooth_std = trajs.std(axis=0)     # (T+1, M, 2)

    # Rank teams by final smoothed attack strength
    final_attack = smooth_mean[-1, :, 0]
    top_indices = np.argsort(final_attack)[-top_n:][::-1]

    # Build x-axis dates: df has one row per unique date (T rows).
    # The smoothed trajectory has T+1 steps (prepended initial state at t=0).
    dates = df["date"].to_numpy()
    if len(dates) == n_steps - 1:
        dates = np.concatenate([dates[:1], dates])
    elif len(dates) != n_steps:
        dates = np.arange(n_steps)

    is_datetime = dates.dtype.kind in {"M", "O"}
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, top_n))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    if is_datetime:
        import matplotlib.dates as mdates
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

    for team_idx, color in zip(top_indices, colors):
        name = team_id_to_name.get(int(team_idx), str(team_idx))

        for dim, ax, label in [(0, axes[0], "Attack"), (1, axes[1], "Defense")]:
            mean = smooth_mean[:, team_idx, dim]
            std = smooth_std[:, team_idx, dim]

            # Uncertainty band: mean ± 1 std
            ax.fill_between(dates, mean - std, mean + std,
                            color=color, alpha=0.15)
            # Mean line
            ax.plot(dates, mean, color=color, linewidth=2.0, label=name)

            # Overlay a few sample trajectories (thin, transparent)
            for i in range(min(n_sample_trajs, n_traj)):
                ax.plot(dates, trajs[i, :, team_idx, dim],
                        color=color, linewidth=0.5, alpha=0.25)

    axes[0].set_title(f"Top {top_n} Teams — Smoothed Attack Trajectories "
                      f"({n_traj} trajectories, ±1 std band)")
    axes[0].set_ylabel("Attack state")
    axes[1].set_title(f"Top {top_n} Teams — Smoothed Defense Trajectories")
    axes[1].set_ylabel("Defense state")
    axes[1].set_xlabel("Date" if is_datetime else "Time step")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {os.path.abspath(save_path)}")
    plt.close(fig)
    return fig


def plot_smoothed_uncertainty(
    smoothed_trajectories,
    team_id_to_name,
    df,
    top_n=8,
    save_path=os.path.join(OUTPUT_DIR, "smoothed_uncertainty.png"),
):
    """Plot the smoothed uncertainty (std across trajectories) over time.

    This shows how the posterior uncertainty evolves — it should decrease
    during periods with many matches and increase during gaps.
    """
    trajs = np.asarray(smoothed_trajectories)
    n_traj, n_steps, n_teams, _ = trajs.shape
    smooth_std = trajs.std(axis=0)  # (T+1, M, 2)

    final_attack = trajs[:, -1, :, 0].mean(axis=0)
    top_indices = np.argsort(final_attack)[-top_n:][::-1]

    dates = df["date"].to_numpy()
    if len(dates) == n_steps - 1:
        dates = np.concatenate([dates[:1], dates])
    elif len(dates) != n_steps:
        dates = np.arange(n_steps)

    is_datetime = dates.dtype.kind in {"M", "O"}
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, top_n))

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    if is_datetime:
        import matplotlib.dates as mdates
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

    for team_idx, color in zip(top_indices, colors):
        name = team_id_to_name.get(int(team_idx), str(team_idx))
        axes[0].plot(dates, smooth_std[:, team_idx, 0],
                     color=color, linewidth=1.5, label=name)
        axes[1].plot(dates, smooth_std[:, team_idx, 1],
                     color=color, linewidth=1.5, label=name)

    axes[0].set_title(f"Top {top_n} Teams — Smoothed Attack Uncertainty (std across {n_traj} trajectories)")
    axes[0].set_ylabel("Attack std")
    axes[1].set_title(f"Top {top_n} Teams — Smoothed Defense Uncertainty")
    axes[1].set_ylabel("Defense std")
    axes[1].set_xlabel("Date" if is_datetime else "Time step")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {os.path.abspath(save_path)}")
    plt.close(fig)
    return fig


def plot_all_smoothing(smoothed_trajectories, team_id_to_name, df, top_n, save_path):
    """Generate all smoothing plots and save them to ``save_path``."""
    os.makedirs(save_path, exist_ok=True)
    plot_smoothed_trajectories(
        smoothed_trajectories=smoothed_trajectories,
        team_id_to_name=team_id_to_name,
        df=df,
        top_n=top_n,
        save_path=os.path.join(save_path, "smoothed_trajectories.png"),
    )
    plot_smoothed_uncertainty(
        smoothed_trajectories=smoothed_trajectories,
        team_id_to_name=team_id_to_name,
        df=df,
        top_n=top_n,
        save_path=os.path.join(save_path, "smoothed_uncertainty.png"),
    )


def plot_em_results(
    results: dict,
    output_dir: str,
):
    """Plot the run_EM diagnostics and save them to ``output_dir``.

    ``results`` is the dict returned by ``run_EM``. Produces:
      - ``em_log_marginal_likelihood_curve.png``: per-epoch log marginal.
      - ``em_mstep_objective_curve.png``: per-epoch M-step objective
        (initial vs candidate vs final).
      - ``em_mstep_log_density_terms.png``: per-epoch init / transition /
        observation complete-data log-density terms.
      - ``em_gamma_diagnostics.png``: covariance eigenvalue and conditioning.
      - ``em_path_diagnostics.png``: materialization and smoother diversity.
      - ``em_transition_decomposition.png``: normalization versus quadratic
        transition contributions.
      - ``em_parameter_history.png``: kappa, observation and diagonal-B paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_marginal_history = results["log_marginal_history"]
    plot_log_marginal_likelihood_curve(
        log_marginal_history,
        save_path=os.path.join(output_dir, "em_log_marginal_likelihood_curve.png"),
    )

    mstep_history = results["mstep_history"]
    epochs = np.arange(1, len(mstep_history) + 1)

    # --- M-step objective: initial vs candidate vs final ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [e["initial_objective"] for e in mstep_history],
            marker="o", label="initial", color="#1f77b4")
    ax.plot(epochs, [e["candidate_objective"] for e in mstep_history],
            marker="o", label="candidate", color="#ff7f0e")
    ax.plot(epochs, [e["final_objective"] for e in mstep_history],
            marker="o", label="final", color="#2ca02c")
    ax.set_xlabel("EM Epoch")
    ax.set_ylabel("M-step objective (negative log joint density)")
    ax.set_title("EM M-step Objective per Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "em_mstep_objective_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- M-step complete-data log-density terms ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [e["initial_log_density"] for e in mstep_history],
            marker="o", label="init", color="#2ca02c")
    ax.plot(epochs, [e["transition_log_density"] for e in mstep_history],
            marker="o", label="transition", color="#d62728")
    ax.plot(epochs, [e["observation_log_density"] for e in mstep_history],
            marker="o", label="observation", color="#9467bd")
    if all("prior_log_density" in e for e in mstep_history):
        ax.plot(epochs, [e["prior_log_density"] for e in mstep_history],
                marker="o", label="prior", color="#ff9896", linestyle="--")
    ax.set_xlabel("EM Epoch")
    ax.set_ylabel("Complete-data log density")
    ax.set_title("EM M-step Log-Density Terms per Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "em_mstep_log_density_terms.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    diagnostics_history = results.get("diagnostics_history", [])
    if diagnostics_history:
        plot_em_covariance_diagnostics(
            diagnostics_history,
            os.path.join(output_dir, "em_gamma_diagnostics.png"),
        )
        plot_em_path_diagnostics(
            diagnostics_history,
            os.path.join(output_dir, "em_path_diagnostics.png"),
        )
        plot_em_transition_decomposition(
            diagnostics_history,
            os.path.join(output_dir, "em_transition_decomposition.png"),
        )

    plot_em_parameter_history(
        results.get("params_history", []),
        os.path.join(output_dir, "em_parameter_history.png"),
    )

    print(f"Saved EM plots to {os.path.abspath(output_dir)}")
