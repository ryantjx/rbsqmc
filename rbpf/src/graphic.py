"""Visualization helpers for RBPF filter and EM results.

Functions:
  - plot_top_strengths: bar chart of top-N attack/defense strengths
  - plot_top_filter_states: attack and defense filter trajectories for top teams
  - plot_correlation_matrix: heatmap of between-team correlation at final state
  - plot_log_normalizing_constant: line plot of filter log marginal likelihood
  - plot_em_convergence: line plot of EM log marginal likelihood across epochs
"""

import os

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "graphic")


def plot_top_strengths(filtered_states, team_id_to_name, top_n=10,
                       save_path=os.path.join(OUTPUT_DIR, "top_strengths.png")):
    """Bar chart of top-N attack, defense, and total strengths at final timestep.

    Args:
        filtered_states: cuthbert FilterStates with particles.x shape (T+1, N, M, 2)
        team_id_to_name: dict mapping team_id -> team name
        top_n: number of teams to show
        save_path: if given, save figure to this path
    """
    x_final = np.array(filtered_states.particles.x[-1])  # (N, M, 2)
    mean_strengths = x_final.mean(axis=0)  # (M, 2)

    attack = mean_strengths[:, 0]
    defense = mean_strengths[:, 1]
    total = attack + defense

    top_attack_idx = np.argsort(attack)[-top_n:][::-1]
    top_defense_idx = np.argsort(defense)[-top_n:][::-1]
    top_total_idx = np.argsort(total)[-top_n:][::-1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    ax1.barh(
        [team_id_to_name[i] for i in top_attack_idx],
        attack[top_attack_idx],
        color="steelblue",
    )
    ax1.set_xlabel("Attack Strength")
    ax1.set_title(f"Top {top_n} Attack Strengths")
    ax1.invert_yaxis()

    ax2.barh(
        [team_id_to_name[i] for i in top_defense_idx],
        defense[top_defense_idx],
        color="firebrick",
    )
    ax2.set_xlabel("Defense Strength")
    ax2.set_title(f"Top {top_n} Defense Strengths")
    ax2.invert_yaxis()

    ax3.barh(
        [team_id_to_name[i] for i in top_total_idx],
        total[top_total_idx],
        color="darkgreen",
    )
    ax3.set_xlabel("Total Strength (Attack + Defense)")
    ax3.set_title(f"Top {top_n} Total Strengths")
    ax3.invert_yaxis()

    # plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()


def plot_top_filter_states(
    filtered_states,
    team_id_to_name,
    top_n=5,
    rank_by="attack",
    timestamps=None,
    save_path=os.path.join(OUTPUT_DIR, "top_filter_states.png"),
):
    """Plot attack and defense filtered states over time for the top teams.

    Teams are selected using their final filtered posterior mean.  Particle
    log weights are used when they are available; otherwise the particle mean
    is used.  The selected teams are then shown in both the attack and defense
    panels so their two state components can be compared directly.

    Args:
        filtered_states: cuthbert FilterStates with particles.x shaped
            ``(T, N, M, 2)`` (or ``(T, N, M, K)`` with the first two
            components representing attack and defense).
        team_id_to_name: dict mapping team index -> team name.
        top_n: number of teams to plot.
        rank_by: final state component used to select teams: ``"attack"``,
            ``"defense"``, or ``"total"`` (attack + defense).
        timestamps: optional x-axis values. If omitted, timestamps from
            ``filtered_states.model_inputs`` are used when they align with
            the state history; otherwise integer time steps are used.
        save_path: if given, save the figure to this path.

    Returns:
        ``(figure, (attack_axis, defense_axis))``.
    """
    x_history = np.asarray(filtered_states.particles.x)
    if x_history.ndim != 4 or x_history.shape[-1] < 2:
        raise ValueError(
            "filtered_states.particles.x must have shape (T, N, M, K) "
            "with K >= 2"
        )

    n_steps, n_particles, n_teams, _ = x_history.shape
    if n_steps == 0 or n_teams == 0:
        raise ValueError("filtered_states must contain at least one state and team")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    top_n = min(int(top_n), n_teams)
    if rank_by not in {"attack", "defense", "total"}:
        raise ValueError("rank_by must be 'attack', 'defense', or 'total'")

    # Compute the filtered posterior mean at every time step.  The fallback
    # keeps this helper usable with lightweight state objects that only expose
    # particles.x (and matches the convention used by plot_top_strengths).
    log_weights = np.asarray(getattr(filtered_states, "log_weights", None))
    if log_weights.shape == (n_steps, n_particles):
        finite_log_weights = np.where(np.isfinite(log_weights), log_weights, -np.inf)
        max_log_weight = np.max(finite_log_weights, axis=1, keepdims=True)
        shifted_weights = np.exp(log_weights - max_log_weight)
        shifted_weights[~np.isfinite(shifted_weights)] = 0.0
        weight_sum = shifted_weights.sum(axis=1, keepdims=True)
        uniform_weights = np.full_like(shifted_weights, 1.0 / n_particles)
        weights = np.divide(
            shifted_weights,
            weight_sum,
            out=uniform_weights,
            where=weight_sum > 0,
        )
        filter_means = np.sum(
            x_history[..., :2] * weights[:, :, None, None], axis=1
        )
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
        timestamps = np.asarray(timestamps)
        if timestamps.size != n_steps:
            timestamps = np.arange(n_steps)
        else:
            timestamps = timestamps.reshape(-1)

    names = [team_id_to_name.get(int(i), str(i)) for i in top_indices]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, top_n))
    fig, (attack_ax, defense_ax) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
    )

    for team_idx, name, color in zip(top_indices, names, colors):
        attack_ax.plot(
            timestamps, filter_means[:, team_idx, 0],
            color=color, linewidth=1.8, label=name,
        )
        defense_ax.plot(
            timestamps, filter_means[:, team_idx, 1],
            color=color, linewidth=1.8, label=name,
        )

    rank_label = {
        "attack": "attack",
        "defense": "defense",
        "total": "attack + defense",
    }[rank_by]
    attack_ax.set_title(f"Top {top_n} Teams by Final Filtered {rank_label}")
    attack_ax.set_ylabel("Attack state")
    defense_ax.set_ylabel("Defense state")
    defense_ax.set_xlabel("Time")
    for axis in (attack_ax, defense_ax):
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    fig.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, (attack_ax, defense_ax)


def plot_correlation_matrix(augmented_results, team_id_to_name, save_path=os.path.join(OUTPUT_DIR, "correlation_matrix.png")):
    """Heatmap of between-team correlation matrix at final timestep.

    Args:
        augmented_results: RBPFFootballResults with gamma_t shape (T+1, M, M)
        team_id_to_name: dict mapping team_id -> team name
        save_path: if given, save figure to this path
    """
    gamma_final = np.array(augmented_results.gamma_t[-1])  # (M, M)

    std = np.sqrt(np.diag(gamma_final))
    std_safe = np.where(std > 1e-10, std, 1.0)
    corr = gamma_final / np.outer(std_safe, std_safe)
    corr = np.clip(corr, -1, 1)

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
    # plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()


def plot_correlation_extremes(augmented_results, team_id_to_name, top_n=5,
                              save_path=os.path.join(OUTPUT_DIR, "correlation_extremes.png")):
    """Bar chart of top-N and bottom-N team pair correlations at final timestep.

    Args:
        augmented_results: RBPFFootballResults with gamma_t shape (T+1, M, M)
        team_id_to_name: dict mapping team_id -> team name
        top_n: number of pairs to show for highest and lowest correlations
        save_path: if given, save figure to this path
    """
    gamma_final = np.array(augmented_results.gamma_t[-1])  # (M, M)

    std = np.sqrt(np.diag(gamma_final))
    std_safe = np.where(std > 1e-10, std, 1.0)
    corr = gamma_final / np.outer(std_safe, std_safe)
    corr = np.clip(corr, -1, 1)

    active = std > 1e-10
    active_idx = np.where(active)[0]

    # Collect all unique pairs (i < j)
    pairs = []
    for ii in range(len(active_idx)):
        for jj in range(ii + 1, len(active_idx)):
            i = active_idx[ii]
            j = active_idx[jj]
            pairs.append((corr[i, j], i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)
    top_pairs = pairs[:top_n]
    bottom_pairs = pairs[-top_n:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Top correlations
    labels = [f"{team_id_to_name[i]} / {team_id_to_name[j]}" for _, i, j in top_pairs]
    vals = [c for c, _, _ in top_pairs]
    ax1.barh(labels, vals, color="darkred")
    ax1.set_xlabel("Correlation")
    ax1.set_title(f"Top {top_n} Team Correlations")
    ax1.invert_yaxis()

    # Bottom correlations
    labels = [f"{team_id_to_name[i]} / {team_id_to_name[j]}" for _, i, j in bottom_pairs]
    vals = [c for c, _, _ in bottom_pairs]
    ax2.barh(labels, vals, color="navy")
    ax2.set_xlabel("Correlation")
    ax2.set_title(f"Bottom {top_n} Team Correlations")
    ax2.invert_yaxis()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()


def plot_log_normalizing_constant(filtered_states, save_path=os.path.join(OUTPUT_DIR, "log_normalizing_constant.png")):
    """Line plot of the log normalizing constant over time.

    Args:
        filtered_states: cuthbert FilterStates with log_normalizing_constant shape (T+1,)
        save_path: if given, save figure to this path
    """
    log_z = np.array(filtered_states.log_normalizing_constant)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(log_z, color="darkgreen", linewidth=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Log Normalizing Constant")
    ax.set_title("Filter Log Marginal Likelihood")
    ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()


def plot_em_convergence(log_marginal_history, save_path=os.path.join(OUTPUT_DIR, "em_convergence.png")):
    """Line plot of EM log marginal likelihood across epochs.

    Args:
        log_marginal_history: list or array of log marginal values per EM epoch
        save_path: if given, save figure to this path
    """
    history = np.array(log_marginal_history)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history, "o-", color="darkblue", linewidth=1.5, markersize=5)
    ax.set_xlabel("EM Epoch")
    ax.set_ylabel("Log Marginal Likelihood")
    ax.set_title("EM Convergence")
    ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.show()


def plot_log_likelihood_history(
    log_likelihood_history: list[float],
    output_path: str,
) -> None:
    """Save EM log-likelihood values and epoch-to-epoch changes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(log_likelihood_history, dtype=float)
    epochs = np.arange(1, values.size + 1)

    fig, (ax_value, ax_change) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    if values.size:
        ax_value.plot(
            epochs, values, marker="o", linewidth=2,
            color="tab:blue", label="Log marginal likelihood",
        )
        ax_value.set_ylabel("Log marginal likelihood")
        ax_value.grid(True, alpha=0.3)
        ax_value.legend(loc="best")

    if values.size > 1:
        changes = np.diff(values)
        change_epochs = epochs[1:]
        bars = ax_change.bar(
            change_epochs, changes, color="tab:green", alpha=0.8,
            label="Change from previous epoch",
        )
        for bar, change in zip(bars, changes):
            ax_change.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{change:.2f}",
                ha="center",
                va="bottom" if change >= 0 else "top",
                fontsize=8,
            )
        ax_change.axhline(0.0, color="black", linewidth=0.8)
        ax_change.legend(loc="best")
    elif values.size == 1:
        ax_change.text(
            0.5, 0.5, "No previous epoch for comparison",
            ha="center", va="center", transform=ax_change.transAxes,
        )

    ax_change.set_xlabel("EM epoch")
    ax_change.set_ylabel("Δ log L")
    ax_change.grid(True, alpha=0.3)
    ax_change.set_xticks(epochs)
    fig.suptitle("EM log-likelihood by epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_all(filtered_states, augmented_results, team_id_to_name, top_n, save_path):
    """Generate all plots and save them to the outputs/graphic directory."""
    os.makedirs(save_path, exist_ok=True)
    plot_top_strengths(filtered_states, team_id_to_name,
                       top_n=top_n,
                       save_path=os.path.join(save_path, "top_strengths.png"))
    plot_top_filter_states(filtered_states, team_id_to_name,
                           top_n=top_n,
                           save_path=os.path.join(save_path, "top_filter_states.png"))
    plot_correlation_matrix(augmented_results, team_id_to_name,
                            save_path=os.path.join(save_path, "correlation_matrix.png"))
    plot_log_normalizing_constant(filtered_states,
                                  save_path=os.path.join(save_path, "log_normalizing_constant.png"))
    plot_correlation_extremes(augmented_results, team_id_to_name,
                              top_n=top_n,
                              save_path=os.path.join(save_path, "correlation_extremes.png"))