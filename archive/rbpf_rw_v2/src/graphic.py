"""Visualization helpers for RBPF filter and EM results.

Functions:
  - plot_top_strengths: bar chart of top-N attack/defense strengths
  - plot_top_filter_states: attack and defense filter trajectories for top teams
  - plot_correlation_matrix: heatmap of between-team correlation at final state
  - plot_correlation_extremes: bar chart of top/bottom team-pair correlations
  - plot_log_normalizing_constant: line plot of filter log marginal likelihood
  - plot_em_convergence: line plot of EM log marginal likelihood across epochs
  - plot_log_likelihood_history: EM log-likelihood values and epoch changes
  - plot_mstep_diagnostics: M-step objective per epoch (start vs best)
  - plot_em_diagnostics: E-step score + M-step optimization score (full traces)
  - plot_gd_performance: GD loss / log-marginal / per-param traces (v2)
  - plot_all: generate all plots and save them
"""

import os

import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphic")


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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_filter_states(
    filtered_states,
    team_id_to_name,
    top_n=5,
    rank_by="attack",
    timestamps=None,
    save_path=os.path.join(OUTPUT_DIR, "top_filter_states.png"),
):
    """Plot attack and defense filtered states over time for the top teams.

    Teams are selected using their final filtered posterior mean. Particle
    log weights are used when they are available; otherwise the particle mean
    is used. The selected teams are then shown in both the attack and defense
    panels so their two state components can be compared directly.

    Args:
        filtered_states: cuthbert FilterStates with particles.x shaped
            ``(T, N, M, 2)`` (or ``(T, N, M, K)`` with the first two
            components representing attack and defense).
        team_id_to_name: dict mapping team index -> team name.
        top_n: number of teams to plot.
        rank_by: final state component used to select teams: ``"attack"``,
            ``"defense"``, or ``"total"`` (attack + defense).
        timestamps: optional x-axis values.
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


def _team_correlation_from_gamma(gamma_final):
    """Normalize an (M, M) team covariance to a correlation matrix.

    With the Kronecker structure, ``gamma_t`` is already the ``M x M`` team
    covariance (the attack/defence factor ``B`` is shared and does not affect
    between-team correlation). We normalize its diagonal to a correlation.
    """
    std = np.sqrt(np.diag(gamma_final))
    std_safe = np.where(std > 1e-10, std, 1.0)
    corr = gamma_final / np.outer(std_safe, std_safe)
    corr = np.clip(corr, -1, 1)
    return corr, std


def plot_correlation_matrix(augmented_results, team_id_to_name, num_teams=None,
                            save_path=os.path.join(OUTPUT_DIR, "correlation_matrix.png")):
    """Heatmap of between-team correlation matrix at final timestep.

    Args:
        augmented_results: RBPFFootballResults with gamma_t shape (T+1, M, M)
        team_id_to_name: dict mapping team_id -> team name
        num_teams: number of teams (inferred from team_id_to_name if None)
        save_path: if given, save figure to this path
    """
    if num_teams is None:
        num_teams = len(team_id_to_name)
    gamma_final = np.array(augmented_results.gamma_t[-1])  # (M, M)
    corr, std = _team_correlation_from_gamma(gamma_final)

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
    plt.close(fig)


def plot_correlation_extremes(augmented_results, team_id_to_name, top_n=5,
                              num_teams=None,
                              save_path=os.path.join(OUTPUT_DIR, "correlation_extremes.png")):
    """Bar chart of top-N and bottom-N team pair correlations at final timestep.

    Args:
        augmented_results: RBPFFootballResults with gamma_t shape (T+1, M, M)
        team_id_to_name: dict mapping team_id -> team name
        top_n: number of pairs to show for highest and lowest correlations
        num_teams: number of teams (inferred from team_id_to_name if None)
        save_path: if given, save figure to this path
    """
    if num_teams is None:
        num_teams = len(team_id_to_name)
    gamma_final = np.array(augmented_results.gamma_t[-1])  # (M, M)
    corr, std = _team_correlation_from_gamma(gamma_final)

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

    labels = [f"{team_id_to_name[i]} / {team_id_to_name[j]}" for _, i, j in top_pairs]
    vals = [c for c, _, _ in top_pairs]
    ax1.barh(labels, vals, color="darkred")
    ax1.set_xlabel("Correlation")
    ax1.set_title(f"Top {top_n} Team Correlations")
    ax1.invert_yaxis()

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
    plt.close(fig)


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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def plot_mstep_diagnostics(
    loss_start: list[float],
    loss_end: list[float],
    log_marginal_history: list[float] | None = None,
    output_path: str = os.path.join(OUTPUT_DIR, "mstep_diagnostics.png"),
) -> None:
    """Plot the M-step objective at the start and end of each epoch.

    The M-step minimizes ``loss = -log L`` (the negative log-likelihood of the
    smoothed states under the current parameters). For each EM epoch this shows
    the objective at the *start* of the M-step and at the *best point reached*
    during that epoch's gradient loop, plus the within-epoch improvement.

    Args:
        loss_start: M-step objective evaluated at the start of each epoch.
        loss_end: best M-step objective reached during each epoch.
        log_marginal_history: optional per-epoch E-step log marginal likelihood
            (drawn on a twin axis, since it is a different scale).
        output_path: where to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(loss_start) + 1)
    loss_start = np.asarray(loss_start, dtype=float)
    loss_end = np.asarray(loss_end, dtype=float)
    changes = loss_end - loss_start  # negative => M-step improved the objective

    fig, ax_value = plt.subplots(figsize=(9, 6))
    ax_value.plot(epochs, loss_start, "o--", color="tab:red",
                  label="M-step loss (start)")
    ax_value.plot(epochs, loss_end, "o-", color="tab:blue",
                  label="M-step loss (best)")
    ax_value.set_xlabel("EM epoch")
    ax_value.set_ylabel("M-step objective (-log L)")
    ax_value.grid(True, alpha=0.3)
    ax_value.legend(loc="best")

    if log_marginal_history is not None and len(log_marginal_history) == len(epochs):
        ax_log = ax_value.twinx()
        ax_log.plot(epochs, np.asarray(log_marginal_history), "s-", color="tab:green",
                    linewidth=1.2, markersize=4, label="E-step log marginal L")
        ax_log.set_ylabel("Log marginal likelihood", color="tab:green")
        ax_log.tick_params(axis="y", labelcolor="tab:green")
        ax_log.legend(loc="lower left")

    for ep, chg in zip(epochs, changes):
        ax_value.annotate(
            f"{chg:.0f}", (ep, loss_end[ep - 1]),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=8,
        )

    ax_value.set_title("M-step objective per epoch (start vs best)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_em_diagnostics(
    log_marginal_history: list[float],
    mstep_loss_start: list[float],
    mstep_loss_end: list[float],
    mstep_loss_trace: list[list[float]],
    output_path: str = os.path.join(OUTPUT_DIR, "em_diagnostics.png"),
) -> None:
    """Comprehensive EM diagnostics: E-step score + M-step optimization score.

    Two stacked panels:

    1. **E-step score** (top): the log marginal likelihood ``log p(y | theta)``
       evaluated by the particle filter at each EM epoch. This is the quantity
       EM is supposed to monotonically increase (up to MC noise). A rising
       trend confirms the E-step is improving the model fit.

    2. **M-step optimization score** (bottom): the full per-gradient-step loss
       trajectory ``-log L`` for every epoch, overlaid. Each epoch's M-step
       should descend from its start value to a lower best value; the gap
       between consecutive epochs' start values shows the E-step's contribution.

    Args:
        log_marginal_history: per-epoch E-step log marginal likelihood.
        mstep_loss_start: M-step objective at the start of each epoch.
        mstep_loss_end: best M-step objective reached in each epoch.
        mstep_loss_trace: full per-gradient-step loss trajectory per epoch.
        output_path: where to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(log_marginal_history) + 1)
    lm = np.asarray(log_marginal_history, dtype=float)
    loss_start = np.asarray(mstep_loss_start, dtype=float)
    loss_end = np.asarray(mstep_loss_end, dtype=float)

    fig, (ax_estep, ax_mstep) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4]},
    )

    # --- Panel 1: E-step score (log marginal likelihood) ---
    ax_estep.plot(epochs, lm, "o-", color="tab:blue", linewidth=1.8, markersize=6,
                  label="E-step log marginal L")
    ax_estep.set_ylabel("E-step score\n(log marginal likelihood)")
    ax_estep.grid(True, alpha=0.3)
    ax_estep.legend(loc="best")
    ax_estep.set_title("EM diagnostics: E-step score and M-step optimization")

    # --- Panel 2: M-step optimization score (full gradient traces) ---
    cmap = plt.get_cmap("viridis")
    n_epochs = len(mstep_loss_trace)
    for i, trace in enumerate(mstep_loss_trace):
        trace = np.asarray(trace, dtype=float)
        color = cmap(i / max(n_epochs - 1, 1))
        ax_mstep.plot(trace, "-", color=color, linewidth=1.0,
                      label=f"epoch {i + 1}" if i in (0, n_epochs - 1) else None)
    # Mark start/end of each epoch's M-step.
    ax_mstep.plot(epochs - 1, loss_start, "o", color="tab:red", markersize=5,
                  label="M-step loss (start)")
    ax_mstep.plot(epochs - 1, loss_end, "s", color="tab:green", markersize=5,
                  label="M-step loss (best)")
    ax_mstep.set_xlabel("M-step gradient step (epochs overlaid)")
    ax_mstep.set_ylabel("M-step objective (-log L)")
    ax_mstep.grid(True, alpha=0.3)
    ax_mstep.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gd_performance(
    log_marginal_history: list[float],
    loss_history: list[float] | None = None,
    param_history: dict[str, list[float]] | None = None,
    output_path: str = os.path.join(OUTPUT_DIR, "gd_performance.png"),
) -> None:
    """Plot the direct-GD training performance (v2).

    Three stacked panels:

    1. **Log marginal likelihood** (top): ``log Z(theta)`` per gradient step.
       A rising trend is the key success signal (v1's EM diverged; v2's GD
       should improve the fit).
    2. **Loss** (middle): ``-log Z(theta)`` per step (the objective being
       minimized). Should fall.
    3. **Parameters** (bottom): per-parameter traces (e.g. ``kappa``,
       ``alpha``, ``beta``) if ``param_history`` is provided.

    Args:
        log_marginal_history: ``log Z(theta)`` per step.
        loss_history: optional ``-log Z(theta)`` per step (drawn on a twin axis
            of the top panel if provided).
        param_history: optional dict of param-name -> per-step values.
        output_path: where to save the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lm = np.asarray(log_marginal_history, dtype=float)
    steps = np.arange(1, lm.size + 1)

    n_panels = 3 if param_history else (2 if loss_history is not None else 1)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True)

    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: log marginal likelihood ---
    ax = axes[0]
    ax.plot(steps, lm, "-", color="tab:blue", linewidth=1.5, label="log Z(θ)")
    if loss_history is not None:
        ax2 = ax.twinx()
        ax2.plot(steps, np.asarray(loss_history, dtype=float), "-",
                 color="tab:red", linewidth=1.0, alpha=0.7, label="-log Z(θ)")
        ax2.set_ylabel("-log Z(θ)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_ylabel("log Z(θ)")
    ax.set_title("Direct GD training performance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # --- Panel 2: loss (if no param history) ---
    if n_panels >= 2 and param_history is None:
        ax = axes[1]
        ax.plot(steps, np.asarray(loss_history, dtype=float), "-",
                color="tab:red", linewidth=1.5, label="-log Z(θ)")
        ax.set_ylabel("-log Z(θ)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # --- Panel 3 (or 2): parameters ---
    if param_history:
        ax = axes[-1]
        for name, values in param_history.items():
            ax.plot(steps, np.asarray(values, dtype=float), "-",
                    linewidth=1.2, label=name)
        ax.set_ylabel("Parameter value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Gradient step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(filtered_states, augmented_results, team_id_to_name, top_n, save_path):
    """Generate all plots and save them to the outputs/graphic directory."""
    os.makedirs(save_path, exist_ok=True)
    num_teams = len(team_id_to_name)
    plot_top_strengths(filtered_states, team_id_to_name,
                       top_n=top_n,
                       save_path=os.path.join(save_path, "top_strengths.png"))
    plot_top_filter_states(filtered_states, team_id_to_name,
                           top_n=top_n,
                           save_path=os.path.join(save_path, "top_filter_states.png"))
    plot_correlation_matrix(augmented_results, team_id_to_name, num_teams=num_teams,
                            save_path=os.path.join(save_path, "correlation_matrix.png"))
    plot_log_normalizing_constant(filtered_states,
                                  save_path=os.path.join(save_path, "log_normalizing_constant.png"))
    plot_correlation_extremes(augmented_results, team_id_to_name,
                              top_n=top_n, num_teams=num_teams,
                              save_path=os.path.join(save_path, "correlation_extremes.png"))
