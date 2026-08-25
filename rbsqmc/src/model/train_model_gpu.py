"""Train / filter / predict pipeline for the RBPF football model.

Phases: ``optimize`` (GPU log-marginal maximization), ``filter`` (local forward
filter + plots), ``predict`` (local sequential prediction). ``all`` (default)
runs all three end-to-end, reproducing the original behaviour. Config comes from
``--config``, the ``RBSQMC_CONFIG`` env var, or the repo default config.
"""

import argparse
import json
import os

import jax
import numpy as np
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from rbsqmc.src.data.data import get_results, get_training_data, concat_football_results
from rbsqmc.src.model.optimization import (
    run_filter_unbiased,
    logmarginal_maximize,
)
from rbsqmc.src.utils.helpers import (
    default_init_params,
    resolve_teams,
    save_params,
    load_params,
)
from rbsqmc.src.utils.graphic import (
    plot_all,
    plot_logmarginal_history_train_test,
    plot_gradient_norm_curve,
)
from datetime import datetime


# Default configuration; a config JSON overrides any of these. ``output_dir``
# is filled in per-run.
DEFAULT_CONFIG = {
    "training_start_date": "1900-01-01",
    "test_start_date": "2025-01-01",
    "prediction_start_date": "2026-06-11",
    "n_particles": 250,
    "max_goals": 8,
    "seed": 0,
    "n_epochs": 20,
    "learning_rate": 0.1,
    "n_reps": 20,
    "include_friendly": True,
    "teams": "worldcup2026",
    "download": False,
    "output_dir": None,
    "gamma_0_prior_params": {
        "scale": 1.0,
        "dof": 5.0,
        "strength": 1.0
    },
}

DEFAULT_CONFIG_PATH = "rbsqmc/scripts/config/model_unbiased_gpu_config.json"


def load_config(config_path: str | None = None) -> dict:
    """Load the run configuration, merging a JSON file over the defaults."""
    cfg = dict(DEFAULT_CONFIG)
    path = config_path or os.environ.get("RBSQMC_CONFIG") or DEFAULT_CONFIG_PATH
    if path and os.path.exists(path):
        with open(path, "r") as f:
            cfg.update(json.load(f))
        print(f"Loaded config from {path}")
    else:
        print(f"No config file found at {path}; using defaults")
    return cfg


def prepare_data(cfg: dict, download: bool = False):
    """Resolve teams, load train/test/prediction splits, and build init params.

    Args:
        download: if True, pull the results CSV from the network instead of
            reading the local ``results.parquet`` cache (required on a fresh
            clone such as the Colab VM, where the parquet cache is absent).
    """
    teams_only = resolve_teams(cfg)
    (train_df, test_df, prediction_df), (
        train_model_inputs,
        test_model_inputs,
        prediction_model_inputs,
    ), team_id_to_name = get_training_data(
        train_start_date=cfg["training_start_date"],
        test_start_date=cfg["test_start_date"],
        prediction_start_date=cfg["prediction_start_date"],
        max_goals=cfg["max_goals"],
        include_friendly=cfg["include_friendly"],
        teams_only=teams_only,
        download=download,
    )
    print("Extracted training data:")
    print(f"  Training data: {len(train_df)} matches. Training data from {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"  Test data: {len(test_df)} matches. Test data from {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"  Prediction data: {len(prediction_df)} matches. Prediction data from {prediction_df['date'].min().date()} to {prediction_df['date'].max().date()}")

    num_teams = len(team_id_to_name)
    params = default_init_params(num_teams=num_teams, team_id_to_name=team_id_to_name)
    return (
        train_df, test_df, prediction_df,
        train_model_inputs, test_model_inputs, prediction_model_inputs,
        team_id_to_name, params,
    )


def _write_run_config(cfg: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote run config to {os.path.join(output_dir, 'run_config.json')}")


def run_optimize(cfg: dict, output_dir: str):
    """GPU phase: log-marginal maximization; writes params + summary + curves."""
    _write_run_config(cfg, output_dir)
    key = jax.random.PRNGKey(cfg["seed"])
    (
        train_df, test_df, prediction_df,
        train_model_inputs, test_model_inputs, prediction_model_inputs,
        team_id_to_name, params,
    ) = prepare_data(cfg, download=cfg.get("download", False))

    # Baseline train logZ with the initial params (for improvement comparison).
    key, baseline_key = jax.random.split(key, 2)
    baseline_logz = float(
        run_filter_unbiased(
            key=baseline_key,
            model_inputs=train_model_inputs,
            params=params,
            n_particles=cfg["n_particles"],
            max_goals=cfg["max_goals"],
        )[0].log_normalizing_constant[-1]
    )
    print(f"[optimize] baseline train logZ = {baseline_logz:.4f}")

    key, opt_key = jax.random.split(key, 2)
    gamma_0_prior_params = None
    if cfg.get("gamma_prior_strength") is not None:
        gamma_0_prior_params = {
            "scale": None,  # will default to initial params.gamma_0
            "dof": cfg.get("gamma_prior_dof", 5.0),
            "strength": cfg["gamma_prior_strength"],
        }
    best_params, train_logz_history, test_logz_history, grad_norm_history = logmarginal_maximize(
        key=opt_key,
        train_model_inputs=train_model_inputs,
        test_model_inputs=test_model_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
        n_epochs=cfg["n_epochs"],
        learning_rate=cfg["learning_rate"],
        n_reps=cfg["n_reps"],
        gamma_0_prior_params=gamma_0_prior_params,
    )

    # Final filter logZ on train+test with the best params.
    observed_inputs = concat_football_results(train_model_inputs, test_model_inputs)
    key, final_key = jax.random.split(key, 2)
    final_filter_logz = float(
        run_filter_unbiased(
            key=final_key,
            model_inputs=observed_inputs,
            params=best_params,
            n_particles=cfg["n_particles"],
            max_goals=cfg["max_goals"],
        )[0].log_normalizing_constant[-1]
    )

    train_logz = [float(v) for v in train_logz_history]
    test_logz = [float(v) for v in test_logz_history]
    grad_norms = [float(v) for v in grad_norm_history]
    best_logz = max(train_logz) if train_logz else float("nan")

    save_params(params=best_params, path=os.path.join(output_dir, "params_unbiased.json"))

    summary = {
        "baseline_logZ": baseline_logz,
        "best_logZ": best_logz,
        "final_filter_logZ": final_filter_logz,
        "train_logZ_history": train_logz,
        "test_logZ_history": test_logz,
        "gradient_norm_history": grad_norms,
        "n_epochs": int(cfg["n_epochs"]),
        "train_match_count": int(train_model_inputs.match_mask.sum()),
        "test_match_count": int(test_model_inputs.match_mask.sum()),
    }
    with open(os.path.join(output_dir, "optimization_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote optimization summary to {os.path.join(output_dir, 'optimization_summary.json')}")

    plot_logmarginal_history_train_test(
        train_logz_history=train_logz_history,
        train_match_count=int(train_model_inputs.match_mask.sum()),
        test_logz_history=test_logz_history,
        test_match_count=int(test_model_inputs.match_mask.sum()),
        save_path=os.path.join(output_dir, "optimization_logZ_curve.png"),
    )
    plot_gradient_norm_curve(
        grad_norm_history=grad_norm_history,
        save_path=os.path.join(output_dir, "gradient_norm_curve.png"),
    )
    # Save the run config again immediately after the run completes (mirrors
    # train_model.py, which persists run_config.json for the completed run).
    _write_run_config(cfg, output_dir)
    return best_params


def run_filter(cfg: dict, params, output_dir: str):
    """Local phase: forward filter on train+test; plot + save states."""
    os.makedirs(output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg["seed"])
    (
        train_df, test_df, prediction_df,
        train_model_inputs, test_model_inputs, prediction_model_inputs,
        team_id_to_name, _init_params,
    ) = prepare_data(cfg)

    observed_inputs = concat_football_results(train_model_inputs, test_model_inputs)
    key, filter_key = jax.random.split(key, 2)
    final_states, final_model_inputs_rbpf = run_filter_unbiased(
        key=filter_key,
        model_inputs=observed_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    full_dates = pd.concat([train_df["date"], test_df["date"]]).to_numpy()
    plot_all(
        filtered_states=final_states,
        augmented_results=final_model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=output_dir,
        timestamps=full_dates,
        params=params,
    )
    _write_timeseries_states(final_states, team_id_to_name, full_dates, output_dir)
    return final_states, final_model_inputs_rbpf, observed_inputs, team_id_to_name


def _write_timeseries_states(filtered_states, team_id_to_name, full_dates, output_dir):
    """Write filtered state trajectories (mean attack/defense per team) as JSON."""
    import numpy as np

    x = np.asarray(filtered_states.particles.x)  # (T+1, N, M, 2)
    mean_states = x.mean(axis=1)  # (T+1, M, 2)
    dates = [str(pd.Timestamp(d).date()) for d in full_dates]
    series = mean_states[1:]  # drop the initial state to align to T observed dates
    teams = {
        team_id_to_name[i]: {
            "attack": series[:, i, 0].astype(float).tolist(),
            "defense": series[:, i, 1].astype(float).tolist(),
        }
        for i in range(len(team_id_to_name))
    }
    path = os.path.join(output_dir, "timeseries_states.json")
    with open(path, "w") as f:
        json.dump({"dates": dates, "teams": teams}, f, indent=2)
    print(f"Saved timeseries states to {os.path.abspath(path)}")


def run_predict(cfg: dict, params, output_dir: str):
    """Local phase: sequential prediction on the upcoming-match split."""
    from rbsqmc.src.model.predict import run_sequential_predict
    from rbsqmc.src.utils.helpers import (
        build_match_predictions,
        save_match_predictions,
    )

    os.makedirs(output_dir, exist_ok=True)
    key = jax.random.PRNGKey(cfg["seed"])
    (
        train_df, test_df, prediction_df,
        train_model_inputs, test_model_inputs, prediction_model_inputs,
        team_id_to_name, _init_params,
    ) = prepare_data(cfg)

    observed_inputs = concat_football_results(train_model_inputs, test_model_inputs)
    key, pred_key = jax.random.split(key, 2)
    pred_grids, pred_logprobs, daily_logp = run_sequential_predict(
        key=pred_key,
        observed_inputs=observed_inputs,
        prediction_inputs=prediction_model_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    predictions = build_match_predictions(
        all_grids=pred_grids,
        all_logp_actual=pred_logprobs,
        prediction_inputs=prediction_model_inputs,
        team_id_to_name=team_id_to_name,
        max_goals=cfg["max_goals"],
    )
    save_match_predictions(predictions, save_dir=output_dir, max_goals=cfg["max_goals"])

    _write_post_prediction_rankings(
        cfg, params, observed_inputs, prediction_model_inputs, team_id_to_name, output_dir
    )
    return predictions


def _write_post_prediction_rankings(cfg, params, observed_inputs, prediction_inputs, team_id_to_name, output_dir):
    """Run the filter through observed + prediction matches and save final rankings."""
    import numpy as np

    full_inputs = concat_football_results(observed_inputs, prediction_inputs)
    key = jax.random.PRNGKey(cfg["seed"] + 1)
    states, _ = run_filter_unbiased(
        key=key,
        model_inputs=full_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )
    x_final = np.asarray(states.particles.x[-1])  # (N, M, 2)
    mean_strengths = x_final.mean(axis=0)  # (M, 2)
    total = mean_strengths[:, 0] + mean_strengths[:, 1]
    order = np.argsort(total)[::-1]
    payload = {
        "rank": list(range(1, len(order) + 1)),
        "team": [team_id_to_name.get(int(i), str(i)) for i in order],
        "attack": mean_strengths[order, 0].astype(float).tolist(),
        "defense": mean_strengths[order, 1].astype(float).tolist(),
        "total": total[order].astype(float).tolist(),
    }
    path = os.path.join(output_dir, "post_prediction_filter_rankings.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved post-prediction rankings to {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
# observe phase: plot selected teams' filtered attack/defense/total states
# ---------------------------------------------------------------------------
OBSERVE_DEFAULT_TEAMS = ["Spain", "England", "France", "Argentina"]

# Solid, clearly-distinct colours for each team (and their labels/lines).
OBSERVE_COLORS = [
    "#D62728",  # red
    "#1F77B4",  # blue
    "#2CA02C",  # green
    "#9467BD",  # purple
    "#FF7F0E",  # orange
    "#8C564B",  # brown
    "#17BECF",  # cyan
    "#E377C2",  # pink
    "#7F7F7F",  # gray
    "#BCBD22",  # olive
]


def _observe_weighted_filter_means(x, log_weights):
    """Weighted particle-mean of team states per time step.

    Args:
        x: ``(T, N, num_teams, 2)`` particle states.
        log_weights: ``(T, N)`` particle log-weights.

    Returns:
        ``(T, num_teams, 2)`` weighted mean (attack/defense in last axis).
    """
    finite_lw = np.where(np.isfinite(log_weights), log_weights, -np.inf)
    shifted = np.exp(log_weights - np.max(finite_lw, axis=1, keepdims=True))
    shifted[~np.isfinite(shifted)] = 0.0
    weight_sum = shifted.sum(axis=1, keepdims=True)
    uniform = np.full_like(shifted, 1.0 / x.shape[1])
    weights = np.divide(shifted, weight_sum, out=uniform, where=weight_sum > 0)
    return np.sum(x[..., :2] * weights[:, :, None, None], axis=1)


def _observe_collect_team_fixtures(prediction_inputs, team_id, team_id_to_name):
    """Return ``[(date, label), ...]`` for every prediction-window match.

    ``label`` is ``"Team vs Opponent (team_score - opp_score)"``.
    """
    fixtures = []
    home = np.asarray(prediction_inputs.matches.home_id)
    away = np.asarray(prediction_inputs.matches.away_id)
    home_score = np.asarray(prediction_inputs.matches.home_score)
    away_score = np.asarray(prediction_inputs.matches.away_score)
    mask = np.asarray(prediction_inputs.match_mask)
    date = np.asarray(prediction_inputs.date)

    team_name = team_id_to_name[int(team_id)]
    for t in range(home.shape[0]):
        for j in range(home.shape[1]):
            if not mask[t, j]:
                continue
            if home[t, j] == team_id:  # team is home
                opp = team_id_to_name[int(away[t, j])]
                ts, os_ = home_score[t, j], away_score[t, j]
            elif away[t, j] == team_id:  # team is away
                opp = team_id_to_name[int(home[t, j])]
                os_, ts = home_score[t, j], away_score[t, j]
            else:
                continue
            label = f"{team_name} vs {opp} ({ts} - {os_})"
            fixtures.append((date[t], label))
    return fixtures


def _observe_plot_team_states(
    dates_axis,
    means,
    team_ids,
    team_id_to_name,
    fixtures_by_team,
    save_path,
    draw_lines=False,
    xlim=None,
):
    """Plot 3 rows (attack / defense / total) for each team."""
    colors = OBSERVE_COLORS[: len(team_ids)]
    fig, (attack_ax, defense_ax, total_ax) = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True
    )

    for team_id, color in zip(team_ids, colors):
        name = team_id_to_name[int(team_id)]
        attack_ax.plot(dates_axis, means[:, int(team_id), 0],
                       color=color, linewidth=2.2, label=name)
        defense_ax.plot(dates_axis, means[:, int(team_id), 1],
                        color=color, linewidth=2.2, label=name)
        total_ax.plot(
            dates_axis,
            means[:, int(team_id), 0] + means[:, int(team_id), 1],
            color=color, linewidth=2.2, label=name,
        )

    if draw_lines:
        # Flatten fixtures across teams and group by day so same-day labels
        # can be staggered vertically to avoid overlap.
        annotations = [
            (date, label, colors[team_ids.index(team_id)])
            for team_id in team_ids
            for date, label in fixtures_by_team[int(team_id)]
        ]
        by_day: dict[int, list] = {}
        for date, label, color in annotations:
            by_day.setdefault(int(date), []).append((date, label, color))

        for day_group in by_day.values():
            # Vertical (rotated) labels extend upward roughly proportional to
            # their text length, so stagger same-day labels by an amount large
            # enough to clear the tallest label rather than a fixed tiny step.
            max_len = max(len(label) for _, label, _ in day_group)
            step = 0.04 + max_len * 0.014  # axes-fraction per label
            for i, (date, label, color) in enumerate(day_group):
                y_frac = 0.02 + i * step
                for ax in (attack_ax, defense_ax, total_ax):
                    ax.axvline(date, color=color, linestyle="--", alpha=0.6)
                total_ax.annotate(
                    label,
                    xy=(date, y_frac),
                    xycoords=("data", "axes fraction"),
                    rotation=90,
                    fontsize=7,
                    color=color,
                    ha="right",
                    va="bottom",
                )

    total_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    total_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    if xlim is not None:
        for ax in (attack_ax, defense_ax, total_ax):
            ax.set_xlim(*xlim)

    attack_ax.set_ylabel("Attack state")
    defense_ax.set_ylabel("Defense state")
    total_ax.set_ylabel("Total strength (attack + defense)")
    total_ax.set_xlabel("Date")
    for ax in (attack_ax, defense_ax, total_ax):
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    attack_ax.set_title(
        "Filtered attack/defense/total states  ·  dashed lines = match in "
        "prediction window" if draw_lines else "Filtered attack/defense/total states"
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {os.path.abspath(save_path)}")


def run_observe(cfg: dict, params, output_dir: str):
    """Local phase: plot selected teams' states over the full sequence.

    Produces three sets of outputs in ``output_dir``:
      * ``observe_team_states_full.png``   full train+test+prediction, no lines
      * ``observe_team_states.png``        prediction phase zoomed + fixture lines
      * ``observe_team_states.json``       per-team state series (quantitative)
    and a ``final_filter_after_prediction/`` folder via ``plot_all`` (final
    rankings, top strengths, correlation, filter states).
    """
    teams = list(cfg.get("observe_teams") or OBSERVE_DEFAULT_TEAMS)
    gamma_scale = float(cfg.get("observe_gamma_scale", 1.0))

    # Diagnostic knob: inflate gamma_0 (prior state covariance) to raise the
    # Kalman gain, making individual match results move strengths more.
    if gamma_scale != 1.0:
        params = params._replace(gamma_0=params.gamma_0 * gamma_scale)
        print(f"Observe: inflating gamma_0 by {gamma_scale:.3f}x")
    suffix = f"_gs{gamma_scale:g}" if gamma_scale != 1.0 else ""

    (
        _train_df, _test_df, _pred_df,
        train_model_inputs, test_model_inputs, prediction_model_inputs,
        team_id_to_name, _init_params,
    ) = prepare_data(cfg)

    observed_inputs = concat_football_results(train_model_inputs, test_model_inputs)
    full_inputs = concat_football_results(observed_inputs, prediction_model_inputs)
    n_obs_days = observed_inputs.timestamp.shape[0]

    key = jax.random.PRNGKey(cfg["seed"])
    print(f"Observe: filtering {full_inputs.timestamp.shape[0]} days "
          "(train+test+prediction)...")
    filtered_states, model_inputs_rbpf = run_filter_unbiased(
        key=key,
        model_inputs=full_inputs,
        params=params,
        n_particles=cfg["n_particles"],
        max_goals=cfg["max_goals"],
    )

    x = np.asarray(filtered_states.particles.x)      # (T+1, N, n_teams, 2)
    lw = np.asarray(filtered_states.log_weights)     # (T+1, N)
    means = _observe_weighted_filter_means(x, lw)    # (T+1, n_teams, 2)

    dates = np.asarray(full_inputs.date)
    dates_axis = np.concatenate([dates[:1], dates])  # align with initial state

    # Resolve requested team ids.
    name_to_id = {name: i for i, name in team_id_to_name.items()}
    team_ids = [name_to_id[name] for name in teams if name in name_to_id]
    missing = [name for name in teams if name not in name_to_id]
    if missing:
        print(f"Observe: warning, unknown team names (skipped): {missing}")
    if not team_ids:
        raise ValueError("No requested teams found in team_id_to_name")

    fixtures_by_team = {
        int(tid): _observe_collect_team_fixtures(
            prediction_model_inputs, tid, team_id_to_name
        )
        for tid in team_ids
    }

    # Quantitative export of the selected teams' state series.
    export_path = os.path.join(output_dir, f"observe_team_states{suffix}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(export_path, "w") as f:
        json.dump({
            "teams": [team_id_to_name[int(tid)] for tid in team_ids],
            "dates": [str(np.datetime64(int(d), "D")) for d in dates_axis],
            "attack": {
                team_id_to_name[int(tid)]: means[:, int(tid), 0].astype(float).tolist()
                for tid in team_ids
            },
            "defense": {
                team_id_to_name[int(tid)]: means[:, int(tid), 1].astype(float).tolist()
                for tid in team_ids
            },
        }, f, indent=2)
    print(f"Saved team states to {os.path.abspath(export_path)}")

    # Plot 1: full train + test + prediction range, no fixture lines.
    _observe_plot_team_states(
        dates_axis, means, team_ids, team_id_to_name, fixtures_by_team,
        os.path.join(output_dir, f"observe_team_states_full{suffix}.png"),
        draw_lines=False,
    )
    # Plot 2: prediction phase zoomed, with fixture lines.
    _observe_plot_team_states(
        dates_axis, means, team_ids, team_id_to_name, fixtures_by_team,
        os.path.join(output_dir, f"observe_team_states{suffix}.png"),
        draw_lines=True,
        xlim=(dates[n_obs_days], dates[-1]),
    )
    # Plot 3: final states / rankings / top strengths AFTER prediction.
    plot_all(
        filtered_states=filtered_states,
        augmented_results=model_inputs_rbpf,
        team_id_to_name=team_id_to_name,
        top_n=10,
        save_path=os.path.join(output_dir, "final_filter_after_prediction"),
        timestamps=dates_axis,
        params=params,
    )


def _default_output_dir() -> str:
    date_text = datetime.now().strftime("gd_%Y%m%d_%H%M%S")
    return f"rbsqmc/outputs/train_model/{date_text}/"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    def add_common(p):
        p.add_argument("--config", help="Path to config JSON (default: RBSQMC_CONFIG or repo default)")
        p.add_argument("--params", help="Path to params JSON (filter/predict phases)")
        p.add_argument("--out", help="Output directory for this phase")

    for name in ("optimize", "filter", "predict", "observe", "all"):
        add_common(sub.add_parser(name))

    args = parser.parse_args(argv)
    command = args.command or "all"
    cfg = load_config(args.config)

    if command == "optimize":
        output_dir = args.out or cfg.get("output_dir") or _default_output_dir()
        cfg["output_dir"] = output_dir
        run_optimize(cfg, output_dir)
        return 0

    if command == "observe":
        if not args.params:
            parser.error("observe phase requires --params")
        output_dir = args.out or os.path.join(cfg.get("output_dir") or _default_output_dir(), "observe")
        run_observe(cfg, load_params(args.params), output_dir)
        return 0

    if command == "filter":
        if not args.params:
            parser.error("filter phase requires --params")
        output_dir = args.out or os.path.join(cfg.get("output_dir") or _default_output_dir(), "filtered")
        run_filter(cfg, load_params(args.params), output_dir)
        return 0

    if command == "predict":
        if not args.params:
            parser.error("predict phase requires --params")
        output_dir = args.out or os.path.join(cfg.get("output_dir") or _default_output_dir(), "predict")
        run_predict(cfg, load_params(args.params), output_dir)
        return 0

    # command == "all": reproduce the original end-to-end behaviour.
    output_dir = args.out or cfg.get("output_dir") or _default_output_dir()
    cfg["output_dir"] = output_dir
    best_params = run_optimize(cfg, output_dir)
    run_filter(cfg, best_params, os.path.join(output_dir, "final_filter"))
    run_predict(cfg, best_params, os.path.join(output_dir, "prediction"))
    return 0


if __name__ == "__main__":
    main()