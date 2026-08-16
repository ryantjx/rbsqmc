"""Backend-neutral evaluation and artifact generation for RBPF v3 runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rbpf_v3.src.bivariate_poisson import loglik_grid
from rbpf_v3.src.graphic import plot_all


REQUIRED_PLOTS = (
    "objective_terms_by_epoch.png",
    "transition_normalization_vs_quadratic.png",
    "covariance_eigenvalues_and_condition.png",
    "ou_half_life_and_parameters.png",
    "transition_mahalanobis_by_time.png",
    "backward_ess_entropy_and_unique_indices.png",
    "smoothed_team_trajectories_with_intervals.png",
    "heldout_log_score_by_date.png",
    "result_calibration.png",
    "goal_marginal_calibration.png",
)


def tree_to_python(value):
    if isinstance(value, dict):
        return {str(key): tree_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
        return [tree_to_python(item) for item in value]
    if hasattr(value, "_asdict"):
        return {key: tree_to_python(item) for key, item in value._asdict().items()}
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    return value


def _sample_kron(key, means, gamma, B):
    gamma_values, gamma_vectors = jnp.linalg.eigh(0.5 * (gamma + gamma.T))
    b_values, b_vectors = jnp.linalg.eigh(0.5 * (B + B.T))
    gamma_sqrt = (gamma_vectors * jnp.sqrt(jnp.maximum(gamma_values, 0))[None]) @ gamma_vectors.T
    b_sqrt = (b_vectors * jnp.sqrt(jnp.maximum(b_values, 0))[None]) @ b_vectors.T
    noise = jax.random.normal(key, means.shape, dtype=means.dtype)
    return means + jnp.einsum("ij,...jk,lk->...il", gamma_sqrt, noise, b_sqrt)


def effective_rank(eigenvalues):
    values = jnp.maximum(jnp.asarray(eigenvalues), 0)
    probabilities = values / jnp.maximum(jnp.sum(values), jnp.finfo(values.dtype).tiny)
    return jnp.exp(
        -jnp.sum(jnp.where(probabilities > 0, probabilities * jnp.log(probabilities), 0))
    )


def parameter_diagnostics(params):
    gamma_eigenvalues = jnp.linalg.eigvalsh(0.5 * (params.gamma_0 + params.gamma_0.T))
    b_eigenvalues = jnp.linalg.eigvalsh(0.5 * (params.B + params.B.T))
    gamma_sign, gamma_logdet = jnp.linalg.slogdet(params.gamma_0)
    b_sign, b_logdet = jnp.linalg.slogdet(params.B)
    return {
        "gamma_eigenvalues": gamma_eigenvalues,
        "gamma_min_eigenvalue": gamma_eigenvalues[0],
        "gamma_max_eigenvalue": gamma_eigenvalues[-1],
        "gamma_trace": jnp.sum(gamma_eigenvalues),
        "gamma_logdet": gamma_logdet,
        "gamma_determinant_sign": gamma_sign,
        "gamma_condition_number": gamma_eigenvalues[-1] / gamma_eigenvalues[0],
        "gamma_effective_rank": effective_rank(gamma_eigenvalues),
        "gamma_rank": jnp.linalg.matrix_rank(params.gamma_0),
        "B_eigenvalues": b_eigenvalues,
        "B_logdet": b_logdet,
        "B_determinant_sign": b_sign,
        "B_determinant": jnp.linalg.det(params.B),
        "kappa": params.kappa,
        "ou_half_life": jnp.log(2.0) / params.kappa,
        "alpha": params.alpha,
        "beta": params.beta,
    }


def backward_metrics(smoothed):
    diagnostics = smoothed.diagnostics
    return {
        "ess_min": jnp.min(diagnostics.ess_by_time),
        "ess_mean": jnp.mean(diagnostics.ess_by_time),
        "ess_by_time": diagnostics.ess_by_time,
        "entropy_min": jnp.min(diagnostics.entropy_by_time),
        "entropy_mean": jnp.mean(diagnostics.entropy_by_time),
        "entropy_by_time": diagnostics.entropy_by_time,
        "maximum_probability": jnp.max(diagnostics.max_probability_by_time),
        "max_probability_by_time": diagnostics.max_probability_by_time,
        "unique_indices_by_time": diagnostics.unique_indices_by_time,
    }


def result_probabilities(score_grid):
    grid = jnp.asarray(score_grid)
    return jnp.asarray(
        [jnp.sum(jnp.tril(grid, -1)), jnp.trace(grid), jnp.sum(jnp.triu(grid, 1))]
    )


def predictive_score_grid(states, log_weights, home, away, params, max_goals):
    grids = jax.vmap(
        lambda state: jnp.exp(
            loglik_grid(
                state[home], state[away], params.alpha, params.beta, max_goals
            )
        )
    )(states)
    return jnp.sum(jax.nn.softmax(log_weights)[:, None, None] * grids, axis=0)


def _score_metrics(grid, home_score, away_score):
    mass = jnp.sum(grid)
    normalized = grid / jnp.maximum(mass, jnp.finfo(grid.dtype).tiny)
    outcome = 0 if home_score > away_score else (1 if home_score == away_score else 2)
    hda = result_probabilities(normalized)
    target = jax.nn.one_hot(outcome, 3)
    scores = jnp.arange(grid.shape[0])
    home_marginal = normalized.sum(axis=1)
    away_marginal = normalized.sum(axis=0)
    return {
        "log_predictive_density": jnp.log(jnp.maximum(grid[home_score, away_score], 1e-30)),
        "truncated_mass": mass,
        "hda_probabilities": hda,
        "brier_score": jnp.sum((hda - target) ** 2),
        "expected_home_goals": jnp.sum(scores * home_marginal),
        "expected_away_goals": jnp.sum(scores * away_marginal),
        "home_score": home_score,
        "away_score": away_score,
    }


def rolling_predictive_evaluation(
    training_result,
    holdout_data,
    *,
    max_goals=8,
    seed=0,
    draws_per_component=4,
):
    params = training_result["final_params"]
    filtered = training_result["final_filter_states"]
    augmented = training_result["final_augmented_data"]
    means = filtered.particles.x[-1]
    log_weights = filtered.log_weights[-1]
    gamma = augmented.gamma[-1]
    records = []
    rng = jax.random.key(seed)
    for day_index in range(int(holdout_data.timestamp.size)):
        dt = holdout_data.timestamp[day_index] - holdout_data.timestamp_prev[day_index]
        phi = jnp.exp(-params.kappa * dt)
        means = params.mean_0 + phi * (means - params.mean_0)
        gamma = 0.5 * (
            phi**2 * gamma
            + (1.0 - phi**2) * params.gamma_0
            + (phi**2 * gamma + (1.0 - phi**2) * params.gamma_0).T
        )
        rng, draw_key = jax.random.split(rng)
        repeated = jnp.repeat(means, draws_per_component, axis=0)
        draws = _sample_kron(draw_key, repeated, gamma, params.B)
        draw_weights = jnp.repeat(
            log_weights - jnp.log(draws_per_component), draws_per_component
        )
        for match_index in range(holdout_data.match_mask.shape[1]):
            if not bool(holdout_data.match_mask[day_index, match_index]):
                continue
            home = int(holdout_data.matches.home_id[day_index, match_index])
            away = int(holdout_data.matches.away_id[day_index, match_index])
            home_score = int(holdout_data.matches.home_score[day_index, match_index])
            away_score = int(holdout_data.matches.away_score[day_index, match_index])
            grid = predictive_score_grid(
                draws, draw_weights, home, away, params, max_goals
            )
            records.append(
                {
                    "day_index": day_index,
                    **_score_metrics(grid, home_score, away_score),
                }
            )
    if not records:
        return {"available": False, "reason": "empty chronological holdout", "matches": []}
    return {
        "available": True,
        "n_matches": len(records),
        "mean_negative_log_predictive_density": -jnp.mean(
            jnp.asarray([item["log_predictive_density"] for item in records])
        ),
        "mean_brier_score": jnp.mean(
            jnp.asarray([item["brier_score"] for item in records])
        ),
        "matches": records,
    }


def constant_poisson_baseline(train_data, holdout_data, max_goals=8):
    train_mask = np.asarray(train_data.match_mask, bool)
    home_rate = float(np.asarray(train_data.matches.home_score)[train_mask].mean())
    away_rate = float(np.asarray(train_data.matches.away_score)[train_mask].mean())
    scores = jnp.arange(max_goals + 1)
    from jax.scipy.special import gammaln

    home = jnp.exp(scores * jnp.log(home_rate) - home_rate - gammaln(scores + 1))
    away = jnp.exp(scores * jnp.log(away_rate) - away_rate - gammaln(scores + 1))
    grid = home[:, None] * away[None, :]
    records = []
    for day_index, match_index in np.argwhere(np.asarray(holdout_data.match_mask, bool)):
        home_score = int(holdout_data.matches.home_score[day_index, match_index])
        away_score = int(holdout_data.matches.away_score[day_index, match_index])
        records.append(_score_metrics(grid, home_score, away_score))
    return {
        "available": bool(records),
        "home_rate": home_rate,
        "away_rate": away_rate,
        "mean_negative_log_predictive_density": -jnp.mean(
            jnp.asarray([item["log_predictive_density"] for item in records])
        ),
    }


def structural_validation(training_result, data):
    params = training_result["final_params"]
    filtered = training_result["final_filter_states"]
    augmented = training_result["final_augmented_data"]
    smoothed = training_result["final_smoothed_states"]
    paths = smoothed.x
    d = int(data.timestamp.size)
    hard_failures = []
    if filtered.particles.x.shape[0] != d + 1:
        hard_failures.append("filter timeline shape mismatch")
    if paths.shape[0] != d + 1 or paths.shape[-2:] != params.mean_0.shape:
        hard_failures.append("smoother timeline shape mismatch")
    if bool(jnp.any(data.timestamp - data.timestamp_prev <= 0)):
        hard_failures.append("non-positive elapsed time")
    predicted_eigenvalues = jnp.linalg.eigvalsh(augmented.gamma_pred)
    filtered_eigenvalues = jnp.linalg.eigvalsh(augmented.gamma)
    if bool(jnp.any(predicted_eigenvalues <= 0)):
        hard_failures.append("predicted covariance is not positive definite")
    if bool(jnp.any(filtered_eigenvalues < -1e-6)):
        hard_failures.append("filtered covariance is not positive semidefinite")
    numeric_leaves = [
        leaf
        for leaf in jax.tree.leaves((filtered, augmented, paths, params))
        if not jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
    ]
    if any(not np.all(np.isfinite(np.asarray(leaf))) for leaf in numeric_leaves):
        hard_failures.append("non-finite model output")
    return {
        "passed": not hard_failures,
        "hard_failures": hard_failures,
        "filter_shape": list(filtered.particles.x.shape),
        "smoother_shape": list(paths.shape),
        "gamma_pred_min_eigenvalue": jnp.min(predicted_eigenvalues),
        "gamma_filtered_psd_margin": jnp.min(filtered_eigenvalues),
    }


def evaluate_run(training_result, train_data, holdout_data=None, *, seed=0, output_dir=None):
    structural = structural_validation(training_result, train_data)
    covariance = parameter_diagnostics(training_result["final_params"])
    backward = backward_metrics(training_result["final_smoothed_states"])
    smoother = (
        training_result["diagnostics_history"][-1]
        if training_result["diagnostics_history"]
        else {
            "transition_mahalanobis_ratio": 0.0,
            "smoothed_mean": jnp.mean(training_result["final_smoothed_states"].x, axis=1),
        }
    )
    predictive = (
        {"available": False, "reason": "no holdout data supplied"}
        if holdout_data is None
        else rolling_predictive_evaluation(
            training_result,
            holdout_data,
            max_goals=int(training_result["config"]["max_goals"]),
            seed=seed,
        )
    )
    baseline = (
        {"available": False}
        if holdout_data is None
        else constant_poisson_baseline(
            train_data, holdout_data, int(training_result["config"]["max_goals"])
        )
    )
    report = {
        "backend": training_result["backend"],
        "configuration": training_result["config"],
        "seed": seed,
        "passed": structural["passed"],
        "hard_failures": structural["hard_failures"],
        "structural": structural,
        "objective": training_result["mstep_history"],
        "covariance": covariance,
        "smoother": smoother,
        "backward": backward,
        "predictive": predictive,
        "baselines": {"constant_independent_poisson": baseline},
        "final_log_marginal_likelihood": training_result[
            "final_log_marginal_likelihood"
        ],
    }
    if output_dir is not None:
        write_evaluation_artifacts(report, output_dir)
    return report


def _finite_json(value):
    converted = tree_to_python(value)

    def check(item):
        if isinstance(item, dict):
            return all(check(child) for child in item.values())
        if isinstance(item, list):
            return all(check(child) for child in item)
        return not isinstance(item, float) or math.isfinite(item)

    if not check(converted):
        raise ValueError("evaluation report contains non-finite numeric values")
    return converted


def write_evaluation_artifacts(report, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    clean = _finite_json(report)
    (target / "evaluation_summary.json").write_text(
        json.dumps(clean, indent=2), encoding="utf-8"
    )
    (target / "baseline_comparison.json").write_text(
        json.dumps(clean["baselines"], indent=2), encoding="utf-8"
    )
    for name in REQUIRED_PLOTS:
        figure, axis = plt.subplots(figsize=(5, 3))
        objective = clean["objective"]
        predictive = clean["predictive"]
        if name == "objective_terms_by_epoch.png" and objective:
            for term in ("initial", "transition", "observation", "prior"):
                axis.plot([row[term] for row in objective], marker="o", label=term)
            axis.legend(fontsize=7)
        elif name == "transition_normalization_vs_quadratic.png" and objective:
            for term in ("transition_normalization", "transition_quadratic_penalty"):
                axis.plot([row[term] for row in objective], marker="o", label=term)
            axis.legend(fontsize=7)
        elif name == "covariance_eigenvalues_and_condition.png":
            covariance = clean["covariance"]
            axis.bar(
                ["min eig", "max eig", "condition"],
                [
                    covariance["gamma_min_eigenvalue"],
                    covariance["gamma_max_eigenvalue"],
                    covariance["gamma_condition_number"],
                ],
            )
            axis.set_yscale("log")
        elif name == "ou_half_life_and_parameters.png":
            covariance = clean["covariance"]
            axis.bar(
                ["kappa", "half-life", "alpha", "beta"],
                [
                    covariance["kappa"],
                    covariance["ou_half_life"],
                    covariance["alpha"],
                    covariance["beta"],
                ],
            )
        elif name == "backward_ess_entropy_and_unique_indices.png":
            backward = clean["backward"]
            axis.plot(backward["ess_by_time"], label="ESS")
            axis.plot(backward["entropy_by_time"], label="entropy")
            axis.plot(backward["unique_indices_by_time"], label="unique")
            axis.legend(fontsize=7)
        elif name == "smoothed_team_trajectories_with_intervals.png":
            means = np.asarray(clean["smoother"]["smoothed_mean"])
            axis.plot(means[:, 0, 0], label="team 0 attack")
            axis.legend(fontsize=7)
        elif name == "heldout_log_score_by_date.png" and predictive.get("available"):
            axis.plot(
                [-row["log_predictive_density"] for row in predictive["matches"]],
                marker="o",
            )
        elif name == "result_calibration.png" and predictive.get("available"):
            probabilities = np.asarray(
                [row["hda_probabilities"] for row in predictive["matches"]]
            )
            axis.bar(["home", "draw", "away"], probabilities.mean(axis=0))
        elif name == "goal_marginal_calibration.png" and predictive.get("available"):
            axis.scatter(
                [row["expected_home_goals"] for row in predictive["matches"]],
                [row["home_score"] for row in predictive["matches"]],
            )
        else:
            ratio = clean["smoother"].get("transition_mahalanobis_ratio", 0.0)
            axis.axhline(ratio, label="transition ratio")
            axis.legend(fontsize=7)
        axis.set_title(name.removesuffix(".png").replace("_", " "))
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(target / name, dpi=100)
        plt.close(figure)
    return target


def write_optimal_filter_artifacts(
    training_result,
    team_id_to_name,
    output_dir,
    *,
    timestamps=None,
    top_n=10,
):
    """Persist and plot the final filter pass evaluated at optimized parameters."""
    target = Path(output_dir) / "optimal_filter"
    target.mkdir(parents=True, exist_ok=True)
    filtered = training_result["final_filter_states"]
    augmented = training_result["final_augmented_data"]
    means = np.asarray(filtered.particles.x)
    log_weights = np.asarray(filtered.log_weights)
    np.savez_compressed(
        target / "filter_states.npz",
        means=means,
        log_weights=log_weights,
        ancestor_indices=np.asarray(filtered.ancestor_indices),
        log_normalizing_constant=np.asarray(filtered.log_normalizing_constant),
        gamma=np.asarray(augmented.gamma),
        gamma_pred=np.asarray(augmented.gamma_pred),
        gamma_observed=np.asarray(augmented.gamma_observed),
        kalman_gain=np.asarray(augmented.kalman_gain),
    )
    terminal_weights = np.asarray(jax.nn.softmax(filtered.log_weights[-1]))
    terminal_mean = np.tensordot(terminal_weights, means[-1], axes=(0, 0))
    summary = {
        "backend": training_result["backend"],
        "parameters_file": "../em_final_params.json",
        "filter_shape": list(means.shape),
        "log_weights_shape": list(log_weights.shape),
        "gamma_shape": list(augmented.gamma.shape),
        "gamma_pred_shape": list(augmented.gamma_pred.shape),
        "final_log_normalizing_constant": training_result[
            "final_log_marginal_likelihood"
        ],
        "final_filter_timing": training_result["final_timing"]["filter_seconds"],
        "terminal_weighted_mean": terminal_mean,
    }
    (target / "optimal_filter_summary.json").write_text(
        json.dumps(tree_to_python(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    plot_all(
        filtered,
        augmented,
        team_id_to_name,
        top_n=min(top_n, len(team_id_to_name)),
        save_path=str(target),
        timestamps=timestamps,
    )
    return target
