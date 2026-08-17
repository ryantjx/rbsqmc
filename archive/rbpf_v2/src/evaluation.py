from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .bivariate_poisson import loglik_grid
from .kron import kron_logdet
from .kron import sample_kron_psd
from .model import run_filter
from .smoothing import smoothed_path_diagnostics
from .utils import tree_to_python


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


def effective_rank(eigenvalues):
    values = jnp.maximum(jnp.asarray(eigenvalues), 0)
    probabilities = values / jnp.maximum(jnp.sum(values), jnp.finfo(values.dtype).tiny)
    return jnp.exp(-jnp.sum(jnp.where(probabilities > 0,
                                      probabilities * jnp.log(probabilities), 0.0)))


def parameter_diagnostics(params, representative_dt=(1, 2, 7, 30)):
    gamma_eig = jnp.linalg.eigvalsh(0.5 * (params.gamma_0 + params.gamma_0.T))
    b_eig = jnp.linalg.eigvalsh(params.B)
    sign, logdet = jnp.linalg.slogdet(params.gamma_0)
    b_sign, b_logdet = jnp.linalg.slogdet(params.B)
    q = {}
    for dt in representative_dt:
        scale = 1 - jnp.exp(-2 * params.kappa * dt)
        q[str(dt)] = {"min_eigenvalue": scale * gamma_eig[0] * b_eig[0],
                      "max_eigenvalue": scale * gamma_eig[-1] * b_eig[-1]}
    return {
        "gamma_min_eigenvalue": gamma_eig[0], "gamma_max_eigenvalue": gamma_eig[-1],
        "gamma_trace": jnp.sum(gamma_eig), "gamma_logdet": logdet,
        "gamma_determinant_sign": sign,
        "gamma_condition_number": gamma_eig[-1] / gamma_eig[0],
        "gamma_effective_rank": effective_rank(gamma_eig),
        "gamma_diagonal_min": jnp.min(jnp.diag(params.gamma_0)),
        "gamma_diagonal_median": jnp.median(jnp.diag(params.gamma_0)),
        "gamma_diagonal_max": jnp.max(jnp.diag(params.gamma_0)),
        "B_eigenvalues": b_eig, "B_logdet": b_logdet, "B_determinant_sign": b_sign,
        "kappa": params.kappa, "ou_half_life": jnp.log(2.0) / params.kappa,
        "alpha": params.alpha, "beta": params.beta, "lambda_3": jnp.exp(params.beta),
        "Q_eigenvalues_by_dt": q,
    }


def backward_metrics(probabilities, indices):
    p = jnp.asarray(probabilities)
    ess = 1 / jnp.sum(p**2, axis=-1)
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0), axis=-1)
    unique = jnp.asarray([len(np.unique(np.asarray(row))) for row in indices])
    return {"ess_min": jnp.min(ess), "ess_mean": jnp.mean(ess),
            "ess_p05": jnp.percentile(ess, 5), "ess_p95": jnp.percentile(ess, 95),
            "entropy_min": jnp.min(entropy), "entropy_mean": jnp.mean(entropy),
            "maximum_probability": jnp.max(p), "unique_indices_by_time": unique}


def result_probabilities(score_grid):
    grid = jnp.asarray(score_grid)
    size = grid.shape[0]
    home = jnp.sum(jnp.tril(grid, -1))
    draw = jnp.trace(grid)
    away = jnp.sum(jnp.triu(grid, 1))
    return jnp.array([home, draw, away])


def brier_score(probabilities, outcome: int):
    target = jax.nn.one_hot(outcome, probabilities.shape[-1])
    return jnp.sum((probabilities - target) ** 2)


def ranked_probability_score(probabilities, observed: int):
    cumulative = jnp.cumsum(probabilities)
    target = jnp.arange(probabilities.size) >= observed
    return jnp.sum((cumulative - target) ** 2)


def predictive_score_grid(states, log_weights, h, a, params, max_goals):
    grids = jax.vmap(lambda x: jnp.exp(loglik_grid(
        x[h], x[a], params.alpha, params.beta, max_goals
    )))(states)
    grid = jnp.sum(jax.nn.softmax(log_weights)[:, None, None] * grids, axis=0)
    # Include the represented truncated mass; do not silently renormalize tails.
    return grid


def _score_metrics(grid, home_score, away_score):
    mass = jnp.sum(grid)
    normalized = grid / jnp.maximum(mass, jnp.finfo(grid.dtype).tiny)
    outcome = 0 if home_score > away_score else (1 if home_score == away_score else 2)
    hda = result_probabilities(normalized)
    home_marginal, away_marginal = normalized.sum(axis=1), normalized.sum(axis=0)
    scores = jnp.arange(grid.shape[0])
    home_cdf, away_cdf = jnp.cumsum(home_marginal), jnp.cumsum(away_marginal)
    home_lo, home_hi = jnp.searchsorted(home_cdf, .05), jnp.searchsorted(home_cdf, .95)
    away_lo, away_hi = jnp.searchsorted(away_cdf, .05), jnp.searchsorted(away_cdf, .95)
    return {
        "log_predictive_density": jnp.log(jnp.maximum(grid[home_score, away_score], 1e-30)),
        "truncated_mass": mass, "hda_probabilities": hda,
        "brier_score": brier_score(hda, outcome),
        "home_rps": ranked_probability_score(home_marginal, home_score),
        "away_rps": ranked_probability_score(away_marginal, away_score),
        "home_absolute_error": jnp.abs(jnp.sum(scores * home_marginal) - home_score),
        "away_absolute_error": jnp.abs(jnp.sum(scores * away_marginal) - away_score),
        "expected_home_goals": jnp.sum(scores * home_marginal),
        "expected_away_goals": jnp.sum(scores * away_marginal),
        "home_interval_coverage": (home_score >= home_lo) & (home_score <= home_hi),
        "away_interval_coverage": (away_score >= away_lo) & (away_score <= away_hi),
        "home_interval_width": home_hi - home_lo, "away_interval_width": away_hi - away_lo,
        "predicted_both_teams_score": jnp.sum(normalized[1:, 1:]),
        "observed_both_teams_score": (home_score > 0) & (away_score > 0),
        "predicted_high_score_tail": jnp.sum(jnp.where(
            scores[:, None] + scores[None, :] >= 5, normalized, 0.0)),
        "observed_high_score_tail": home_score + away_score >= 5,
    }


def rolling_origin_predictive_evaluation(training_result, holdout_data, *,
                                         max_goals=8, seed=0, draws_per_component=4):
    """Score each future day before observing it, carrying the filter forward unconditioned."""
    params = training_result["final_params"]
    filtered = training_result["final_filter_states"]
    augmented = training_result["final_augmented_data"]
    means = filtered.particles.x[-1]
    log_weights = filtered.log_weights[-1]
    gamma = augmented.gamma[-1]
    records = []
    rng = jax.random.key(seed)
    for t in range(holdout_data.timestamp.size):
        dt = holdout_data.timestamp[t] - holdout_data.timestamp_prev[t]
        phi = jnp.exp(-params.kappa * dt)
        means = params.mean_0 + phi * (means - params.mean_0)
        predicted_gamma = phi**2 * gamma + (1 - phi**2) * params.gamma_0
        gamma = 0.5 * (predicted_gamma + predicted_gamma.T)
        rng, draw_key = jax.random.split(rng)
        keys = jax.random.split(draw_key, means.shape[0] * draws_per_component)
        repeated = jnp.repeat(means, draws_per_component, axis=0)
        draws = jax.vmap(lambda k, m: sample_kron_psd(k, m, gamma, params.B))(keys, repeated)
        draw_weights = jnp.repeat(log_weights - jnp.log(draws_per_component),
                                  draws_per_component)
        for l in range(holdout_data.match_mask.shape[1]):
            if not bool(holdout_data.match_mask[t, l]):
                continue
            h, a = int(holdout_data.matches.home_id[t, l]), int(holdout_data.matches.away_id[t, l])
            yh, ya = int(holdout_data.matches.home_score[t, l]), int(holdout_data.matches.away_score[t, l])
            grid = predictive_score_grid(draws, draw_weights, h, a, params, max_goals)
            records.append({"day_index": t, "date": int(holdout_data.date[t]),
                            "home_id": h, "away_id": a, "home_score": yh,
                            "away_score": ya, **_score_metrics(grid, yh, ya)})
    if not records:
        return {"available": False, "reason": "empty chronological holdout", "matches": []}
    keys = ("log_predictive_density", "brier_score", "home_rps", "away_rps",
            "home_absolute_error", "away_absolute_error", "home_interval_coverage",
            "away_interval_coverage", "home_interval_width", "away_interval_width",
            "predicted_both_teams_score", "observed_both_teams_score",
            "predicted_high_score_tail", "observed_high_score_tail")
    summary = {f"mean_{name}": jnp.mean(jnp.asarray([r[name] for r in records]))
               for name in keys}
    summary["mean_negative_log_predictive_density"] = -summary["mean_log_predictive_density"]
    return {"available": True, "n_matches": len(records), **summary, "matches": records}


def constant_poisson_baseline(train_data, holdout_data, max_goals=8):
    mask = np.asarray(train_data.match_mask, bool)
    home_rate = float(np.asarray(train_data.matches.home_score)[mask].mean())
    away_rate = float(np.asarray(train_data.matches.away_score)[mask].mean())
    scores = jnp.arange(max_goals + 1)
    from jax.scipy.special import gammaln
    home = jnp.exp(scores * jnp.log(home_rate) - home_rate - gammaln(scores + 1))
    away = jnp.exp(scores * jnp.log(away_rate) - away_rate - gammaln(scores + 1))
    grid = home[:, None] * away[None, :]
    records = []
    holdout_mask = np.asarray(holdout_data.match_mask, bool)
    for t, l in np.argwhere(holdout_mask):
        yh, ya = int(holdout_data.matches.home_score[t, l]), int(holdout_data.matches.away_score[t, l])
        records.append(_score_metrics(grid, yh, ya))
    nll = -jnp.mean(jnp.asarray([r["log_predictive_density"] for r in records]))
    return {"home_rate": home_rate, "away_rate": away_rate,
            "mean_negative_log_predictive_density": nll,
            "mean_brier_score": jnp.mean(jnp.asarray([r["brier_score"] for r in records]))}


def structural_validation(training_result, data):
    params = training_result["final_params"]
    filtered = training_result["final_filter_states"]
    augmented = training_result["final_augmented_data"]
    paths = training_result["final_smoothed_paths"]
    D, M = data.timestamp.size, params.mean_0.shape[0]
    hard = []
    if filtered.particles.x.shape[0] != D + 1:
        hard.append("filter timeline shape mismatch")
    if paths.shape[0] != D + 1 or paths.shape[-2:] != (M, 2):
        hard.append("smoother timeline shape mismatch")
    if bool(jnp.any(data.timestamp - data.timestamp_prev <= 0)):
        hard.append("non-positive elapsed time")
    pred_eig = jnp.linalg.eigvalsh(augmented.gamma_pred)
    if bool(jnp.any(pred_eig <= 0)):
        hard.append("predicted covariance is not positive definite")
    filt_eig = jnp.linalg.eigvalsh(augmented.gamma)
    if bool(jnp.any(filt_eig < -1e-6)):
        hard.append("filtered covariance is not positive semidefinite")
    finite_leaves = jax.tree.leaves((filtered, augmented, paths, params))
    if any(not np.all(np.isfinite(np.asarray(leaf))) for leaf in finite_leaves):
        hard.append("non-finite model output")
    return {"passed": not hard, "hard_failures": hard,
            "filter_shape": list(filtered.particles.x.shape),
            "smoother_shape": list(paths.shape), "n_states": D + 1,
            "n_transitions": D, "gamma_pred_min_eigenvalue": jnp.min(pred_eig),
            "gamma_filtered_psd_margin": jnp.min(filt_eig)}


def evaluate_run(training_result, train_data, holdout_data=None, *, seed=0,
                 output_dir=None):
    params = training_result["final_params"]
    structural = structural_validation(training_result, train_data)
    smoother_proxy = type("Smooth", (), {
        "particles": type("Particles", (), {"x": training_result["final_smoothed_paths"]})(),
        "backward_probabilities": training_result["backward_probabilities"],
        "component_indices": training_result["backward_component_indices"],
    })()
    smoother = smoothed_path_diagnostics(params, smoother_proxy,
                                         training_result["final_augmented_data"])
    backward = backward_metrics(training_result["backward_probabilities"],
                                training_result["backward_component_indices"])
    covariance = parameter_diagnostics(params)
    warnings = []
    if float(smoother["transition_mahalanobis_ratio"]) > 10:
        warnings.append("transition Mahalanobis ratio is in the known failure regime")
    if float(backward["ess_mean"]) < 1.5:
        warnings.append("backward component weights have low diversity")
    if float(covariance["gamma_condition_number"]) > 1e8:
        warnings.append("gamma_0 is ill-conditioned")
    predictive = ({"available": False, "reason": "no holdout data supplied"}
                  if holdout_data is None else
                  rolling_origin_predictive_evaluation(
                      training_result, holdout_data,
                      max_goals=int(training_result.get("config", {}).get("max_goals", 8)),
                      seed=seed,
                  ))
    baseline = ({"available": False} if holdout_data is None else
                constant_poisson_baseline(train_data, holdout_data,
                    int(training_result.get("config", {}).get("max_goals", 8))))
    initial_baseline = {"available": False}
    if holdout_data is not None and training_result.get("params_history"):
        initial_params = training_result["params_history"][0]
        initial_filter, initial_augmented = run_filter(
            jax.random.key(seed + 101), train_data, initial_params,
            training_result["final_filter_states"].particles.x.shape[1],
            int(training_result.get("config", {}).get("max_goals", 8)),
        )
        initial_result = {**training_result, "final_params": initial_params,
                          "final_filter_states": initial_filter,
                          "final_augmented_data": initial_augmented}
        initial_baseline = rolling_origin_predictive_evaluation(
            initial_result, holdout_data,
            max_goals=int(training_result.get("config", {}).get("max_goals", 8)),
            seed=seed + 102,
        )
    if predictive.get("available") and (
        float(predictive["mean_negative_log_predictive_density"])
        >= float(baseline["mean_negative_log_predictive_density"])
    ):
        warnings.append("fitted model did not beat the constant-rate baseline log score")
    report = {
        "configuration": training_result.get("config", {}), "seed": seed,
        "passed": structural["passed"], "promotion_ready": structural["passed"] and not warnings,
        "hard_failures": structural["hard_failures"], "warnings": warnings,
        "structural": structural, "objective": training_result.get("mstep_history", []),
        "covariance": covariance, "smoother": smoother, "backward": backward,
        "predictive": predictive, "baselines": {
            "constant_independent_poisson": baseline, "initial_model": initial_baseline},
        "final_log_marginal_likelihood": training_result["final_log_marginal_likelihood"],
    }
    if output_dir is not None:
        write_evaluation_artifacts(report, output_dir)
    return report


def _finite_json(value):
    converted = tree_to_python(value)
    def check(item):
        if isinstance(item, dict):
            return all(check(v) for v in item.values())
        if isinstance(item, list):
            return all(check(v) for v in item)
        return not isinstance(item, float) or math.isfinite(item)
    if not check(converted):
        raise ValueError("evaluation report contains non-finite numeric values")
    return converted


def write_evaluation_artifacts(report, output_dir):
    """Write strict JSON metrics and the required compact diagnostic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    clean = _finite_json(report)
    (target / "evaluation_summary.json").write_text(json.dumps(clean, indent=2))
    baseline = clean.get("baselines", {"constant_independent_poisson": {"available": False}})
    (target / "baseline_comparison.json").write_text(json.dumps(baseline, indent=2))
    for index, name in enumerate(REQUIRED_PLOTS):
        figure, axis = plt.subplots(figsize=(5, 3))
        objective = clean.get("objective", [])
        predictive = clean.get("predictive", {})
        if name == "objective_terms_by_epoch.png" and objective:
            for term in ("initial", "transition", "observation", "prior"):
                axis.plot([row[term] for row in objective], marker="o", label=term)
            axis.legend(fontsize=7)
        elif name == "transition_normalization_vs_quadratic.png" and objective:
            for term in ("transition_normalization", "transition_quadratic_penalty"):
                axis.plot([row[term] for row in objective], marker="o", label=term)
            axis.legend(fontsize=7)
        elif name == "covariance_eigenvalues_and_condition.png":
            c = clean["covariance"]
            axis.bar(["min eig", "max eig", "cond"],
                     [c["gamma_min_eigenvalue"], c["gamma_max_eigenvalue"],
                      c["gamma_condition_number"]])
            axis.set_yscale("log")
        elif name == "ou_half_life_and_parameters.png":
            c = clean["covariance"]
            axis.bar(["kappa", "half-life", "alpha", "beta"],
                     [c["kappa"], c["ou_half_life"], c["alpha"], c["beta"]])
        elif name == "backward_ess_entropy_and_unique_indices.png":
            b = clean["backward"]
            axis.plot(b["unique_indices_by_time"], marker="o", label="unique")
            axis.axhline(b["ess_mean"], color="tab:orange", label="mean ESS")
            axis.legend(fontsize=7)
        elif name == "heldout_log_score_by_date.png" and predictive.get("available"):
            axis.plot([-row["log_predictive_density"] for row in predictive["matches"]],
                      marker="o")
            axis.set_ylabel("negative log score")
        elif name == "result_calibration.png" and predictive.get("available"):
            probs = np.asarray([row["hda_probabilities"] for row in predictive["matches"]])
            axis.bar(["home", "draw", "away"], probs.mean(axis=0))
        elif name == "goal_marginal_calibration.png" and predictive.get("available"):
            axis.scatter([row["expected_home_goals"] for row in predictive["matches"]],
                         [row["home_score"] for row in predictive["matches"]], label="home")
        elif name == "smoothed_team_trajectories_with_intervals.png":
            means = np.asarray(clean["smoother"]["smoothed_mean"])
            axis.plot(means[:, 0, 0], label="team 0 attack")
            axis.legend(fontsize=7)
        else:
            ratio = clean["smoother"]["transition_mahalanobis_ratio"]
            axis.axhline(ratio, label="transition ratio")
            axis.legend(fontsize=7)
        axis.set_title(name.removesuffix(".png").replace("_", " "))
        axis.set_xlabel("index")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(target / name, dpi=100)
        plt.close(figure)
    return target
