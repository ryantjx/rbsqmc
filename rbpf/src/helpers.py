import jax

from rbpf.src.utils import EMParams, FootballResults, RBPFFootballResults, Matches, RawEMParams
import jax.numpy as jnp
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rbpf.src.graphic import plot_log_marginal_likelihood_curve
from optax import Params

TEAM_CORRELATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "worldcup2026_team_regions.json"
)

def save_em_results(
    results: dict,
    output_dir: str,
):
    """Persist the run_EM results to JSON files in ``output_dir``.

    ``results`` is the dict returned by ``run_EM``. Writes:
      - ``em_final_params.json``: the final decoded parameters.
      - ``em_params_history.json``: per-epoch parameter trajectory.
      - ``em_log_marginal_history.json``: likelihoods for ``theta_0`` through
        the final ``theta_K``.
      - ``em_mstep_history.json``: per-epoch M-step diagnostics.
      - ``em_diagnostics_history.json``: RBPF representation, transition and
        covariance diagnostics, when present.
      - ``em_run_metadata.json``: run configuration and final likelihood,
        when present.
    """
    os.makedirs(output_dir, exist_ok=True)

    final_params = results["final_params"]
    with open(os.path.join(output_dir, "em_final_params.json"), "w") as f:
        json.dump(params_to_dict(final_params), f, indent=2)

    params_history = results["params_history"]
    with open(os.path.join(output_dir, "em_params_history.json"), "w") as f:
        json.dump(
            [params_to_dict(p) for p in params_history],
            f,
            indent=2,
        )

    log_marginal_history = results["log_marginal_history"]
    with open(os.path.join(output_dir, "em_log_marginal_history.json"), "w") as f:
        json.dump(
            [float(x) for x in log_marginal_history],
            f,
            indent=2,
        )

    mstep_history = results["mstep_history"]
    with open(os.path.join(output_dir, "em_mstep_history.json"), "w") as f:
        json.dump(to_jsonable(mstep_history), f, indent=2)

    if "diagnostics_history" in results:
        with open(
            os.path.join(output_dir, "em_diagnostics_history.json"), "w"
        ) as f:
            json.dump(to_jsonable(results["diagnostics_history"]), f, indent=2)

    metadata = dict(results.get("run_metadata", {}))
    if "final_log_marginal_likelihood" in results:
        metadata["final_log_marginal_likelihood"] = results[
            "final_log_marginal_likelihood"
        ]
    if metadata:
        with open(os.path.join(output_dir, "em_run_metadata.json"), "w") as f:
            json.dump(to_jsonable(metadata), f, indent=2)

    print(f"Saved EM results to {os.path.abspath(output_dir)}")


def to_jsonable(value):
    """Recursively convert JAX/NumPy diagnostic values into JSON values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def timeline_diagnostics(model_inputs: FootballResults) -> dict:
    """Summarize the observation timeline and identify invalid transitions."""
    dt = jnp.asarray(model_inputs.timestamp - model_inputs.timestamp_prev)
    invalid = jnp.where(dt <= 0)[0]
    return {
        "n_transitions": dt.size,
        "dt_min": jnp.min(dt),
        "dt_median": jnp.median(dt),
        "dt_mean": jnp.mean(dt),
        "dt_max": jnp.max(dt),
        "n_dt_le_1": jnp.sum(dt <= 1),
        "n_dt_le_2": jnp.sum(dt <= 2),
        "n_dt_le_7": jnp.sum(dt <= 7),
        "invalid_dt_indices": invalid,
        "all_dt_positive": jnp.all(dt > 0),
    }


def parameter_diagnostics(
    params: EMParams,
    representative_dt: tuple[int, ...] = (1, 2, 7, 30),
) -> dict:
    """Compute covariance/OU diagnostics for one EM parameter value."""
    gamma = 0.5 * (params.gamma_0 + params.gamma_0.T)
    B = 0.5 * (params.B + params.B.T)

    gamma_eigenvalues = jnp.linalg.eigvalsh(gamma)
    B_eigenvalues = jnp.linalg.eigvalsh(B)
    gamma_min = gamma_eigenvalues[0]
    gamma_max = gamma_eigenvalues[-1]
    gamma_trace = jnp.sum(gamma_eigenvalues)
    eigenvalue_weights = jnp.maximum(gamma_eigenvalues, 0.0) / gamma_trace
    effective_rank = jnp.exp(
        -jnp.sum(
            jnp.where(
                eigenvalue_weights > 0,
                eigenvalue_weights * jnp.log(eigenvalue_weights),
                0.0,
            )
        )
    )

    gamma_sign, gamma_logdet = jnp.linalg.slogdet(gamma)
    B_sign, B_logdet = jnp.linalg.slogdet(B)
    condition_number = jnp.where(
        gamma_min > 0,
        gamma_max / gamma_min,
        jnp.inf,
    )

    q_eigenvalues = {}
    full_base_min = gamma_min * B_eigenvalues[0]
    full_base_max = gamma_max * B_eigenvalues[-1]
    for delta in representative_dt:
        variance_scale = 1.0 - jnp.exp(-2.0 * params.kappa * delta)
        q_eigenvalues[str(delta)] = {
            "variance_scale": variance_scale,
            "min_eigenvalue": variance_scale * full_base_min,
            "max_eigenvalue": variance_scale * full_base_max,
        }

    return {
        "gamma_min_eigenvalue": gamma_min,
        "gamma_max_eigenvalue": gamma_max,
        "gamma_condition_number": condition_number,
        "gamma_trace": gamma_trace,
        "gamma_logdet": gamma_logdet,
        "gamma_determinant_sign": gamma_sign,
        "gamma_effective_rank": effective_rank,
        "gamma_diagonal_min": jnp.min(jnp.diag(gamma)),
        "gamma_diagonal_median": jnp.median(jnp.diag(gamma)),
        "gamma_diagonal_max": jnp.max(jnp.diag(gamma)),
        "B_min_eigenvalue": B_eigenvalues[0],
        "B_max_eigenvalue": B_eigenvalues[-1],
        "B_logdet": B_logdet,
        "B_determinant_sign": B_sign,
        "kappa": params.kappa,
        "ou_half_life": jnp.log(2.0) / params.kappa,
        "alpha": params.alpha,
        "beta": params.beta,
        "Q_eigenvalues_by_dt": q_eigenvalues,
    }


def _require_finite(params: EMParams, prefix: str = ""):
    """Raise if any EM parameter leaf is NaN or infinite."""
    for name, value in params._asdict().items():
        arr = jnp.asarray(value)
        if not bool(jnp.all(jnp.isfinite(arr))):
            raise RuntimeError(
                f"{prefix}[monitor_params] Parameter '{name}' is not finite:\n"
                f"{arr}"
            )


def _validate_diagnostics(diagnostics: dict, prefix: str = ""):
    """Raise if any monitored parameter falls outside a healthy range.

    Thresholds are conservative and chosen to catch diverging/ill-conditioned
    EM runs early (typically the cause of a flat log-marginal-likelihood).
    """
    def fail(metric: str, value, reason: str):
        raise RuntimeError(
            f"{prefix}[monitor_params] Bad parameter value: {metric}={value:.6g} "
            f"({reason}). EM has likely diverged."
        )

    # --- attribute covariance B (should be SPD, det = 1) ---
    B_min = diagnostics["B_min_eigenvalue"]
    B_max = diagnostics["B_max_eigenvalue"]
    if B_min <= 0:
        fail("B_min_eigenvalue", B_min, "B is not positive-definite")
    if B_max > 1e3:
        fail("B_max_eigenvalue", B_max, "B eigenvalue exploded")
    if B_min < 1e-3:
        fail("B_min_eigenvalue", B_min, "B is near-singular")
    if not (-1.0 <= diagnostics["B_attack_defence_corr"] <= 1.0):
        fail(
            "B_attack_defence_corr",
            diagnostics["B_attack_defence_corr"],
            "correlation outside [-1, 1]",
        )

    # --- team covariance gamma_0 (should be SPD) ---
    gamma_min = diagnostics["gamma_min_eigenvalue"]
    gamma_max = diagnostics["gamma_max_eigenvalue"]
    if gamma_min <= 0:
        fail("gamma_min_eigenvalue", gamma_min, "gamma_0 is not positive-definite")
    if gamma_max > 1e4:
        fail("gamma_max_eigenvalue", gamma_max, "gamma_0 eigenvalue exploded")
    if gamma_min < 1e-6:
        fail("gamma_min_eigenvalue", gamma_min, "gamma_0 is near-singular")

    # --- OU dynamics ---
    kappa = diagnostics["kappa"]
    if kappa < 0:
        fail("kappa", kappa, "negative mean-reversion")
    if kappa > 10.0:
        fail("kappa", kappa, "kappa exploded (OU collapses to white noise)")

    # --- observation parameters ---
    alpha = diagnostics["alpha"]
    if alpha < 0:
        fail("alpha", alpha, "negative Poisson rate/scale")
    if alpha > 100.0:
        fail("alpha", alpha, "alpha exploded")


def monitor_params(
    params: EMParams,
    prefix: str = "",
    verbose: bool = True,
    validate: bool = True,
) -> dict:
    """Compute and (optionally) print a readable summary of all EM parameters.

    Returns a flat dict of scalar diagnostics covering every optimized
    parameter: the attribute covariance ``B`` (including its attack/defence
    correlation), the team-covariance ``gamma_0`` (trace, condition number and
    effective rank), the OU ``kappa`` / half-life, and the observation
    parameters ``alpha``/``beta``.

    Args:
        params: the EM parameters to inspect.
        prefix: string prepended to each printed line (e.g. epoch indentation).
        verbose: if True, print the diagnostics lines.
        validate: if True, raise ``RuntimeError`` as soon as any parameter
            drifts into an extreme/invalid region (NaN, non-PD covariance,
            degenerate conditioning, out-of-range correlation, etc.). This is
            useful for aborting an EM run early instead of silently returning a
            flat likelihood.
    """
    # NaN / non-finite guard first so downstream checks are meaningful.
    _require_finite(params, prefix)
    gamma = 0.5 * (params.gamma_0 + params.gamma_0.T)
    B = 0.5 * (params.B + params.B.T)

    gamma_evals = jnp.linalg.eigvalsh(gamma)
    B_evals = jnp.linalg.eigvalsh(B)

    gamma_min, gamma_max = float(gamma_evals[0]), float(gamma_evals[-1])
    gamma_trace = float(jnp.sum(gamma_evals))
    gamma_cond = float(gamma_max / gamma_min) if gamma_min > 0 else float("inf")

    eigenvalue_weights = jnp.maximum(gamma_evals, 0.0) / gamma_trace
    effective_rank = float(
        jnp.exp(
            -jnp.sum(
                jnp.where(
                    eigenvalue_weights > 0,
                    eigenvalue_weights * jnp.log(eigenvalue_weights),
                    0.0,
                )
            )
        )
    )

    B_min, B_max = float(B_evals[0]), float(B_evals[-1])
    B_corr = float(B[0, 1] / jnp.sqrt(B[0, 0] * B[1, 1]))
    B_cond = float(B_max / B_min) if B_min > 0 else float("inf")

    kappa = float(params.kappa)
    diagnostics = {
        # attribute covariance B
        "B_min_eigenvalue": B_min,
        "B_max_eigenvalue": B_max,
        "B_condition_number": B_cond,
        "B_logdet": float(jnp.linalg.slogdet(B)[1]),
        "B_attack_defence_corr": B_corr,
        "B_attack_variance": float(B[0, 0]),
        "B_defence_variance": float(B[1, 1]),
        # team covariance gamma_0
        "gamma_min_eigenvalue": gamma_min,
        "gamma_max_eigenvalue": gamma_max,
        "gamma_condition_number": gamma_cond,
        "gamma_trace": gamma_trace,
        "gamma_effective_rank": effective_rank,
        "gamma_logdet": float(jnp.linalg.slogdet(gamma)[1]),
        # OU dynamics
        "kappa": kappa,
        "ou_half_life": float(jnp.log(2.0) / kappa) if kappa > 0 else float("inf"),
        # observation parameters
        "alpha": float(params.alpha),
        "beta": float(params.beta),
    }

    if validate:
        _validate_diagnostics(diagnostics, prefix)

    if verbose:
        print(
            f"{prefix} kappa={kappa:.5f} half_life={diagnostics['ou_half_life']:.2f} "
            f"alpha={diagnostics['alpha']:.4f} beta={diagnostics['beta']:.4f}",
            flush=True,
        )
        print(
            f"{prefix} B: min_eig={B_min:.5f} max_eig={B_max:.5f} "
            f"cond={B_cond:.3f} corr={B_corr:+.4f} det={jnp.sqrt(B_min * B_max):.5f} "
            f"attack_var={diagnostics['B_attack_variance']:.5f} "
            f"defence_var={diagnostics['B_defence_variance']:.5f}",
            flush=True,
        )
        print(
            f"{prefix} gamma_0: min_eig={gamma_min:.5f} max_eig={gamma_max:.5f} "
            f"cond={gamma_cond:.3f} trace={gamma_trace:.5f} "
            f"eff_rank={effective_rank:.3f}",
            flush=True,
        )

    return diagnostics


def record_mstep_diagnostics(
    history: list[dict],
    epoch: int,
    complete_data_loss: jax.Array,
    log_marginal_likelihood: jax.Array,
    grad_norms: dict[str, float],
) -> None:
    """Append one epoch's M-step diagnostics to ``history``.

    Stores the complete-data loss (what the M-step actually minimizes), the
    observed-data log marginal likelihood (a noisy, lagging monitor), and the
    per-parameter gradient norms. The accumulated ``history`` can later be
    passed to ``print_mstep_summary`` to see the per-epoch trend.
    """
    history.append({
        "epoch": int(epoch),
        "complete_data_loss": float(complete_data_loss),
        "log_marginal_likelihood": float(log_marginal_likelihood),
        "grad_norms": {k: float(v) for k, v in grad_norms.items()},
    })


def print_mstep_summary(history: list[dict]) -> None:
    """Print a per-epoch trend of the M-step complete-data loss.

    Highlights whether the M-step is actually improving (loss decreasing)
    versus stalling (loss flat). If ``history`` is empty, prints a message and
    returns.
    """
    if not history:
        print("[M-step summary] No M-step diagnostics recorded.")
        return

    print("[M-step summary] Complete-data loss over epochs (lower is better):")
    for entry in history:
        loss = entry["complete_data_loss"]
        logz = entry["log_marginal_likelihood"]
        print(
            f"  epoch {entry['epoch']}: complete-data loss={loss:.4f}  "
            f"logZ={logz:.4f}",
            flush=True,
        )

    losses = [entry["complete_data_loss"] for entry in history]
    first, last = losses[0], losses[-1]
    change = last - first
    if len(losses) >= 2 and abs(first) > 0:
        rel = change / abs(first)
        if rel < -1e-4:
            print(
                f"[M-step summary] complete-data loss decreased by {change:.4f} "
                f"({rel * 100:.2f}%) — M-step is making progress.",
                flush=True,
            )
        elif rel > 1e-4:
            print(
                f"[M-step summary] WARNING: complete-data loss INCREASED by "
                f"{change:.4f} ({rel * 100:.2f}%) — the M-step is diverging.",
                flush=True,
            )
        else:
            print(
                f"[M-step summary] complete-data loss is essentially flat "
                f"(delta={change:.4f}) — the M-step is likely stalling.",
                flush=True,
            )


def resolve_teams(cfg: dict) -> set[str] | None:
    """Resolve the ``teams`` config entry to a set of team names.

    ``teams`` may be:
      - a preset name: ``"teams_small"`` | ``"worldcup2026"`` | ``"active"``
      - an explicit list of team names (e.g. ``["England", "France"]``)
      - ``"all"`` or empty/missing -> None (use all teams)
    """
    from rbpf.src.data import TEAMS_SMALL, WORLDCUP_2026_TEAMS, ACTIVE_TEAMS

    value = cfg.get("teams", "teams_small")
    presets = {
        "teams_small": TEAMS_SMALL,
        "worldcup2026": WORLDCUP_2026_TEAMS,
        "active": ACTIVE_TEAMS,
    }
    if isinstance(value, str):
        name = value.strip().lower()
        if name in presets:
            return presets[name]
        if name in ("", "all", "none"):
            return None
        raise ValueError(
            f"Unknown 'teams' preset '{value}'. Choose from {sorted(presets)} "
            "or pass an explicit list of team names."
        )
    if isinstance(value, (list, tuple, set)):
        return set(value)
    raise ValueError(f"Invalid 'teams' value in config: {value!r}")


def params_to_dict(params: EMParams) -> dict:
    """Convert EMParams to a JSON-serializable dict."""
    return {
        "mean_0": jnp.array(params.mean_0).tolist(),
        "gamma_0": jnp.array(params.gamma_0).tolist(),
        "B": jnp.array(params.B).tolist(),
        "kappa": float(params.kappa),
        "alpha": float(params.alpha),
        "beta": float(params.beta),
    }


def params_from_dict(d: dict) -> EMParams:
    """Convert a JSON dict back to EMParams."""
    return EMParams(
        mean_0=jnp.array(d["mean_0"]),
        gamma_0=jnp.array(d["gamma_0"]),
        B=jnp.array(d["B"]),
        kappa=jnp.array(d["kappa"]),
        alpha=jnp.array(d["alpha"]),
        beta=jnp.array(d["beta"]),
    )


def save_params(params: EMParams, path: str):
    """Save parameters to a JSON file."""
    with open(path, "w") as f:
        json.dump(params_to_dict(params), f, indent=2)
    print(f"Saved parameters to {path}")


def load_params(path: str) -> EMParams:
    """Load parameters from a JSON file."""
    with open(path, "r") as f:
        d = json.load(f)
    print(f"Loaded parameters from {path}")
    return params_from_dict(d)


def params_track_to_dict(params_track: EMParams) -> dict:
    """Convert a stacked ``params_track`` (leading axis = epochs) to JSON.

    Each leaf has shape ``(n_epochs, ...)``. Returns a dict of nested lists.
    """
    return {
        "mean_0": jnp.asarray(params_track.mean_0).tolist(),
        "gamma_0": jnp.asarray(params_track.gamma_0).tolist(),
        "B": jnp.asarray(params_track.B).tolist(),
        "kappa": jnp.asarray(params_track.kappa).tolist(),
        "alpha": jnp.asarray(params_track.alpha).tolist(),
        "beta": jnp.asarray(params_track.beta).tolist(),
    }


def generate_results_jax(matches: pd.DataFrame) -> tuple[pd.DataFrame, FootballResults]:
    # group by date
    date_grouped = matches.groupby("date")
    # get number of unique dates and max number of matches per day
    T = date_grouped.ngroups
    M = date_grouped.size().max()

    print(f"Generating JAX arrays for {T} unique dates and up to {M} matches per day.")

    # Use NumPy arrays for accumulation (mutable); converted to JAX at the end.
    # Pad with -1 as a sentinel: never a valid score or team id (team ids are >= 0).
    home_score = np.full((T, M), 0, dtype=np.int32)
    away_score = np.full((T, M), 0, dtype=np.int32)
    home_id = np.full((T, M), 0, dtype=np.int32)
    away_id = np.full((T, M), 0, dtype=np.int32)
    match_mask = np.zeros((T, M), dtype=bool)

    dates = []
    timestamps = []
    timestamps_prev = []

    prev_timestamp = 0
    # loop over each date group and fill in the arrays
    for t, (date, group) in enumerate(date_grouped):
        n = len(group)

        home_score[t, :n] = group["home_score"].to_numpy()
        away_score[t, :n] = group["away_score"].to_numpy()
        home_id[t, :n] = group["home_id"].to_numpy()
        away_id[t, :n] = group["away_id"].to_numpy()

        match_mask[t, :n] = True

        timestamp = group["timestamp"].iloc[0]

        dates.append(date)
        timestamps.append(timestamp)
        timestamps_prev.append(prev_timestamp)
        # update prev_timestamp for the next iteration
        prev_timestamp = timestamp

    matches_jax = Matches(
        home_score=jnp.asarray(home_score),
        away_score=jnp.asarray(away_score),
        home_id=jnp.asarray(home_id),
        away_id=jnp.asarray(away_id),
    )

    # dates are pandas Timestamps -> convert to numeric (days since start_date)
    # via np.datetime64 for JAX compatibility. DataFrame keeps the readable form.
    date_values = jnp.asarray([np.datetime64(d).astype("datetime64[D]").astype("int64")
                               for d in dates])

    results = FootballResults(
        date=date_values, # (T, ). used for reference in the future.
        timestamp=jnp.asarray(timestamps), # (T, )
        timestamp_prev=jnp.asarray(timestamps_prev), # (T, )
        matches=matches_jax, # (T, M)
        match_mask=jnp.asarray(match_mask), # (T, M) boolean mask
    )
    # per-date matches as readable dictionaries (one entry per date).
    # Distinct from `results.matches` (the JAX Matches tensor).
    matches_per_date = [
        [
            {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "tournament": row["tournament"],
            }
            for _, row in group.iterrows()
        ]
        for _, group in date_grouped
    ]

    df = pd.DataFrame(
        {
            "date": dates,
            "timestamp": timestamps,
            "timestamp_prev": timestamps_prev,
            "matches": matches_per_date,   # list of T lists of match dicts
        }
    )
    return df, results


def _regional_correlation_matrix(
        team_id_to_name: dict[int, str],
        path: str = TEAM_CORRELATION_PATH,
    ) -> tuple[jnp.ndarray, float]:
    """Build a team correlation matrix from a regional JSON specification."""
    with open(path, "r") as f:
        config = json.load(f)

    regions = config["regions"]
    region_by_team = {
        team: region
        for region, teams in regions.items()
        for team in teams
    }
    if len(region_by_team) != sum(len(teams) for teams in regions.values()):
        raise ValueError("A team appears in more than one regional group")

    team_names = [team_id_to_name[i] for i in range(len(team_id_to_name))]
    # Teams not present in the regional config (e.g. defunct/non-FIFA sides)
    # simply keep the baseline "between regions" correlation.
    missing = [name for name in team_names if name not in region_by_team]
    if missing:
        print(
            f"Warning: {len(missing)} teams not in regional config; "
            f"using baseline between-region correlation: {missing}"
        )

    correlation = config["correlation"]
    within = float(correlation["within_region"])
    between = float(correlation["between_regions"])
    state_std = float(correlation["state_standard_deviation"])

    C0 = jnp.full((len(team_names), len(team_names)), between)
    C0 = C0.at[jnp.diag_indices(len(team_names))].set(1.0)
    for i, name_i in enumerate(team_names):
        region_i = region_by_team.get(name_i)
        if region_i is None:
            continue
        for j, name_j in enumerate(team_names):
            if i != j and region_i == region_by_team.get(name_j):
                C0 = C0.at[i, j].set(within)
    return C0, state_std


def default_init_params(
        num_teams: int,
        team_id_to_name: dict[int, str] | None = None,
    ) -> EMParams:
    """Generate default initial parameters.

    When a team-name mapping is supplied, use the regional correlation prior
    from ``data/team_regions_all.json`` (active national teams). Otherwise
    retain the exchangeable correlation prior for generic datasets.
    """
    if team_id_to_name is not None:
        if len(team_id_to_name) != num_teams:
            raise ValueError("team_id_to_name must contain exactly num_teams entries")
        C0, state_std = _regional_correlation_matrix(team_id_to_name)
        sigmas = state_std * jnp.ones(num_teams)
    else:
        sigmas = 0.4 * jnp.ones(num_teams)
        rho_team = 0.03
        C0 = (
            (1.0 - rho_team) * jnp.eye(num_teams)
            + rho_team * jnp.ones((num_teams, num_teams))
        )
    D = jnp.diag(sigmas)
    gamma_0 = D @ C0 @ D
    cov_attack_defence = 0.2
    B = jnp.array([
        [1.0,  cov_attack_defence],
        [cov_attack_defence,  1.0],
    ])
    # B=jnp.eye(2)  # For now, use identity for B to avoid identifiability issues

    print(f"Default initial parameters: num_teams={num_teams}, gamma_0.shape={gamma_0.shape}, B.shape={B.shape}")
    print(f"    B: {B}")
    print(f"    mean_0: {jnp.mean(jnp.zeros((num_teams, 2)))}")
    print(f"    gamma_0: {jnp.mean(gamma_0)}")
    print(f"    kappa: {jnp.array(0.001)}")
    print(f"    alpha: {jnp.array(0.2)}")
    print(f"    beta: {jnp.array(-4.0)}")

    return EMParams(
        mean_0=jnp.zeros((num_teams, 2)),
        gamma_0=gamma_0,
        B=B,
        kappa=jnp.array(0.001),
        alpha=jnp.array(0.2),
        beta=jnp.array(-4.0),
    )

def generate_rbpf_trajectory(
    model_inputs: FootballResults,
    gamma: jnp.ndarray,
    gamma_pred: jnp.ndarray,
    gamma_observed: jnp.ndarray,
    kalman_gain: jnp.ndarray
) -> RBPFFootballResults:
    """
    Generate augmented data for the RBPF model, including the previous state and time difference.
    """
    return RBPFFootballResults(
        timestamp=model_inputs.timestamp,
        timestamp_prev=model_inputs.timestamp_prev,
        matches=model_inputs.matches,
        match_mask=model_inputs.match_mask,
        gamma=gamma,
        gamma_pred=gamma_pred,
        gamma_observed=gamma_observed,
        kalman_gain=kalman_gain
    )

KAPPA_FLOOR = 1e-6
GAMMA_CHOL_FLOOR = 1e-4
MAX_B_LOG_RATIO = 5.0
B_CHOL_FLOOR = 1e-4

def inverse_softplus(x):
    x = jnp.maximum(x, 1e-8)
    return x + jnp.log(-jnp.expm1(-x))

def _psd_from_cholesky(L, n):
    """Build PD matrix A = L L^T from a free n x n factor."""
    L_low = jnp.tril(L)
    diag = (
        jax.nn.softplus(jnp.diag(L_low))
        + GAMMA_CHOL_FLOOR
    )
    L_low = L_low.at[jnp.diag_indices(n)].set(diag)
    return L_low @ L_low.T


def _cholesky_from_psd(A, n):
    """Encode Gamma_0 into free Cholesky parameters."""
    A = 0.5 * (A + A.T)
    L = jnp.linalg.cholesky(A)

    diagonal = jnp.diag(L)

    # Inverse of:
    # diagonal = softplus(diag_raw) + GAMMA_CHOL_FLOOR
    diag_raw = inverse_softplus(
        diagonal - GAMMA_CHOL_FLOOR
    )

    L_free = jnp.tril(L)
    L_free = L_free.at[jnp.diag_indices(n)].set(diag_raw)

    return L_free

def _cholesky_from_psd_B(A, n):
    """Encode an SPD attribute covariance into free Cholesky parameters."""
    A = 0.5 * (A + A.T)
    L = jnp.linalg.cholesky(A)

    diagonal = jnp.diag(L)
    # Inverse of: diagonal = softplus(diag_raw) + B_CHOL_FLOOR
    diag_raw = inverse_softplus(diagonal - B_CHOL_FLOOR)

    L_free = jnp.tril(L)
    L_free = L_free.at[jnp.diag_indices(n)].set(diag_raw)

    return L_free


def decode_B_cholesky(b_raw):
    """Decode a free 2x2 factor into a det(B)=1 SPD attribute covariance.

    The overall scale of B is not identified (Gamma_0 (x) B), so we fix
    det(B) = 1 and absorb the true scale into Gamma_0 in encode_EM_params.
    """
    n = b_raw.shape[0]
    L = jnp.tril(b_raw)
    diag = (
        jax.nn.softplus(jnp.diag(L))
        + B_CHOL_FLOOR
    )
    L = L.at[jnp.diag_indices(n)].set(diag)

    B = L @ L.T
    return B / jnp.sqrt(jnp.linalg.det(B))


def encode_EM_params(params: EMParams) -> RawEMParams:
    """Encode identified EM parameters into unconstrained parameters."""
    num_teams = params.mean_0.shape[0]

    B = 0.5 * (params.B + params.B.T)
    # Full scale of B moves into Gamma_0 so the decoded B has det(B) = 1.
    B_scale = jnp.sqrt(jnp.linalg.det(B))
    gamma_0_identified = B_scale * params.gamma_0
    B_identified = B / B_scale

    kappa_raw = inverse_softplus(
        params.kappa - KAPPA_FLOOR
    )

    return RawEMParams(
        gamma_0_chol=_cholesky_from_psd(
            gamma_0_identified,
            num_teams,
        ),
        b_chol_raw=_cholesky_from_psd_B(B_identified, 2),
        kappa_raw=kappa_raw,
        alpha=params.alpha,
        beta=params.beta,
    )


def decode_EM_params(
    raw_params: RawEMParams,
    fixed_mean_0: jax.Array,
) -> EMParams:
    """Decode unconstrained parameters into identified EM parameters."""
    num_teams = raw_params.gamma_0_chol.shape[0]

    gamma_0 = _psd_from_cholesky(
        raw_params.gamma_0_chol,
        num_teams,
    )

    kappa = (
        jax.nn.softplus(raw_params.kappa_raw)
        + KAPPA_FLOOR
    )

    return EMParams(
        mean_0=fixed_mean_0,
        gamma_0=gamma_0,
        B=decode_B_cholesky(raw_params.b_chol_raw),
        kappa=kappa,
        alpha=raw_params.alpha,
        beta=raw_params.beta,
    )

def log_inverse_wishart_kernel(
    gamma_0: jax.Array,
    scale: jax.Array,
    dof: float,
) -> jax.Array:
    dimension = gamma_0.shape[0]
    _, logdet = jnp.linalg.slogdet(gamma_0)

    trace_term = jnp.trace(
        jnp.linalg.solve(gamma_0, scale)
    )

    return (
        -0.5 * (dof + dimension + 1.0) * logdet
        -0.5 * trace_term
    )
