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
    # cov_attack_defence = 0.2
    # B = jnp.array([
    #     [1.0,  cov_attack_defence],
    #     [cov_attack_defence,  1.0],
    # ])
    B = jnp.eye(2)  # independent attack/defence evolution
    return EMParams(
        mean_0=jnp.zeros((num_teams, 2)),
        gamma_0=gamma_0,
        B=B,
        kappa=jnp.array(0.01),
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

def decode_diagonal_B(B_ratio_raw):
    log_ratio = MAX_B_LOG_RATIO * jnp.tanh(B_ratio_raw)

    return jnp.diag(
        jnp.array([
            jnp.exp(log_ratio),
            jnp.exp(-log_ratio),
        ])
    )


def encode_EM_params(params: EMParams) -> RawEMParams:
    """Encode identified EM parameters into unconstrained parameters."""
    num_teams = params.mean_0.shape[0]

    attack_variance = params.B[0, 0]
    defence_variance = params.B[1, 1]

    # Overall scale of diagonal B. Move it into Gamma_0 so det(B) = 1.
    B_scale = jnp.sqrt(attack_variance * defence_variance)
    gamma_0_identified = B_scale * params.gamma_0

    log_ratio = 0.5 * jnp.log(
        attack_variance / defence_variance
    )
    bounded_ratio = jnp.clip(
        log_ratio / MAX_B_LOG_RATIO,
        -1.0 + 1e-6,
        1.0 - 1e-6,
    )
    B_ratio_raw = jnp.arctanh(bounded_ratio)

    kappa_raw = inverse_softplus(
        params.kappa - KAPPA_FLOOR
    )

    return RawEMParams(
        gamma_0_chol=_cholesky_from_psd(
            gamma_0_identified,
            num_teams,
        ),
        B_ratio_raw=B_ratio_raw,
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
        B=decode_diagonal_B(raw_params.B_ratio_raw),
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
