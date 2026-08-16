import jax

from rbpf_ou_v2.src.utils import EMParams, FootballResults, RBPFFootballResults
import jax.numpy as jnp
import json
import os
import pandas as pd


_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Regional-correlation config files. ``worldcup2026_team_regions.json`` covers
# only the 48 World Cup teams; ``active_teams.json`` also carries a complete
# ``regions`` map for all ~228 ACTIVE national teams. The regional prior is
# only meaningful when *every* team has a region, so we auto-select the config
# with the best coverage of the requested team set (preferring full coverage).
TEAM_CORRELATION_PATH = os.path.join(_DATA_DIR, "worldcup2026_team_regions.json")
ACTIVE_TEAMS_REGION_PATH = os.path.join(_DATA_DIR, "active_teams.json")
_REGION_CONFIG_CANDIDATES = [TEAM_CORRELATION_PATH, ACTIVE_TEAMS_REGION_PATH]


# Smallest eigenvalue floor for reconstructed / jittered covariance matrices.
# Must match train.py's ``_EIGEN_FLOOR`` so every PSD path uses the same scale.
#
# float32 epsilon is ~1e-7. If a matrix's smallest eigenvalue is near that
# (e.g. ~1e-8, which is what ``_psd_from_cholesky``'s 1e-4 *diagonal* floor on
# ``L`` leaves after ``L L^T`` squares it), then ``jnp.linalg.cholesky`` sees a
# matrix that is effectively singular in float32 and returns NaN — and the GPU
# (which computes in float32 by default) is far more likely to hit this than a
# CPU XLA run with higher intermediate precision. Flooring eigenvalues at 1e-4
# keeps the condition number <= ~1e4, which Cholesky factors robustly.
_EIGEN_FLOOR = 1e-4


def _scale_aware_jitter(A: jnp.ndarray) -> jnp.ndarray:
    """A scale-aware diagonal jitter for a (near-)PSD matrix ``A``.

    ``jitter = _EIGEN_FLOOR * max(1, max|diag(A)|)``. Adding ``jitter * I`` to a
    (near-)PSD matrix makes its smallest eigenvalue >= jitter, guaranteeing a
    condition number that ``jnp.linalg.cholesky`` can factor robustly in float32.
    The floor is built with differentiable ops only (no eigendecomposition), so
    it is safe to use *inside* the differentiable filter path.
    """
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(jnp.diag(A))))
    return _EIGEN_FLOOR * scale


def kron_sample_psd(key, mean, A, B):
    """Sample from N(mean, A (x) B) without forming A (x) B.

    mean: (M*K,) flattened vec_C of an (M, K) matrix. A: (M, M), B: (K, K).
    Returns (M*K,).

    Uses a differentiable Cholesky reparameterization: ``X = mean + L_A Z L_B^T``
    where ``L_A``/``L_B`` are Cholesky factors of ``A``/``B`` (each jittered to
    be strictly positive-definite so the Cholesky gradient is finite). This
    keeps the gradient through ``A``/``B`` well-defined, which the direct-GD
    trainer needs to learn ``gamma_0``/``B``. The jitter is small enough that
    zero-variance directions stay essentially at the mean.

    The jitter is **scale-aware** (``_EIGEN_FLOOR * max(1, max|diag|)``) rather
    than the fixed ``1e-6`` it historically used. At large team counts (e.g.
    228 ACTIVE_TEAMS) ``gamma_0`` reconstructed from ``_psd_from_cholesky`` has
    eigenvalues as small as ``1e-4**2 = 1e-8``, and a ``1e-6`` jitter was too
    small to keep the ``M x M`` Cholesky positive-definite in GPU float32,
    producing a NaN gradient at step 0 (finite on CPU, NaN on GPU).
    """
    M = A.shape[0]
    K = B.shape[0]
    mean_MK = mean.reshape(M, K)
    A = 0.5 * (A + A.T) + _scale_aware_jitter(A) * jnp.eye(M)
    B = 0.5 * (B + B.T) + _scale_aware_jitter(B) * jnp.eye(K)
    L_A = jnp.linalg.cholesky(A)
    L_B = jnp.linalg.cholesky(B)
    z = jax.random.normal(key, (M, K))
    X = L_A @ z @ L_B.T
    return (mean_MK + X).reshape(-1)


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
        kappa=d["kappa"],
        alpha=d["alpha"],
        beta=d["beta"],
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
    return params_from_dict(d)


def _load_regional_config(path: str) -> tuple[dict[str, list[str]], float, float]:
    """Load a regional config file and return (region->teams, within, between)."""
    with open(path, "r") as f:
        config = json.load(f)
    regions = config["regions"]
    region_by_team = {
        team: region
        for region, teams in regions.items()
        for team in teams
    }
    if len(region_by_team) != sum(len(teams) for teams in regions.values()):
        raise ValueError(f"{path}: a team appears in more than one regional group")
    correlation = config["correlation"]
    return regions, float(correlation["within_region"]), float(correlation["between_regions"])


def _pick_regional_config(
    team_id_to_name: dict[int, str],
    candidates: list[str] = _REGION_CONFIG_CANDIDATES,
) -> tuple[dict[str, list[str]], float, float]:
    """Pick the regional config with the best coverage of the requested teams.

    The regional prior is only meaningful when every team has an assigned
    region. Among the candidate config files we choose the one that covers the
    most of the requested ``team_id_to_name`` (falling back to the first config
    in a tie, e.g. the 48-team file when the team set is a subset of it). This
    makes ``active_teams.json``'s full ~228-team region map the natural choice
    for ACTIVE_TEAMS runs instead of silently dropping 182 teams to the baseline
    between-region correlation.
    """
    team_names = {team_id_to_name[i] for i in range(len(team_id_to_name))}
    best: tuple[str, dict[str, list[str]], float, float] | None = None
    best_missing: int | None = None
    for path in candidates:
        regions, within, between = _load_regional_config(path)
        covered = set().union(*(set(teams) for teams in regions.values()))
        missing = team_names - covered
        if best is None or best_missing is None or len(missing) < best_missing:
            best = (path, regions, within, between)
            best_missing = len(missing)
        if best_missing == 0:
            break  # full coverage; no better config exists
    assert best is not None, "no regional config candidates provided"
    return best[1], best[2], best[3]


def _regional_correlation_matrix(
    team_id_to_name: dict[int, str],
    path: str | None = None,
) -> jnp.ndarray:
    """Build a team correlation matrix from a regional JSON specification.

    Returns the correlation matrix ``C0`` (diagonal 1). The absolute variance
    scale is not part of the prior: ``gamma_0`` is initialized at the correlation
    scale and the model estimates the true covariance from the data.

    If ``path`` is given, that config is used verbatim; otherwise the config
    with the best coverage of ``team_id_to_name`` is selected automatically
    (see ``_pick_regional_config``).
    """
    if path is not None:
        regions, within, between = _load_regional_config(path)
    else:
        regions, within, between = _pick_regional_config(team_id_to_name)
    region_by_team = {
        team: region
        for region, teams in regions.items()
        for team in teams
    }

    team_names = [team_id_to_name[i] for i in range(len(team_id_to_name))]
    # Teams not present in the regional config (e.g. defunct/non-FIFA sides)
    # simply keep the baseline "between regions" correlation.
    missing = [name for name in team_names if name not in region_by_team]
    if missing:
        print(
            f"Warning: {len(missing)} teams not in regional config; "
            f"using baseline between-region correlation: {missing}"
        )

    C0 = jnp.full((len(team_names), len(team_names)), between)
    C0 = C0.at[jnp.diag_indices(len(team_names))].set(1.0)
    for i, name_i in enumerate(team_names):
        region_i = region_by_team.get(name_i)
        if region_i is None:
            continue
        for j, name_j in enumerate(team_names):
            if i != j and region_i == region_by_team.get(name_j):
                C0 = C0.at[i, j].set(within)
    return C0


def _project_psd_correlation(C: jnp.ndarray) -> jnp.ndarray:
    """Project a symmetric matrix onto the positive-definite cone and renormalize
    to a correlation matrix (diagonal 1).

    A correlation matrix must be positive-semidefinite (all eigenvalues >= 0).
    A strongly negative ``between_regions`` value (e.g. -0.3 with 6 regions)
    can make the raw regional matrix indefinite (min eigenvalue < 0), which
    breaks the Cholesky factorization used by the filter. This clamps negative
    eigenvalues to a small **positive** floor ``_EIGEN_FLOOR`` (not 0), so the
    result is strictly positive-definite and stays PD even after float32
    rounding during the diagonal renormalization (clamping to exactly 0.0 can
    leave a tiny negative eigenvalue, e.g. -1.5e-6, which re-triggers the
    Cholesky instability). Then rescales each row/column so the diagonal is 1.
    """
    C = 0.5 * (C + C.T)  # symmetrize
    eigvals, eigvecs = jnp.linalg.eigh(C)
    eigvals = jnp.maximum(eigvals, _EIGEN_FLOOR)  # strictly PD (floor > 0)
    C_psd = (eigvecs * eigvals) @ eigvecs.T
    # Renormalize the diagonal to 1 (correlation scale).
    d = jnp.sqrt(jnp.diag(C_psd))
    d = jnp.maximum(d, 1e-8)  # avoid division by zero
    C_corr = C_psd / jnp.outer(d, d)
    C_corr = 0.5 * (C_corr + C_corr.T)
    return C_corr


def default_init_params(
    num_teams: int,
    team_id_to_name: dict[int, str] | None = None,
) -> EMParams:
    """Generate default initial parameters for the OU (scalar-phi AR(1)) model.

    The covariance is Kronecker-structured ``Sigma = gamma (x) B`` with a
    *shared* attack/defence factor ``B``. We build:

    - ``gamma_0``: team covariance initialized as the regional *correlation*
      matrix ``C0`` (diagonal 1), projected onto the PSD cone so it is always
      a valid correlation matrix. The absolute variance scale is unidentifiable
      with ``scale`` (fixed at 1), so we initialize ``gamma_0`` at the
      correlation scale and let the model estimate the true covariance from
      the data.
    - ``B``:       shared ``2 x 2`` attack/defence covariance.
    - ``kappa``:   scalar mean-reversion rate of the OU transition.
    """
    rho_team = 0.03
    if team_id_to_name is not None:
        if len(team_id_to_name) != num_teams:
            raise ValueError("team_id_to_name must contain exactly num_teams entries")
        C0 = _regional_correlation_matrix(team_id_to_name)
    else:
        C0 = (
            (1.0 - rho_team) * jnp.eye(num_teams)
            + rho_team * jnp.ones((num_teams, num_teams))
        )
    # gamma_0 starts as the correlation matrix (diagonal 1), projected onto the
    # PSD cone so it is always a valid correlation matrix; the model estimates
    # the true covariance scale from the data.
    gamma_0 = _project_psd_correlation(C0)

    cov_attack_defence = 0.2
    B = jnp.array([
        [1.0, cov_attack_defence],
        [cov_attack_defence, 1.0],
    ])

    return EMParams(
        mean_0=jnp.zeros((num_teams, 2)),
        gamma_0=gamma_0,
        B=B,
        kappa=0.01,
        alpha=0.2,
        beta=-4.0,
    )


def to_jax_data(df: pd.DataFrame) -> FootballResults:
    """
    Convert a pandas DataFrame to a FootballResults named tuple with jax arrays.
    """
    return FootballResults(
        match_index_id=jax.numpy.array(df.index.values),
        timestamp=jax.numpy.array(df["timestamp"].values),
        timestamp_prev=jax.numpy.array(df["timestamp_prev"].values),
        home_team_id=jax.numpy.array(df["home_team_id"].values),
        away_team_id=jax.numpy.array(df["away_team_id"].values),
        home_score=jax.numpy.array(df["home_score"].values),
        away_score=jax.numpy.array(df["away_score"].values),
    )


def generate_augmented_data(
    model_inputs: FootballResults,
    gamma_updated: jnp.ndarray,
    gamma_pred: jnp.ndarray,
    kalman_gain: jnp.ndarray,
) -> RBPFFootballResults:
    """
    Generate augmented data for the RBPF model, including the deterministic
    team-covariance trajectory (filtered posterior, prediction, and Kalman gain).
    """
    return RBPFFootballResults(
        match_index_id=model_inputs.match_index_id,
        timestamp=model_inputs.timestamp,
        timestamp_prev=model_inputs.timestamp_prev,
        home_team_id=model_inputs.home_team_id,
        away_team_id=model_inputs.away_team_id,
        home_score=model_inputs.home_score,
        away_score=model_inputs.away_score,
        gamma_t=gamma_updated,
        gamma_pred_t=gamma_pred,
        kalman_gain_t=kalman_gain,
    )
