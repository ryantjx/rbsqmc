from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rbpf_v3_pytest_matplotlib")
os.environ.setdefault("RBSQMC_PLATFORM", "cpu")

import jax
import pandas as pd
import pytest

from rbpf_v3.src.helpers import default_init_params, generate_results_jax
from rbpf_v3.src.model import run_filter


@pytest.fixture(scope="session")
def small_problem():
    rows = []
    for timestamp, date, scores in zip(
        (1, 3, 8, 10),
        pd.to_datetime(("2024-01-01", "2024-01-03", "2024-01-08", "2024-01-10")),
        ((1, 0), (0, 0), (2, 1), (1, 2)),
    ):
        rows.append(
            {
                "date": date,
                "timestamp": timestamp,
                "home_team": "A",
                "away_team": "B",
                "home_id": 0,
                "away_id": 1,
                "home_score": scores[0],
                "away_score": scores[1],
                "tournament": "Test",
            }
        )
    _, data = generate_results_jax(pd.DataFrame(rows))
    params = default_init_params(2)
    filtered, augmented = run_filter(jax.random.key(10), data, params, 8, 8)
    return data, params, filtered, augmented
