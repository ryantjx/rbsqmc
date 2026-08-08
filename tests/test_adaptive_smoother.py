import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "rbpf"))

from highdim_rbpf_smoothing import adaptive_smoother


def test_adaptive_smoother_returns_constant_path_when_blend_zero():
    observations = [0.5, 0.7, 0.3, 0.9]
    smoothed = adaptive_smoother(observations, start_value=1.0, blend=0.0)

    assert smoothed == [1.0, 1.0, 1.0, 1.0]


def test_adaptive_smoother_raises_for_confidence_length_mismatch():
    with pytest.raises(ValueError, match="confidence"):
        adaptive_smoother([1.0, 2.0], start_value=0.0, confidence=[0.5])


def test_adaptive_smoother_handles_empty_input():
    assert adaptive_smoother([], start_value=0.0) == []
