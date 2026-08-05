"""Unit tests for the matplotlib figure builders (need the dashboard extra)."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from dynamic_pricing_lab import charts  # noqa: E402

T, S, K = 30, 5, 3
PRICES = (0.5, 0.8, 0.9)
DEMANDS = (0.5, 0.2, 0.1)


def _condition(rng):
    cum = np.cumsum(rng.random(T))
    return {
        "cum_mean": cum,
        "cum_lo": cum - 1.0,
        "cum_hi": cum + 1.0,
        "regret": np.linspace(0.0, 5.0, T),
        "share": np.array([70.0, 20.0, 10.0]),
        "snap_periods": np.linspace(1, T, S).astype(int),
        "snap_mean": rng.random((S, K)),
        "snap_sd": 0.1 * rng.random((S, K)),
        "snap_mix": rng.random((S, K)),
    }


@pytest.fixture
def results():
    rng = np.random.default_rng(0)
    return {name: _condition(rng) for name in charts.STYLE}


def test_comparison_figures(results):
    for fig in (
        charts.cum_revenue(results),
        charts.regret(results),
        charts.shares(results, PRICES),
    ):
        assert fig.axes, "figure has no axes"
        charts.plt.close(fig)


def test_single_scenario_figures(results):
    r = results["Myopic + ads"]
    for fig in (
        charts.posterior(r, PRICES, DEMANDS, bernoulli=True),
        charts.posterior(r, PRICES, DEMANDS, bernoulli=False),
        charts.policy_mix(r, PRICES),
    ):
        assert fig.axes
        charts.plt.close(fig)


def test_sweep_figure():
    grid = np.arange(0.0, 5.0, 1.0)
    n = len(grid)
    curves = {
        "Forward-looking": ([1.0] * n, [0.1] * n, charts.NAVY),
        "Myopic": ([0.5] * n, [0.1] * n, charts.BRICK),
    }
    fig = charts.sweep(grid, curves)
    assert fig.axes
    charts.plt.close(fig)
