"""Smoke tests for the NumPy-only simulation core."""

from dataclasses import replace

import numpy as np
import pytest

from dynamic_pricing_lab import Scenario, oracle_cum, simulate, simulate_fixed

TINY = Scenario(T=60, n_reps=4, sw_window=20)


def test_simulate_shapes_and_keys():
    out = simulate(TINY)
    assert out["revenue"].shape == (TINY.n_reps, TINY.T)
    assert out["arms"].shape == (TINY.n_reps, TINY.T)
    assert out["price_freq"].shape == (TINY.n_reps, TINY.k)
    # exactly one offer per period and replicate
    assert out["price_freq"].sum(axis=1).tolist() == [TINY.T] * TINY.n_reps
    assert np.isfinite(out["revenue"]).all()
    n_snaps = len(out["snap_periods"])
    assert out["snap_mean"].shape == (n_snaps, TINY.k)
    assert out["snap_sd"].shape == (n_snaps, TINY.k)
    assert out["snap_mix"].shape == (n_snaps, TINY.k)


def test_same_seed_is_deterministic():
    a, b = simulate(TINY), simulate(TINY)
    np.testing.assert_array_equal(a["revenue"], b["revenue"])
    np.testing.assert_array_equal(a["arms"], b["arms"])


@pytest.mark.parametrize("policy", ["ts", "sw_ts", "discounted_ts", "decay_aware_ts"])
def test_every_policy_runs(policy):
    sc = replace(TINY, policy=policy, ad_A=2.0, ad_theta=0.05)
    assert np.isfinite(simulate(sc)["revenue"]).all()


@pytest.mark.parametrize("dist", ["Bernoulli", "Normal"])
@pytest.mark.parametrize("patience", [0.0, 0.95])
def test_demand_models_and_patience(dist, patience):
    sc = replace(TINY, dist=dist, patience=patience)
    assert np.isfinite(simulate(sc)["revenue"]).all()


def test_ad_cost_charged_up_front():
    # With zero efficiency the WTP lift vanishes, so the only effect of
    # advertising is the one-time A**2 cost in the first period.
    base = simulate(TINY)
    ads = simulate(replace(TINY, ad_A=3.0, ad_theta=0.0))
    np.testing.assert_allclose(
        ads["revenue"][:, 0], base["revenue"][:, 0] - 9.0, atol=1e-4
    )
    np.testing.assert_array_equal(ads["revenue"][:, 1:], base["revenue"][:, 1:])


def test_fixed_price_and_oracle():
    assert simulate_fixed(TINY, arm=0).shape == (TINY.n_reps, TINY.T)
    oracle = oracle_cum(TINY)
    assert oracle.shape == (TINY.T,)
    # cumulative revenue of a fixed policy never decreases after t = 0
    assert (np.diff(oracle) >= -1e-6).all()
