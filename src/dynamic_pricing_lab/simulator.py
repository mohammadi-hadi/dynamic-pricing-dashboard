# SPDX-FileCopyrightText: 2026 Hadi Mohammadi <https://mohammadi.cv>
# SPDX-License-Identifier: MIT

"""Vectorized simulation core for the dynamic-pricing dashboard.

Thompson sampling pricing against myopic or forward-looking customers,
with an optional one-time advertising effort whose willingness-to-pay
lift decays over the season. All replicates of a scenario are simulated
simultaneously (state arrays of shape (n_reps, k)), which keeps the
dashboard responsive even when it runs in the browser.

Period-by-period dynamics:

* ``Bernoulli``: Beta posterior per arm with shape parameters
  ``(1 + successes)/scale`` and ``(1 + failures)/scale``; the seller
  offers the arm maximizing ``theta * price`` over posterior draws.
* ``Normal``: conjugate Gaussian posterior over expected revenue per arm
  under a N(0,1) prior (mean ``sum(r)/(1 + plays)``, sd
  ``1/sqrt(1 + plays)``); the seller offers the arm with the largest
  posterior draw.
* ``Poisson``: per-period purchase *counts* with the Bernoulli means as
  rates; Gamma(1, 1) prior on each arm's rate, updated to
  Gamma(1 + purchases, 1 + plays); the seller offers the arm maximizing
  ``theta * price`` over Gamma posterior draws. The rate is unbounded
  above, so the advertising lift is floored at zero only. Vanilla
  Thompson sampling only (no forgetting variants).

Customer behaviour is controlled by the patience factor ``patience``
(0 = myopic, close to 1 = forward-looking). Forward-looking customers
weigh the offered price against the empirical distribution of past
offers (Bernoulli) or a reservation threshold built from the average
offered price (Normal). A fraction ``return_frac`` of customers who
waited returns in the next period.

Advertising (one-time effort ``ad_A`` paid as ``ad_A**2`` in the first
period, efficiency ``ad_theta``) lifts every customer's
willingness-to-pay by ``c(t) = ad_theta * ad_A * decay(t)``: geometric
``gamma**t`` or hyperbolic ``1/(t+1)`` forgetting.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Patience is capped just below 1 so the Normal-case reservation
# threshold (p - delta*avg)/(1 - delta) stays finite.
_DELTA_CAP = 1.0 - 1e-9


@dataclass(frozen=True)
class Scenario:
    label: str = "scenario"
    prices: tuple[float, ...] = (0.5, 0.8, 0.9)
    demands: tuple[float, ...] = (0.5, 0.2, 0.1)  # baseline purchase probabilities (Bernoulli)
    dist: Literal["Bernoulli", "Normal", "Poisson"] = "Bernoulli"
    T: int = 2000
    n_reps: int = 50
    patience: float = 0.0  # 0 = myopic, ->1 = forward-looking
    return_frac: float = 1.0  # fraction of waiting customers who return
    scale: float = 1.0  # Beta-posterior concentration divisor
    ad_A: float = 0.0  # one-time advertising effort (cost ad_A**2)
    ad_theta: float = 0.0  # advertising efficiency
    ad_gamma: float = 0.998  # forgetting rate for geometric decay
    ad_decay: Literal["geometric", "hyperbolic"] = "geometric"
    policy: Literal["ts", "sw_ts", "discounted_ts", "decay_aware_ts"] = "ts"
    sw_window: int = 500
    ts_discount: float = 0.99
    seed: int = 2026

    @property
    def k(self) -> int:
        return len(self.prices)

    def sigma_arr(self) -> np.ndarray:
        p = np.asarray(self.prices, dtype=float)
        d = np.asarray(self.demands, dtype=float)
        return p * d / 4.0

    def wtp_shift(self, t: int) -> float:
        """Advertising-induced WTP lift c(t) at 0-based period t."""
        if self.ad_A == 0.0 or self.ad_theta == 0.0:
            return 0.0
        if self.ad_decay == "geometric":
            return self.ad_theta * self.ad_A * self.ad_gamma**t
        return self.ad_theta * self.ad_A / (t + 1.0)


def simulate(
    sc: Scenario, progress: Callable[[float], None] | None = None
) -> dict[str, np.ndarray]:
    """Simulate ``sc.n_reps`` selling seasons of ``sc.T`` periods at once.

    Returns a dict of NumPy arrays:

    * ``"revenue"``: float32, ``(n_reps, T)`` — per-period net revenue
    * ``"arms"``: int8, ``(n_reps, T)`` — index of the offered price
    * ``"price_freq"``: int64, ``(n_reps, k)`` — offers per price over the season
    * ``"snap_periods"``: ``(S,)`` — periods at which posteriors were snapshotted
    * ``"snap_mean"``, ``"snap_sd"``: ``(S, k)`` — posterior mean/sd across replicates
    * ``"snap_mix"``: ``(S, k)`` — share of offers per price since the last snapshot

    ``progress`` is an optional callback taking the completed fraction
    in [0, 1].
    """
    k, T, R = sc.k, sc.T, sc.n_reps
    price = np.asarray(sc.prices, dtype=float)
    d = np.asarray(sc.demands, dtype=float)
    sigma = sc.sigma_arr()
    delta = min(sc.patience, _DELTA_CAP)
    strategic = sc.patience != 0.0
    rng = np.random.default_rng(np.random.SeedSequence([sc.seed]))
    rows = np.arange(R)

    revenue = np.zeros((R, T), dtype=np.float64)
    arms = np.zeros((R, T), dtype=np.int8)
    price_count = np.zeros((R, k), dtype=np.int64)

    # Bernoulli posterior state (windowed / discounted pseudo-counts for
    # the sw_ts / discounted_ts policies)
    success = np.zeros((R, k))
    failure = np.zeros((R, k))
    obs_arm = np.zeros((R, T), dtype=np.int8)  # only read by sw_ts
    obs_buy = np.zeros((R, T), dtype=np.int8)

    # Normal posterior state
    total_play = np.zeros((R, k), dtype=np.int64)
    mu_tilde = np.zeros((R, k))
    sigma_tilde = np.ones((R, k))

    # decay-aware TS state: Gaussian posterior over the baseline demand,
    # learned from observations de-shifted by the known lift c(t)
    da_mean = np.zeros((R, k))
    da_n = np.zeros((R, k), dtype=np.int64)

    # Poisson posterior state: Gamma(1 + purchases, 1 + plays) per arm
    pois_count = np.zeros((R, k))
    pois_play = np.zeros((R, k), dtype=np.int64)

    # forward-looking bookkeeping
    sum_price_offered = np.zeros(R)
    prev_arm = np.full(R, -1, dtype=np.int64)
    prev_price_incr_prob = np.ones(R)

    snap_every = max(1, T // 200)
    snap_periods, snap_mean, snap_sd, snap_mix = [], [], [], []
    win_counts = np.zeros(k)  # offers per arm since the last snapshot
    win_periods = 0

    for t in range(T):
        c_t = sc.wtp_shift(t)

        # ---------------------------------------------- seller offers a price
        if sc.dist == "Bernoulli":
            if sc.policy == "decay_aware_ts":
                theta = rng.normal(da_mean, 1.0 / np.sqrt(1.0 + da_n))
                pull = np.argmax(np.clip(theta + c_t, 0.0, 1.0) * price, axis=1)
            else:
                alpha = (1.0 + success) / sc.scale
                beta_ = (1.0 + failure) / sc.scale
                theta = rng.beta(alpha, beta_)
                pull = np.argmax(theta * price, axis=1)
        elif sc.dist == "Poisson":
            theta = rng.gamma(1.0 + pois_count, 1.0 / (1.0 + pois_play))
            pull = np.argmax(theta * price, axis=1)
        else:
            theta = rng.normal(mu_tilde, sigma_tilde)
            pull = np.argmax(theta, axis=1)

        offered = price[pull]
        arms[:, t] = pull
        price_count[rows, pull] += 1
        sum_price_offered += offered
        avg_price = sum_price_offered / (t + 1)

        # ---------------------------------------------- customers decide
        if sc.dist == "Bernoulli":
            d_eff = np.clip(d + c_t, 0.0, 1.0)
            w_bar = price_count / (t + 1)
            if strategic:
                # buy-now probability: mass of past offers at prices high
                # enough that waiting is not worthwhile
                ratio = (offered + delta - 1.0) / delta
                mask = price[None, :] >= ratio[:, None]
                price_incr_prob = np.where(mask, w_bar, 0.0).sum(axis=1)
            else:
                price_incr_prob = np.ones(R)

            cust_dec_prob = d_eff[pull] * price_incr_prob
            buy = rng.random(R) < cust_dec_prob

            # returning customers who waited at t-1
            prob_wait = np.where(
                prev_arm >= 0, d_eff[prev_arm] * (1.0 - prev_price_incr_prob), 0.0
            )
            extra = offered * (rng.random(R) < d_eff[pull]) * sc.return_frac * prob_wait
            revenue[:, t] = np.where(buy, offered, 0.0) + extra

            # posterior update from the immediate decision only
            if sc.policy == "decay_aware_ts":
                resid = buy.astype(float) - c_t
                da_mean[rows, pull] = (
                    da_mean[rows, pull] * da_n[rows, pull] + resid
                ) / (da_n[rows, pull] + 1)
                da_n[rows, pull] += 1
            else:
                if sc.policy == "discounted_ts":
                    success *= sc.ts_discount
                    failure *= sc.ts_discount
                success[rows[buy], pull[buy]] += 1.0
                failure[rows[~buy], pull[~buy]] += 1.0
                if sc.policy == "sw_ts":
                    obs_arm[:, t] = pull
                    obs_buy[:, t] = buy
                    old = t - sc.sw_window
                    if old >= 0:
                        j = obs_arm[:, old].astype(np.int64)
                        was_buy = obs_buy[:, old].astype(bool)
                        success[rows[was_buy], j[was_buy]] -= 1.0
                        failure[rows[~was_buy], j[~was_buy]] -= 1.0

            prev_price_incr_prob = price_incr_prob
            prev_arm = pull
        elif sc.dist == "Poisson":
            # Poisson demand: purchase counts with the Bernoulli means as
            # rates; the rate is unbounded above, so the lift is floored
            # at zero only.
            lam = np.maximum(d[pull] + c_t, 0.0)
            w_bar = price_count / (t + 1)
            if strategic:
                ratio = (offered + delta - 1.0) / delta
                mask = price[None, :] >= ratio[:, None]
                price_incr_prob = np.where(mask, w_bar, 0.0).sum(axis=1)
            else:
                price_incr_prob = np.ones(R)

            n_buy = rng.poisson(lam * price_incr_prob)
            prob_wait = np.where(
                prev_arm >= 0,
                np.maximum(d[prev_arm] + c_t, 0.0)
                * np.maximum(1.0 - prev_price_incr_prob, 0.0),
                0.0,
            )
            extra = offered * rng.poisson(lam * sc.return_frac * prob_wait)
            revenue[:, t] = offered * n_buy + extra

            pois_count[rows, pull] += n_buy
            pois_play[rows, pull] += 1
            prev_price_incr_prob = price_incr_prob
            prev_arm = pull
        else:
            # Normal demand: the WTP lift is equivalent to lowering the
            # effective price faced by the customer.
            p_eff = offered - c_t
            if strategic:
                threshold = (p_eff - delta * avg_price) / (1.0 - delta)
            else:
                threshold = p_eff
            cust_dec_prob = np.maximum(1.0 - np.maximum(p_eff, threshold), 0.0)

            rev = rng.normal(cust_dec_prob * offered, sigma[pull])
            rev = np.maximum(rev, 0.0)
            revenue[:, t] = rev

            mu_tilde[rows, pull] = (
                mu_tilde[rows, pull] * (total_play[rows, pull] + 1) + rev
            ) / (total_play[rows, pull] + 2)
            total_play[rows, pull] += 1
            sigma_tilde[rows, pull] = 1.0 / np.sqrt(1.0 + total_play[rows, pull])

        # one-time advertising cost, charged in the first period
        if t == 0 and sc.ad_A != 0.0:
            revenue[:, 0] -= sc.ad_A**2

        win_counts += np.bincount(pull, minlength=k)
        win_periods += 1
        if t % snap_every == 0 or t == T - 1:
            snap_mix.append(win_counts / (win_periods * R))
            win_counts = np.zeros(k)
            win_periods = 0
            if sc.dist == "Bernoulli":
                alpha = (1.0 + success) / sc.scale
                beta_ = (1.0 + failure) / sc.scale
                pm = alpha / (alpha + beta_)  # posterior mean of demand
            elif sc.dist == "Poisson":
                pm = (1.0 + pois_count) / (1.0 + pois_play)  # mean rate
            else:
                pm = mu_tilde  # posterior mean of expected revenue
            snap_periods.append(t + 1)
            snap_mean.append(pm.mean(axis=0))
            snap_sd.append(pm.std(axis=0))

        if progress is not None and (t % 200 == 0 or t == T - 1):
            progress((t + 1) / T)

    return {
        "revenue": revenue.astype(np.float32),
        "arms": arms,
        "price_freq": price_count,
        "snap_periods": np.array(snap_periods),
        "snap_mean": np.vstack(snap_mean),
        "snap_sd": np.vstack(snap_sd),
        "snap_mix": np.vstack(snap_mix),
    }


def simulate_fixed(sc: Scenario, arm: int) -> np.ndarray:
    """Revenue paths (n_reps, T) when the seller always offers ``prices[arm]``.

    The customer side (strategic waiting, returning customers,
    advertising) behaves exactly as in :func:`simulate`; only the
    seller's arm choice is fixed. Used as the oracle benchmark for
    regret.
    """
    T, R = sc.T, sc.n_reps
    price = np.asarray(sc.prices, dtype=float)
    d = np.asarray(sc.demands, dtype=float)
    sigma = sc.sigma_arr()
    delta = min(sc.patience, _DELTA_CAP)
    strategic = sc.patience != 0.0
    rng = np.random.default_rng(np.random.SeedSequence([sc.seed, 10_000 + arm]))

    revenue = np.zeros((R, T), dtype=np.float64)
    offered = price[arm]
    prev_price_incr_prob = np.ones(R)

    for t in range(T):
        c_t = sc.wtp_shift(t)
        if sc.dist == "Bernoulli":
            d_eff = float(np.clip(d[arm] + c_t, 0.0, 1.0))
            if strategic:
                # all past offers sit at this arm's price
                ratio = (offered + delta - 1.0) / delta
                price_incr_prob = np.full(R, 1.0 if offered >= ratio else 0.0)
            else:
                price_incr_prob = np.ones(R)
            buy = rng.random(R) < d_eff * price_incr_prob
            prob_wait = d_eff * (1.0 - prev_price_incr_prob) if t > 0 else 0.0
            extra = offered * (rng.random(R) < d_eff) * sc.return_frac * prob_wait
            revenue[:, t] = np.where(buy, offered, 0.0) + extra
            prev_price_incr_prob = price_incr_prob
        elif sc.dist == "Poisson":
            lam = max(d[arm] + c_t, 0.0)
            if strategic:
                # all past offers sit at this arm's price
                ratio = (offered + delta - 1.0) / delta
                price_incr_prob = np.full(R, 1.0 if offered >= ratio else 0.0)
            else:
                price_incr_prob = np.ones(R)
            n_buy = rng.poisson(lam * price_incr_prob)
            prob_wait = lam * np.maximum(1.0 - prev_price_incr_prob, 0.0) if t > 0 else 0.0
            extra = offered * rng.poisson(lam * sc.return_frac * prob_wait, size=R)
            revenue[:, t] = offered * n_buy + extra
            prev_price_incr_prob = price_incr_prob
        else:
            p_eff = offered - c_t
            if strategic:
                threshold = (p_eff - delta * offered) / (1.0 - delta)
            else:
                threshold = p_eff
            cust_dec_prob = max(1.0 - max(p_eff, threshold), 0.0)
            rev = rng.normal(cust_dec_prob * offered, sigma[arm], size=R)
            revenue[:, t] = np.maximum(rev, 0.0)

        if t == 0 and sc.ad_A != 0.0:
            revenue[:, 0] -= sc.ad_A**2

    return revenue.astype(np.float32)


def oracle_cum(sc: Scenario) -> np.ndarray:
    """Cumulative revenue (T,) of the best fixed-price policy.

    Each arm is simulated under the scenario's customer side; the oracle
    is the elementwise best mean cumulative-revenue curve.
    """
    curves = [simulate_fixed(sc, a).cumsum(axis=1).mean(axis=0) for a in range(sc.k)]
    return np.max(np.vstack(curves), axis=0)
