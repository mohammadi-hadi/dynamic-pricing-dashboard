"""Interactive dashboard: dynamic pricing with demand learning,
forward-looking customers, and advertising.

Runs the vectorized Thompson sampling simulator in ``simulator.py``.
Works both as a regular Streamlit app and in the browser via stlite.
"""

import math
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from simulator import Scenario, oracle_cum, simulate

st.set_page_config(page_title="Dynamic Pricing Lab", page_icon="📈", layout="wide")

BLUE = "#0072B2"  # forward-looking
RED = "#D55E00"  # myopic
GRAY = "#666666"
ARM_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

CONDITIONS = {
    "Forward-looking + ads": dict(strategic=True, ads=True, color=BLUE, ls="-"),
    "Forward-looking, no ads": dict(strategic=True, ads=False, color=BLUE, ls="--"),
    "Myopic + ads": dict(strategic=False, ads=True, color=RED, ls="-"),
    "Myopic, no ads": dict(strategic=False, ads=False, color=RED, ls="--"),
}
BASELINE = "Myopic, no ads"


def parse_floats(text: str):
    try:
        vals = tuple(float(x) for x in text.replace(";", ",").split(",") if x.strip())
    except ValueError:
        return None
    return vals if vals else None


@st.cache_data(max_entries=160, show_spinner=False)
def run_scenario(sc: Scenario) -> dict:
    """Simulate a scenario plus its fixed-price oracle; return summaries."""
    out = simulate(sc)
    cum = out["revenue"].astype(np.float64).cumsum(axis=1)
    n = cum.shape[0]
    mean = cum.mean(axis=0)
    se = cum.std(axis=0, ddof=1) / math.sqrt(n)
    oracle = oracle_cum(sc)
    return {
        "cum_mean": mean,
        "cum_lo": mean - 1.96 * se,
        "cum_hi": mean + 1.96 * se,
        "final": cum[:, -1],
        "regret": oracle - mean,
        "share": 100.0 * out["price_freq"].mean(axis=0) / sc.T,
        "snap_periods": out["snap_periods"],
        "snap_mean": out["snap_mean"],
        "snap_sd": out["snap_sd"],
    }


def welch(a: np.ndarray, b: np.ndarray):
    """Welch t statistic and a normal-approximation p-value."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    p = math.erfc(abs(t) / math.sqrt(2.0))
    return t, p


def base_figure(xlabel, ylabel, height=3.1):
    fig, ax = plt.subplots(figsize=(6.6, height), dpi=110)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    return fig, ax


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📈 Dynamic Pricing Lab")
    st.caption(
        "A seller learns unknown demand by Thompson sampling while "
        "customers may wait for lower prices — and a one-time advertising "
        "effort lifts willingness-to-pay before fading away."
    )
    with st.form("params"):
        st.subheader("Market")
        dist = st.radio("Demand model", ["Bernoulli", "Normal"], horizontal=True)
        prices_text = st.text_input("Price menu", "0.5, 0.8, 0.9")
        demands_text = st.text_input("Baseline demand per price", "0.5, 0.2, 0.1")
        st.caption(
            "Comma-separated, same length. Prices must lie in (0, 1). In the "
            "Normal model the purchase probability is 1 − price; demands "
            "only set the noise scale σᵢ = pᵢdᵢ/4."
        )
        T = st.slider("Season length T", 500, 4000, 2000, step=250)
        n_reps = st.slider("Replicates", 10, 200, 50, step=10)

        st.subheader("Customers")
        patience = st.slider(
            "Patience λ of forward-looking customers", 0.0, 0.99, 0.99, step=0.01
        )
        return_frac = st.slider("Returning fraction ρ of waiters", 0.0, 1.0, 1.0, 0.05)

        st.subheader("Advertising")
        ad_A = st.slider("One-time effort A (cost A²)", 0.0, 12.0, 5.0, step=0.5)
        ad_theta = st.slider("Efficiency θ", 0.0, 0.2, 0.05, step=0.005)
        ad_gamma = st.number_input(
            "Forgetting rate γ", 0.5, 0.99999, 0.998, step=0.001, format="%.4f"
        )
        ad_decay = st.radio("Decay shape", ["geometric", "hyperbolic"], horizontal=True)

        st.subheader("Learning policy")
        policy = st.selectbox(
            "Seller's bandit policy (Bernoulli only)",
            ["ts", "sw_ts", "discounted_ts", "decay_aware_ts"],
            format_func=lambda s: {
                "ts": "Thompson sampling",
                "sw_ts": "Sliding-window TS",
                "discounted_ts": "Discounted TS",
                "decay_aware_ts": "Decay-aware TS",
            }[s],
        )
        with st.expander("Advanced"):
            sw_window = st.number_input("Sliding window", 50, 4000, 500, step=50)
            ts_discount = st.number_input("Observation discount", 0.5, 1.0, 0.99, 0.005)
            scale = st.number_input("Beta prior concentration", 0.1, 10.0, 1.0, 0.1)
            seed = st.number_input("Random seed", 0, 10**6, 2026, step=1)

        chosen = st.multiselect(
            "Scenarios to compare", list(CONDITIONS), default=list(CONDITIONS)
        )
        run_clicked = st.form_submit_button("▶ Run simulation", type="primary")

prices = parse_floats(prices_text)
demands = parse_floats(demands_text)
valid = True
if prices is None or demands is None or len(prices) != len(demands):
    st.error("Prices and demands must be comma-separated numbers of equal length.")
    valid = False
elif not all(0.0 < p < 1.0 for p in prices) or not all(0.0 < d <= 1.0 for d in demands):
    st.error("Prices must lie in (0, 1) and demands in (0, 1].")
    valid = False


def make_scenario(name: str, ad_override=None) -> Scenario:
    spec = CONDITIONS[name]
    A = ad_A if spec["ads"] else 0.0
    if ad_override is not None:
        A = ad_override
    return Scenario(
        label=name,
        prices=prices,
        demands=demands,
        dist=dist,
        T=int(T),
        n_reps=int(n_reps),
        patience=patience if spec["strategic"] else 0.0,
        return_frac=return_frac,
        scale=scale,
        ad_A=A,
        ad_theta=ad_theta,
        ad_gamma=float(ad_gamma),
        ad_decay=ad_decay,
        policy=policy if dist == "Bernoulli" else "ts",
        sw_window=int(sw_window),
        ts_discount=ts_discount,
        seed=int(seed),
    )


tab_lab, tab_sweep, tab_about = st.tabs(
    ["🎛️ Scenario lab", "📣 Optimal advertising", "📖 Model notes"]
)

# ---------------------------------------------------------------- scenario lab
with tab_lab:
    if run_clicked and valid and chosen:
        bar = st.progress(0.0, text="Simulating…")
        results = {}
        for i, name in enumerate(chosen):
            bar.progress(i / len(chosen), text=f"Simulating: {name}")
            results[name] = run_scenario(make_scenario(name))
        bar.empty()
        st.session_state["lab"] = {
            "results": results,
            "prices": prices,
            "demands": demands,
            "dist": dist,
            "T": int(T),
            "n_reps": int(n_reps),
        }

    lab = st.session_state.get("lab")
    if not lab:
        st.info("Set the parameters in the sidebar and press **Run simulation**.")
    else:
        res, lp, ld = lab["results"], lab["prices"], lab["demands"]
        names = list(res)

        cols = st.columns(len(names))
        base_mean = res[BASELINE]["final"].mean() if BASELINE in res else None
        for col, name in zip(cols, names):
            m = res[name]["final"].mean()
            s = res[name]["final"].std(ddof=1)
            delta = (
                f"{m - base_mean:+.1f} vs myopic, no ads"
                if base_mean is not None and name != BASELINE
                else None
            )
            col.metric(name, f"{m:,.1f} ± {s:,.1f}", delta)
        st.caption(
            f"Mean net revenue over {lab['n_reps']} replicates of "
            f"T = {lab['T']} periods (± one standard deviation). "
            "Advertising scenarios are net of the A² cost."
        )

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = base_figure("Period", "Cumulative net revenue")
            for name in names:
                spec, r = CONDITIONS[name], res[name]
                x = np.arange(1, len(r["cum_mean"]) + 1)
                ax.plot(x, r["cum_mean"], color=spec["color"], ls=spec["ls"],
                        lw=1.6, label=name)
                ax.fill_between(x, r["cum_lo"], r["cum_hi"],
                                color=spec["color"], alpha=0.15, lw=0)
            ax.legend(fontsize=7.5, loc="upper left", frameon=False)
            ax.set_title("Revenue accumulation (mean, 95% band)", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            fig, ax = base_figure("Period", "Cumulative regret")
            for name in names:
                spec, r = CONDITIONS[name], res[name]
                x = np.arange(1, len(r["regret"]) + 1)
                ax.plot(x, r["regret"], color=spec["color"], ls=spec["ls"],
                        lw=1.6, label=name)
            ax.legend(fontsize=7.5, loc="upper left", frameon=False)
            ax.set_title("Regret vs best fixed price", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)

        c3, c4 = st.columns(2)
        with c3:
            fig, ax = base_figure("Posted price", "Share of periods (%)", height=3.0)
            width = 0.8 / len(names)
            xs = np.arange(len(lp))
            for j, name in enumerate(names):
                spec = CONDITIONS[name]
                ax.bar(xs + (j - (len(names) - 1) / 2) * width,
                       res[name]["share"], width=width * 0.92,
                       color=spec["color"], alpha=1.0 if spec["ls"] == "-" else 0.45,
                       label=name)
            ax.set_xticks(xs, [f"{p:.2f}" for p in lp])
            ax.legend(fontsize=7.5, frameon=False)
            ax.set_title("How often each price is offered", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)
        with c4:
            pick = st.selectbox("Posterior evolution for scenario", names)
            r = res[pick]
            ylab = ("Posterior mean of demand" if lab["dist"] == "Bernoulli"
                    else "Posterior mean of expected revenue")
            fig, ax = base_figure("Period", ylab, height=2.85)
            for a in range(len(lp)):
                ccol = ARM_COLORS[a % len(ARM_COLORS)]
                ax.plot(r["snap_periods"], r["snap_mean"][:, a], color=ccol,
                        lw=1.5, label=f"p = {lp[a]:.2f}")
                ax.fill_between(r["snap_periods"],
                                r["snap_mean"][:, a] - r["snap_sd"][:, a],
                                r["snap_mean"][:, a] + r["snap_sd"][:, a],
                                color=ccol, alpha=0.15, lw=0)
                if lab["dist"] == "Bernoulli":
                    ax.axhline(ld[a], color=ccol, ls=":", lw=0.9, alpha=0.7)
            ax.legend(fontsize=7.5, frameon=False)
            ax.set_title("What the seller believes (dotted = truth)", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)

        if len(names) > 1:
            st.subheader("Pairwise comparison of final net revenue")
            rows = []
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    t, p = welch(res[a]["final"], res[b]["final"])
                    rows.append({
                        "A": a, "B": b,
                        "mean A": round(res[a]["final"].mean(), 1),
                        "mean B": round(res[b]["final"].mean(), 1),
                        "Δ (A−B)": round(res[a]["final"].mean()
                                         - res[b]["final"].mean(), 1),
                        "Welch t": round(t, 1),
                        "p (normal approx.)": f"{p:.2g}" if p > 1e-15 else "< 1e-15",
                    })
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(
                "Positive Δ means scenario A earns more. With few replicates, "
                "treat the p-values as indicative only."
            )

# ---------------------------------------------------------------- A* sweep
with tab_sweep:
    st.markdown(
        "How much should the seller advertise? Net revenue first rises with "
        "the effort $A$ (the willingness-to-pay lift stimulates demand and "
        "weakens the incentive to wait) and then falls (the $A^2$ cost "
        "dominates) — an interior optimum $A^{*}$."
    )
    sc1, sc2, sc3 = st.columns(3)
    a_max = sc1.slider("Largest effort to try", 2.0, 16.0, 10.0, step=1.0)
    a_step = sc2.select_slider("Grid step", [0.5, 1.0, 2.0], value=1.0)
    sweep_reps = sc3.slider("Replicates per point", 10, 100, 25, step=5)

    if st.button("▶ Run sweep", type="primary") and valid:
        grid = np.arange(0.0, a_max + 1e-9, a_step)
        curves = {"Forward-looking": ([], [], BLUE), "Myopic": ([], [], RED)}
        bar = st.progress(0.0, text="Sweeping advertising efforts…")
        for gi, A in enumerate(grid):
            for label, cond in [("Forward-looking", "Forward-looking + ads"),
                                ("Myopic", "Myopic + ads")]:
                sc = replace(make_scenario(cond, ad_override=float(A)),
                             n_reps=int(sweep_reps))
                r = run_scenario(sc)
                curves[label][0].append(r["final"].mean())
                curves[label][1].append(r["final"].std(ddof=1)
                                        / math.sqrt(len(r["final"])))
            bar.progress((gi + 1) / len(grid), text=f"A = {A:g} done")
        bar.empty()
        st.session_state["sweep"] = {"grid": grid, "curves": curves}

    sweep = st.session_state.get("sweep")
    if sweep:
        grid, curves = sweep["grid"], sweep["curves"]
        fig, ax = base_figure("Advertising effort A", "Mean net revenue", height=3.3)
        rows = {"A": np.round(grid, 2)}
        for label, (means, ses, color) in curves.items():
            means, ses = np.array(means), np.array(ses)
            ax.plot(grid, means, color=color, lw=1.7, marker="o", ms=3.5, label=label)
            ax.fill_between(grid, means - 1.96 * ses, means + 1.96 * ses,
                            color=color, alpha=0.15, lw=0)
            best = int(np.argmax(means))
            ax.plot(grid[best], means[best], marker="*", ms=15, color=color,
                    mec="white", mew=0.8, zorder=5)
            ax.annotate(f"A* = {grid[best]:g}", (grid[best], means[best]),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=8.5, color=color)
            rows[label] = np.round(means, 1)
        ax.legend(fontsize=8.5, frameon=False, loc="lower center")
        ax.set_title("Net revenue by advertising effort (95% band, ★ = best)",
                     fontsize=10)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(
            "Both curves use the sidebar market, customer, and decay settings; "
            "the myopic curve sets λ = 0. Patient customers typically support "
            "a larger optimal effort: the lift also erodes their option value "
            "of waiting."
        )

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
### The market

A monopolist sells one product over $T$ periods, posting each period one
price from a menu $p_1 < \\dots < p_K$ with **unknown** demand. Selling
and learning happen at the same time: every posted price is also an
experiment.

**Thompson sampling.** In the Bernoulli model the seller keeps a Beta
posterior on each price's purchase probability, draws
$\\vartheta_i \\sim \\mathrm{Beta}(\\alpha_i, \\beta_i)$, and posts the
price maximizing $p_i\\,\\vartheta_i$. In the Normal model the posterior
is Gaussian over expected revenue per period.

**Forward-looking customers** (patience $\\lambda$) compare buying now
against waiting. In the Bernoulli model a customer facing price $p$ buys
now with probability equal to the share of past offers at prices
$p_j \\ge (p + \\lambda - 1)/\\lambda$ — the more often prices were high,
the less credible a future discount. A fraction $\\rho$ of customers who
waited returns next period. In the Normal model the customer buys iff the
valuation exceeds
$\\max\\{p,\\ (p - \\lambda\\bar p)/(1 - \\lambda)\\}$, where $\\bar p$
is the average posted price. Setting $\\lambda = 0$ recovers myopic
customers.

**Advertising.** A one-time effort $A$, paid up front at cost $A^2$,
lifts every customer's willingness-to-pay in period $t$ by
"""
    )
    st.latex(r"c(t) \;=\; \theta A\,\gamma^{\,t-1},\qquad \gamma\in(0,1),")
    st.markdown(
        """
so demand is **nonstationary exactly while the seller is learning** —
but the drift is chosen and paid for by the seller, and its total size
$\\sum_t c(t) = \\theta A/(1-\\gamma)$ does not grow with the horizon.
The hyperbolic option replaces $\\gamma^{t-1}$ with $1/t$.

**Policies.** Besides plain Thompson sampling you can try
sliding-window TS (drops observations older than the window),
discounted TS (geometrically down-weights the past), and decay-aware TS
(knows the advertising schedule and de-biases its observations). Under
this bounded, self-inflicted drift, forgetting tends to *lose* money —
it discards exactly the observations that identify the best price.

### Things to try

- Run the four default scenarios: advertising helps forward-looking
  customers several times more than myopic ones.
- Sweep the advertising effort and watch the interior optimum $A^*$
  move as you change patience $\\lambda$.
- Switch the policy to sliding-window TS and watch revenue drop.
- Choose a five-price menu (e.g. `0.5, 0.6, 0.7, 0.8, 0.9` with demands
  `0.5, 0.45, 0.35, 0.2, 0.1`) so the best price is interior — patience
  can then *hurt* the seller.

### Reading the output

*Net revenue* is cumulative sales revenue minus the one-time $A^2$
advertising cost. *Regret* compares the learning seller with a
clairvoyant who always posts the best fixed price against the same
customers. Shaded bands are 95% confidence bands across replicates.

---
Simulator and dashboard by **Hadi Mohammadi** ·
[Source code](https://github.com/mohammadi-hadi/dynamic-pricing-dashboard)
"""
    )
