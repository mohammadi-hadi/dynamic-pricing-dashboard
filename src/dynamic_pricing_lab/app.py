"""Dynamic Pricing Lab — interactive dashboard.

A seller learns unknown demand by Thompson sampling while customers may
strategically wait and a one-time advertising effort lifts
willingness-to-pay. Runs the vectorized simulator in ``simulator.py``;
figures live in ``charts.py``. Works as a regular Streamlit app and
in-browser via stlite.
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

try:  # installed package (pip install dynamic-pricing-lab)
    from dynamic_pricing_lab import charts
    from dynamic_pricing_lab.simulator import Scenario, oracle_cum, simulate
except ModuleNotFoundError:  # flat files: stlite in the browser, or a bare checkout
    import charts
    from simulator import Scenario, oracle_cum, simulate

st.set_page_config(page_title="Dynamic Pricing Lab", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: "Inter", -apple-system, sans-serif; }
    h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif !important;
                 letter-spacing: -0.01em; color: #1a1a1a; }
    h1 { font-weight: 700 !important; }
    .kicker { font-family: "JetBrains Mono", ui-monospace, monospace;
              font-size: 0.72rem; font-weight: 500; letter-spacing: 0.14em;
              text-transform: uppercase; color: #6e6e6e; display: block; }
    .lede { color: #444; max-width: 62ch; font-size: 1.02rem; }
    [data-testid="stMetricLabel"] p {
        font-family: "JetBrains Mono", ui-monospace, monospace;
        font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: #6e6e6e; }
    [data-testid="stMetricValue"] {
        font-family: "Source Serif 4", Georgia, serif; font-weight: 600;
        color: #1a1a1a; }
    .insight { background: #f2f1ec; border-left: 3px solid #1e3a5f;
               border-radius: 2px; padding: 0.85rem 1.1rem; margin: 0.35rem 0;
               color: #1a1a1a; font-size: 0.95rem; }
    .insight b { color: #1e3a5f; }
    button[data-baseweb="tab"] p { font-size: 0.92rem; font-weight: 500; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-family: "Inter", sans-serif !important; font-size: 0.8rem;
        letter-spacing: 0.1em; text-transform: uppercase; color: #444; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASELINE = "Myopic, no ads"
CONDITIONS = {
    "Forward-looking + ads": dict(strategic=True, ads=True),
    "Forward-looking, no ads": dict(strategic=True, ads=False),
    "Myopic + ads": dict(strategic=False, ads=True),
    "Myopic, no ads": dict(strategic=False, ads=False),
}

DEFAULTS = {
    "dist": "Bernoulli",
    "prices_text": "0.5, 0.8, 0.9",
    "demands_text": "0.5, 0.2, 0.1",
    "T": 2000,
    "reps": 50,
    "patience": 0.99,
    "rho": 1.0,
    "A": 5.0,
    "theta": 0.05,
    "gamma": 0.998,
    "decay": "geometric",
    "policy": "ts",
    "window": 500,
    "discount": 0.99,
    "scale": 1.0,
    "seed": 2026,
    "conds": list(CONDITIONS),
}

PRESETS = {
    "Paper calibration (2×2)": {},
    "Five-price menu — patience can backfire": {
        "prices_text": "0.5, 0.6, 0.7, 0.8, 0.9",
        "demands_text": "0.5, 0.45, 0.35, 0.2, 0.1",
    },
    "Hyperbolic decay — advertising value destroyed": {"decay": "hyperbolic"},
    "Sliding-window TS — forgetting loses money": {"policy": "sw_ts"},
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    choice = st.session_state.get("preset", "Custom")
    if choice in PRESETS:
        for k, v in {**DEFAULTS, **PRESETS[choice]}.items():
            st.session_state[k] = v


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
    return {
        "cum_mean": mean,
        "cum_lo": mean - 1.96 * se,
        "cum_hi": mean + 1.96 * se,
        "final": cum[:, -1],
        "regret": oracle_cum(sc) - mean,
        "share": 100.0 * out["price_freq"].mean(axis=0) / sc.T,
        "snap_periods": out["snap_periods"],
        "snap_mean": out["snap_mean"],
        "snap_sd": out["snap_sd"],
        "snap_mix": out["snap_mix"],
    }


def welch(a: np.ndarray, b: np.ndarray):
    """Welch t statistic and a normal-approximation p-value."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    return t, math.erfc(abs(t) / math.sqrt(2.0))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<span class="kicker">Scenario builder</span>', unsafe_allow_html=True)
    st.selectbox(
        "Preset scenario",
        ["Custom"] + list(PRESETS),
        key="preset",
        on_change=apply_preset,
        help="Applying a preset resets every control, then changes the "
        "highlighted parameters. Edit anything afterwards.",
    )
    with st.form("params"):
        st.subheader("Market")
        st.radio("Demand model", ["Bernoulli", "Normal"], horizontal=True, key="dist")
        st.text_input("Price menu", key="prices_text")
        st.text_input("Baseline demand per price", key="demands_text")
        st.caption(
            "Comma-separated, equal length, prices in (0, 1). In the Normal "
            "model the purchase probability is 1 − price; demands only set "
            "the noise scale σᵢ = pᵢdᵢ/4."
        )
        st.slider("Season length T", 500, 4000, step=250, key="T")
        st.slider("Replicates", 10, 200, step=10, key="reps")

        st.subheader("Customers")
        st.slider("Patience λ (forward-looking)", 0.0, 0.99, step=0.01, key="patience")
        st.slider("Returning fraction ρ of waiters", 0.0, 1.0, step=0.05, key="rho")

        st.subheader("Advertising")
        st.slider("One-time effort A (cost A²)", 0.0, 12.0, step=0.5, key="A")
        st.slider("Efficiency θ", 0.0, 0.2, step=0.005, key="theta")
        st.number_input("Forgetting rate γ", 0.5, 0.99999, step=0.001,
                        format="%.4f", key="gamma")
        st.radio("Decay shape", ["geometric", "hyperbolic"], horizontal=True,
                 key="decay")

        st.subheader("Learning policy")
        st.selectbox(
            "Seller's bandit policy (Bernoulli only)",
            ["ts", "sw_ts", "discounted_ts", "decay_aware_ts"],
            format_func=lambda s: {
                "ts": "Thompson sampling",
                "sw_ts": "Sliding-window TS",
                "discounted_ts": "Discounted TS",
                "decay_aware_ts": "Decay-aware TS",
            }[s],
            key="policy",
        )
        with st.expander("Advanced"):
            st.number_input("Sliding window", 50, 4000, step=50, key="window")
            st.number_input("Observation discount", 0.5, 1.0, step=0.005,
                            key="discount")
            st.number_input("Beta prior concentration", 0.1, 10.0, step=0.1,
                            key="scale")
            st.number_input("Random seed", 0, 10**6, step=1, key="seed")

        st.multiselect("Scenarios to compare", list(CONDITIONS), key="conds")
        run_clicked = st.form_submit_button("▶ Run simulation", type="primary")

prices = parse_floats(st.session_state.prices_text)
demands = parse_floats(st.session_state.demands_text)
valid = True
if prices is None or demands is None or len(prices) != len(demands):
    st.error("Prices and demands must be comma-separated numbers of equal length.")
    valid = False
elif not all(0.0 < p < 1.0 for p in prices) or not all(0.0 < d <= 1.0 for d in demands):
    st.error("Prices must lie in (0, 1) and demands in (0, 1].")
    valid = False


def make_scenario(name: str, ad_override=None) -> Scenario:
    assert prices is not None and demands is not None  # callers are gated on `valid`
    s = st.session_state
    spec = CONDITIONS[name]
    A = s.A if spec["ads"] else 0.0
    if ad_override is not None:
        A = ad_override
    return Scenario(
        label=name,
        prices=prices,
        demands=demands,
        dist=s.dist,
        T=int(s.T),
        n_reps=int(s.reps),
        patience=s.patience if spec["strategic"] else 0.0,
        return_frac=s.rho,
        scale=s.scale,
        ad_A=A,
        ad_theta=s.theta,
        ad_gamma=float(s.gamma),
        ad_decay=s.decay,
        policy=s.policy if s.dist == "Bernoulli" else "ts",
        sw_window=int(s.window),
        ts_discount=s.discount,
        seed=int(s.seed),
    )


def insights(res: dict) -> list:
    out = []
    m = {name: r["final"].mean() for name, r in res.items()}
    fa, f0 = "Forward-looking + ads", "Forward-looking, no ads"
    ma, m0 = "Myopic + ads", "Myopic, no ads"
    if fa in m and f0 in m and ma in m and m0 in m:
        g_f = 100.0 * (m[fa] - m[f0]) / m[f0]
        g_m = 100.0 * (m[ma] - m[m0]) / m[m0]
        line = (f"Advertising changes net revenue by <b>{g_f:+.2f}%</b> with "
                f"forward-looking customers and <b>{g_m:+.2f}%</b> with myopic ones")
        if g_m > 0 and g_f > 0:
            line += (f" — <b>{g_f / g_m:.1f}×</b> more valuable when customers "
                     "can wait, because the lift also erodes the option value "
                     "of waiting.")
        else:
            line += "."
        out.append(line)
    if f0 in m and m0 in m:
        d = m[f0] - m[m0]
        verdict = "helps" if d > 0 else "hurts"
        out.append(
            f"Without advertising, customer patience <b>{verdict}</b> the "
            f"seller here: forward-looking demand yields <b>{d:+.1f}</b> "
            "revenue units versus myopic demand."
        )
    pairs = [(fa, ma), (f0, m0)]
    for a, b in pairs:
        if a in res and b in res:
            ra, rb = res[a]["regret"][-1], res[b]["regret"][-1]
            if ra > 0 and rb / ra > 1.15:
                out.append(
                    f"The seller learns faster against forward-looking "
                    f"customers: final regret <b>{ra:.1f}</b> versus "
                    f"<b>{rb:.1f}</b> ({rb / ra:.1f}× lower)."
                )
            break
    return out


# ---------------------------------------------------------------- header
st.markdown('<span class="kicker">Interactive simulator</span>',
            unsafe_allow_html=True)
st.title("Dynamic Pricing Lab")
st.markdown(
    '<p class="lede">Price a product without knowing its demand. Thompson '
    "sampling learns while selling; customers may wait for discounts; a "
    "one-time advertising effort lifts willingness-to-pay and fades. Set a "
    "scenario in the sidebar and see what happens.</p>",
    unsafe_allow_html=True,
)

tab_lab, tab_sweep, tab_about = st.tabs(
    ["Scenario lab", "Optimal advertising", "Model notes"]
)

# ---------------------------------------------------------------- scenario lab
with tab_lab:
    if run_clicked and valid and st.session_state.conds:
        bar = st.progress(0.0, text="Simulating…")
        results = {}
        for i, name in enumerate(st.session_state.conds):
            bar.progress(i / len(st.session_state.conds), text=f"Simulating: {name}")
            results[name] = run_scenario(make_scenario(name))
        bar.empty()
        st.session_state["lab"] = {
            "results": results,
            "prices": prices,
            "demands": demands,
            "dist": st.session_state.dist,
            "T": int(st.session_state.T),
            "n_reps": int(st.session_state.reps),
        }

    lab = st.session_state.get("lab")
    if not lab:
        st.info("Set the parameters in the sidebar and press **▶ Run simulation** "
                "— or pick a preset scenario first.")
    else:
        res, lp, ld = lab["results"], lab["prices"], lab["demands"]
        names = list(res)

        cols = st.columns(len(names))
        base_mean = res[BASELINE]["final"].mean() if BASELINE in res else None
        for col, name in zip(cols, names):
            m = res[name]["final"].mean()
            s = res[name]["final"].std(ddof=1)
            delta = (f"{m - base_mean:+.1f} vs baseline"
                     if base_mean is not None and name != BASELINE else None)
            col.metric(name, f"{m:,.1f} ± {s:,.1f}", delta)
        st.caption(
            f"Mean net revenue over {lab['n_reps']} replicates of "
            f"T = {lab['T']} periods (± one standard deviation), net of the "
            "A² advertising cost. Baseline: myopic customers, no ads."
        )

        for line in insights(res):
            st.markdown(f'<div class="insight">{line}</div>',
                        unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = charts.cum_revenue(res)
            st.pyplot(fig)
            charts.plt.close(fig)
        with c2:
            fig = charts.regret(res)
            st.pyplot(fig)
            charts.plt.close(fig)

        st.markdown('<span class="kicker">Inside the policy</span>',
                    unsafe_allow_html=True)
        pick = st.selectbox("Scenario to inspect", names)
        c3, c4 = st.columns(2)
        with c3:
            fig = charts.policy_mix(res[pick], lp)
            st.pyplot(fig)
            charts.plt.close(fig)
        with c4:
            fig = charts.posterior(res[pick], lp, ld, lab["dist"] == "Bernoulli")
            st.pyplot(fig)
            charts.plt.close(fig)

        c5, c6 = st.columns(2)
        with c5:
            fig = charts.shares(res, lp)
            st.pyplot(fig)
            charts.plt.close(fig)
        with c6:
            if len(names) > 1:
                st.markdown("**Pairwise final net revenue**")
                rows = []
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        t, p = welch(res[a]["final"], res[b]["final"])
                        rows.append({
                            "A": a, "B": b,
                            "Δ (A−B)": round(res[a]["final"].mean()
                                             - res[b]["final"].mean(), 1),
                            "Welch t": round(t, 1),
                            "p (approx.)": f"{p:.2g}" if p > 1e-15 else "< 1e-15",
                        })
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                st.caption(
                    "Positive Δ means scenario A earns more. With few "
                    "replicates, treat the p-values as indicative only."
                )

# ---------------------------------------------------------------- A* sweep
with tab_sweep:
    st.markdown(
        "How much should the seller advertise? Net revenue first rises with "
        "the effort $A$ — the willingness-to-pay lift stimulates demand and "
        "weakens the incentive to wait — then falls as the $A^2$ cost "
        "dominates: an interior optimum $A^{*}$."
    )
    sc1, sc2, sc3 = st.columns(3)
    a_max = sc1.slider("Largest effort to try", 2.0, 16.0, 10.0, step=1.0)
    a_step = sc2.select_slider("Grid step", [0.5, 1.0, 2.0], value=1.0)
    sweep_reps = sc3.slider("Replicates per point", 10, 100, 25, step=5)

    if st.button("▶ Run sweep", type="primary") and valid:
        grid = np.arange(0.0, a_max + 1e-9, a_step)
        curves = {"Forward-looking": ([], [], charts.NAVY),
                  "Myopic": ([], [], charts.BRICK)}
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
        fig = charts.sweep(grid, curves)
        st.pyplot(fig)
        charts.plt.close(fig)
        best = {lbl: grid[int(np.argmax(c[0]))] for lbl, c in curves.items()}
        if best["Forward-looking"] > best["Myopic"]:
            st.markdown(
                f'<div class="insight">Patient customers support a larger '
                f'optimal effort: <b>A* = {best["Forward-looking"]:g}</b> '
                f'forward-looking versus <b>A* = {best["Myopic"]:g}</b> myopic '
                "— advertising also buys a weaker incentive to wait.</div>",
                unsafe_allow_html=True,
            )
        rows = {"A": np.round(grid, 2)}
        rows.update({lbl: np.round(np.asarray(c[0]), 1)
                     for lbl, c in curves.items()})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(
            "Both curves use the sidebar market, customer, and decay "
            "settings; the myopic curve sets λ = 0."
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
$p_j \\ge (p + \\lambda - 1)/\\lambda$ — the more often prices were
high, the less credible a future discount. A fraction $\\rho$ of
customers who waited returns next period. In the Normal model the
customer buys iff the valuation exceeds
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

### Reading the output

*Net revenue* is cumulative sales revenue minus the one-time $A^2$
advertising cost. *Regret* compares the learning seller with a
clairvoyant who always posts the best fixed price against the same
customers. Shaded bands are 95% confidence bands across replicates.
Everything is computed live in your browser — nothing leaves your
machine.

---

<span class="kicker">Author</span>

**Hadi Mohammadi** — companion tool to a research project on dynamic
pricing with demand learning, forward-looking customers, and
advertising.
[mohammadi.cv](https://mohammadi.cv) ·
[Source code](https://github.com/mohammadi-hadi/dynamic-pricing-dashboard) ·
[Google Scholar](https://scholar.google.com/citations?user=w4Jt-FAAAAAJ&hl=en) ·
[LinkedIn](https://www.linkedin.com/in/mohammadi-hadi/)
""",
        unsafe_allow_html=True,
    )
