# Dynamic Pricing Lab

**Live dashboard: <https://mohammadi.cv/dynamic-pricing-dashboard/>**
(also reachable via <https://mohammadi-hadi.github.io/dynamic-pricing-dashboard/>)

An interactive simulator for dynamic pricing with demand learning. A
monopolist sells over a season, learning unknown demand by Thompson
sampling, while customers may strategically wait for lower prices and a
one-time advertising effort lifts willingness-to-pay before fading away.

Set your own scenario — price menu, demand, customer patience,
advertising effort/efficiency/forgetting, learning policy — run it, and
compare revenue, regret, price usage, and posterior learning across
market conditions.

The dashboard runs **entirely in your browser** (Python via WebAssembly,
using [stlite](https://github.com/whitphx/stlite)): nothing is installed
and no data leaves your machine. The first visit downloads the runtime
(~40 MB), so give it half a minute.

## What you can explore

- **Scenario lab** — compare myopic vs. forward-looking customers, with
  and without advertising: cumulative net revenue with confidence bands,
  regret against the best fixed price, how often each price is offered,
  and how the seller's posterior beliefs converge.
- **Optimal advertising** — sweep the one-time effort *A* and find the
  interior optimum *A\**, separately for patient and myopic customers.
- **Learning policies** — pit plain Thompson sampling against
  sliding-window, discounted, and decay-aware variants under the
  advertising-induced demand drift.

## Model in one paragraph

Prices come from a menu p₁ < … < p_K with unknown purchase
probabilities. Forward-looking customers (patience λ) buy now only when
a future discount is not credible given the observed price history, and
a fraction ρ of waiters returns next period. A one-time advertising
effort A, paid up front at cost A², lifts every customer's
willingness-to-pay by c(t) = θAγ^(t−1) — demand drifts exactly while
the seller is learning, but the total drift θA/(1−γ) is bounded and
chosen by the seller. Full details are on the dashboard's *Model notes*
tab.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

`simulator.py` has no dependencies beyond NumPy and can be used on its
own:

```python
from simulator import Scenario, simulate

out = simulate(Scenario(patience=0.99, ad_A=5, ad_theta=0.05, ad_gamma=0.998))
print(out["revenue"].sum(axis=1).mean())  # mean net season revenue
```

All replicates of a scenario are simulated simultaneously (vectorized
across seasons), so a 50-replicate, 2,000-period run takes about a
second natively and a few seconds in the browser.

## Repository layout

| File | Purpose |
| --- | --- |
| `simulator.py` | Vectorized simulation core (Bernoulli & Normal demand, four bandit policies, advertising, fixed-price oracle) |
| `app.py` | Streamlit dashboard (scenario presets, insights, comparison tables) |
| `charts.py` | Matplotlib figures in the site's editorial style |
| `index.html` | Browser host page (stlite) served via GitHub Pages, styled to match [mohammadi.cv](https://mohammadi.cv) |
| `.streamlit/config.toml` | Streamlit theme (paper background, navy accent) |
| `.github/workflows/deploy.yml` | Pages deployment |

## Author

Hadi Mohammadi — companion tool to a research project on dynamic
pricing with demand learning, forward-looking customers, and
advertising.
