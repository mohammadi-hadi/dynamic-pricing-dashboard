# Dynamic Pricing Lab

[![CI](https://github.com/mohammadi-hadi/dynamic-pricing-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammadi-hadi/dynamic-pricing-dashboard/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/mohammadi-hadi/dynamic-pricing-dashboard)](https://github.com/mohammadi-hadi/dynamic-pricing-dashboard/releases)
[![Container](https://img.shields.io/badge/ghcr.io-container-1e3a5f?logo=docker&logoColor=white)](https://github.com/mohammadi-hadi/dynamic-pricing-dashboard/pkgs/container/dynamic-pricing-dashboard)
[![License: MIT](https://img.shields.io/badge/license-MIT-1e3a5f)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21802866.svg)](https://doi.org/10.5281/zenodo.21802866)

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
  interior optimum *A\**, separately for patient and myopic customers,
  against the full-information benchmark *A\*₍FI₎ = p₁θ/(2(1−γ))*.
- **Patience × advertising** — a heatmap of net revenue over the
  (patience, effort) plane with the ridge *A\*(λ)*: the optimal effort
  rises as customers get more patient.
- **Learning policies** — pit plain Thompson sampling against
  sliding-window, discounted, and decay-aware variants under the
  advertising-induced demand drift.
- **Three demand models** — Bernoulli purchases, Normal revenue, and a
  Poisson purchase-count variant with a Gamma posterior (the paper's
  unbounded-demand robustness check).

## Model in one paragraph

Prices come from a menu p₁ < … < p_K with unknown purchase
probabilities. Forward-looking customers (patience λ) buy now only when
a future discount is not credible given the observed price history, and
a fraction ρ of waiters returns next period. A one-time advertising
effort A, paid up front at cost A², lifts every customer's
willingness-to-pay by c(t) = θAγ^(t−1) — demand drifts exactly while
the seller is learning, but the drift is chosen and paid for by the
seller: the cumulative lift is at most θA/(1−γ) and the total variation
of the demand path at most θA. Full details are on the dashboard's
*Model notes* tab.

## Install

The repository is a Python package, `dynamic-pricing-lab`, on PyPI:

```bash
pip install "dynamic-pricing-lab[dashboard]"
dynamic-pricing-lab   # opens the dashboard at http://localhost:8501
```

(or straight from GitHub:
`pip install "dynamic-pricing-lab[dashboard] @ git+https://github.com/mohammadi-hadi/dynamic-pricing-dashboard.git"`)

Leave out `[dashboard]` for the NumPy-only simulation core. Prebuilt
wheels are attached to [GitHub
releases](https://github.com/mohammadi-hadi/dynamic-pricing-dashboard/releases),
and the dashboard is also published as a container image on GitHub
Packages:

```bash
docker run --rm -p 8501:8501 ghcr.io/mohammadi-hadi/dynamic-pricing-dashboard
```

## Use as a library

The simulation core has no dependencies beyond NumPy and can be used on
its own:

```python
from dynamic_pricing_lab import Scenario, simulate

out = simulate(Scenario(patience=0.99, ad_A=5, ad_theta=0.05, ad_gamma=0.998))
print(out["revenue"].sum(axis=1).mean())  # mean net season revenue
```

`simulate_fixed` and `oracle_cum` — the fixed-price benchmark behind the
regret curves — are exported too.

All replicates of a scenario are simulated simultaneously (vectorized
across seasons), so a 50-replicate, 2,000-period run takes about a
second natively and a few seconds in the browser.

## Develop

```bash
git clone https://github.com/mohammadi-hadi/dynamic-pricing-dashboard.git
cd dynamic-pricing-dashboard
pip install -e ".[dashboard]"
dynamic-pricing-lab   # or: streamlit run src/dynamic_pricing_lab/app.py
pytest                # smoke-tests for the simulation core
```

## Repository layout

| File | Purpose |
| --- | --- |
| `src/dynamic_pricing_lab/simulator.py` | Vectorized simulation core (Bernoulli & Normal demand, four bandit policies, advertising, fixed-price oracle) |
| `src/dynamic_pricing_lab/app.py` | Streamlit dashboard (scenario presets, insights, comparison tables) |
| `src/dynamic_pricing_lab/charts.py` | Matplotlib figures in the site's editorial style |
| `src/dynamic_pricing_lab/__main__.py` | `dynamic-pricing-lab` launcher for the installed dashboard |
| `index.html` | Browser host page (stlite) served via GitHub Pages, styled to match [mohammadi.cv](https://mohammadi.cv) |
| `.streamlit/config.toml` | Streamlit theme (paper background, navy accent) |
| `pyproject.toml` | Package metadata for `dynamic-pricing-lab` |
| `Dockerfile` | Dashboard container image (published to ghcr.io) |
| `.github/workflows/` | Pages deployment, CI, release wheels, container publishing |

## Citing

If you use this software or build on its model in academic work,
please cite it (GitHub's "Cite this repository" button uses
[`CITATION.cff`](CITATION.cff)):

```bibtex
@software{mohammadi_dynamic_pricing_lab_2026,
  author  = {Mohammadi, Hadi},
  title   = {Dynamic Pricing Lab},
  year    = {2026},
  version = {1.1.1},
  doi     = {10.5281/zenodo.21802866},
  url     = {https://github.com/mohammadi-hadi/dynamic-pricing-dashboard},
  license = {MIT}
}
```

## Author

Hadi Mohammadi — companion tool to a research project on dynamic
pricing with demand learning, forward-looking customers, and
advertising. MIT-licensed: reuse is welcome, but the copyright notice
must be preserved and academic use should credit the author.
