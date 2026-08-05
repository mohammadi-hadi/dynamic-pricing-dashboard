"""Dynamic Pricing Lab — dynamic pricing with demand learning.

Vectorized simulator for a monopolist who prices by Thompson sampling
against myopic or forward-looking customers, with an optional one-time
advertising effort whose willingness-to-pay lift decays over the season.

The core API needs nothing beyond NumPy:

    from dynamic_pricing_lab import Scenario, simulate

    out = simulate(Scenario(patience=0.99, ad_A=5, ad_theta=0.05))
    print(out["revenue"].sum(axis=1).mean())  # mean net season revenue

The Streamlit dashboard is an optional extra: install
``dynamic-pricing-lab[dashboard]`` and run ``dynamic-pricing-lab``.
"""

from dynamic_pricing_lab.simulator import (
    Scenario,
    oracle_cum,
    simulate,
    simulate_fixed,
)

__version__ = "1.0.0"

__all__ = ["Scenario", "simulate", "simulate_fixed", "oracle_cum", "__version__"]
