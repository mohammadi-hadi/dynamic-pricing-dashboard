"""Matplotlib figures for the dashboard, in the site's editorial style:
paper background, navy accent, hairline rules, left-aligned titles."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# palette shared with mohammadi.cv
PAPER = "#fafaf7"
SURFACE = "#ffffff"
INK = "#1a1a1a"
INK2 = "#444444"
INK3 = "#6e6e6e"
RULE = "#e3e1dc"
NAVY = "#1e3a5f"  # forward-looking
BRICK = "#a63d2f"  # myopic
ARM_COLORS = [NAVY, BRICK, "#5a6f43", "#b28a2e", "#6e4a6d", "#4a6e7d"]

# condition -> (color, linestyle); ads solid, no ads dashed
STYLE = {
    "Forward-looking + ads": (NAVY, "-"),
    "Forward-looking, no ads": (NAVY, (0, (5, 2))),
    "Myopic + ads": (BRICK, "-"),
    "Myopic, no ads": (BRICK, (0, (5, 2))),
}

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK3,
        "ytick.color": INK3,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.facecolor": SURFACE,
    }
)


def _ax(xlabel, ylabel, title, height=3.1):
    fig, ax = plt.subplots(figsize=(6.6, height), dpi=110)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=RULE, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def cum_revenue(results) -> plt.Figure:
    fig, ax = _ax("Period", "Cumulative net revenue",
                  "Revenue accumulation — mean and 95% band")
    for name, r in results.items():
        color, ls = STYLE[name]
        x = np.arange(1, len(r["cum_mean"]) + 1)
        ax.plot(x, r["cum_mean"], color=color, ls=ls, lw=1.6, label=name)
        ax.fill_between(x, r["cum_lo"], r["cum_hi"], color=color, alpha=0.13, lw=0)
    ax.legend(loc="upper left")
    return fig


def regret(results) -> plt.Figure:
    fig, ax = _ax("Period", "Cumulative regret",
                  "Regret against the best fixed price")
    for name, r in results.items():
        color, ls = STYLE[name]
        x = np.arange(1, len(r["regret"]) + 1)
        ax.plot(x, r["regret"], color=color, ls=ls, lw=1.6, label=name)
    ax.legend(loc="upper left")
    return fig


def shares(results, prices) -> plt.Figure:
    fig, ax = _ax("Posted price", "Share of periods (%)",
                  "How often each price is offered", height=3.0)
    names = list(results)
    width = 0.8 / len(names)
    xs = np.arange(len(prices))
    for j, name in enumerate(names):
        color, ls = STYLE[name]
        ax.bar(xs + (j - (len(names) - 1) / 2) * width, results[name]["share"],
               width=width * 0.9, color=color, alpha=1.0 if ls == "-" else 0.45,
               label=name)
    ax.set_xticks(xs, [f"{p:.2f}" for p in prices])
    ax.legend()
    return fig


def posterior(r, prices, demands, bernoulli: bool) -> plt.Figure:
    ylab = "Posterior mean of demand" if bernoulli else "Posterior mean of revenue"
    fig, ax = _ax("Period", ylab,
                  "What the seller believes (dotted = truth)", height=2.9)
    for a in range(len(prices)):
        c = ARM_COLORS[a % len(ARM_COLORS)]
        ax.plot(r["snap_periods"], r["snap_mean"][:, a], color=c, lw=1.5,
                label=f"p = {prices[a]:.2f}")
        ax.fill_between(r["snap_periods"],
                        r["snap_mean"][:, a] - r["snap_sd"][:, a],
                        r["snap_mean"][:, a] + r["snap_sd"][:, a],
                        color=c, alpha=0.13, lw=0)
        if bernoulli:
            ax.axhline(demands[a], color=c, ls=":", lw=0.9, alpha=0.75)
    ax.legend()
    return fig


def policy_mix(r, prices) -> plt.Figure:
    fig, ax = _ax("Period", "Share of offers (%)",
                  "Exploration fades into exploitation", height=2.9)
    x = r["snap_periods"]
    mix = 100.0 * r["snap_mix"].T  # (k, S)
    colors = [ARM_COLORS[a % len(ARM_COLORS)] for a in range(len(prices))]
    ax.stackplot(x, mix, colors=colors, alpha=0.85, lw=0,
                 labels=[f"p = {p:.2f}" for p in prices])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc="center right")
    return fig


def sweep(grid, curves) -> plt.Figure:
    fig, ax = _ax("Advertising effort A", "Mean net revenue",
                  "Net revenue by advertising effort — ★ marks A*", height=3.3)
    for label, (means, ses, color) in curves.items():
        means, ses = np.asarray(means), np.asarray(ses)
        ax.plot(grid, means, color=color, lw=1.7, marker="o", ms=3.5, label=label)
        ax.fill_between(grid, means - 1.96 * ses, means + 1.96 * ses,
                        color=color, alpha=0.13, lw=0)
        best = int(np.argmax(means))
        ax.plot(grid[best], means[best], marker="*", ms=15, color=color,
                mec=SURFACE, mew=0.8, zorder=5)
        ax.annotate(f"A* = {grid[best]:g}", (grid[best], means[best]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8.5,
                    color=color)
    ax.legend(loc="lower center")
    return fig
