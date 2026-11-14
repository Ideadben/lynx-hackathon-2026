"""Does penalising far-away days more help, once α is tuned the same way for every variant?

Second-order Tikhonov (the best order), Γ = [10 * D2 (optionally scaled by lag) ; diag(d_j)], with:
    flat                       d_j = 0.1                      (current)
    ridge growing linearly     d_j = 0.1 (1 + j/50)
    ridge growing quadratic    d_j = 0.1 (1 + j/50)^2
    ridge growing exp(j/100)   d_j = 0.1 exp(j/100)
    ridge growing exp(j/50)    d_j = 0.1 exp(j/50)
    smoothness growing         curvature rows scaled by (1 + j/50), d_j = 0.1
    both growing               smoothness growing and ridge growing linearly
For each, a full α sweep. Compared at the same number of effective parameters (so no variant wins by
a better α), plus the best α on dev blocks and on validation. Expanding window, yearly refit, L = 250,
H = 5. Test block not used.

Writes YANNICK/output/tikhonov_lagweight.png and .csv.
Run: uv run python YANNICK/tikhonov_lagweight.py
"""

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from tikhonov_gamma import J, L, first_diff, normal_equations, second_diff  # noqa: F401
from tikhonov_overfit import prepare

ALPHAS = np.logspace(-2, 6, 25)
VARIANTS = {
    "flat (current)": (second_diff(), 0.1 * np.ones(L)),
    "ridge growing linearly": (second_diff(), 0.1 * (1 + J / 50)),
    "ridge growing quadratically": (second_diff(), 0.1 * (1 + J / 50) ** 2),
    "ridge growing exp(j/100)": (second_diff(), 0.1 * np.exp(J / 100)),
    "ridge growing exp(j/50)": (second_diff(), 0.1 * np.exp(J / 50)),
    "smoothness growing": (second_diff(1 + J / 50), 0.1 * np.ones(L)),
    "both growing": (second_diff(1 + J / 50), 0.1 * (1 + J / 50)),
}
TARGETS = (3, 8, 15, 25, 40)


def main() -> None:
    import matplotlib.pyplot as plt

    pd.set_option("display.width", 220)
    bt = Backtest(PRICES, MACRO)
    X, y, usable, years = prepare(bt.prices)
    NE = normal_equations(X, y, usable, years)
    assets = [c for c in bt.prices.columns if c != "AsOfDate"]
    d = pd.DatetimeIndex(pd.to_datetime(bt.prices["AsOfDate"].to_pandas()))
    n = len(d)
    val = (d[int(n * .75)], d[int(n * .9)])

    def score(G, alpha):
        GtG = G.T @ G
        fc = np.full(y.shape, np.nan)
        dfs = []
        for year, (XtX, Xty, ny) in NE.items():
            A = XtX + alpha * np.trace(XtX) / L / ny * GtG
            w = np.linalg.solve(A, Xty)
            dfs.append(np.trace(np.linalg.solve(A, XtX)))
            fc[years == year] = X[years == year] @ w
        fr = bt.prices.select("AsOfDate").with_columns(pl.Series(a, fc[:, i], nan_to_null=True) for i, a in enumerate(assets))
        r = bt.run(lambda p, *e: fr, period="full", check_leakage=False).returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        cut = lambda a, b: float(adjusted_sharpe(r[(r.index >= a) & (r.index < b)].dropna().to_numpy()))  # noqa: E731
        yr = r[r.index.year >= 2011]
        return {"dev": cut(pd.Timestamp("2003-01-01"), pd.Timestamp("2011-01-01")), "val": cut(*val),
                "post": yr.groupby(yr.index.year).apply(lambda x: float(sharpe(x.dropna().to_numpy()))).mean(),
                "df": float(np.mean(dfs))}

    rows = []
    for name, (D, dvec) in VARIANTS.items():
        G = np.vstack([10 * D, np.diag(dvec)])
        for a in ALPHAS:
            rows.append({"variant": name, "alpha": a, **score(G, a)})
        print(f"  done: {name}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv("YANNICK/output/tikhonov_lagweight.csv", index=False)

    summary = {}
    for name, g in t.groupby("variant", sort=False):
        bd, bv = g.loc[g["dev"].idxmax()], g.loc[g["val"].idxmax()]
        s = {"best dev": f"{bd.dev:+.2f} (val {bd.val:+.2f}, {bd.df:.0f}p)", "best val": f"{bv.val:+.2f} (dev {bv.dev:+.2f}, {bv.df:.0f}p)"}
        for k in TARGETS:  # interpolate dev and val at k effective parameters
            gg = g.sort_values("df")
            s[f"{k}p dev/val"] = f"{np.interp(k, gg.df, gg.dev):+.2f} / {np.interp(k, gg.df, gg.val):+.2f}"
        summary[name] = s
    print("\nAt the same number of effective parameters (p), interpolated along each α sweep: dev blocks / validation")
    print(pd.DataFrame(summary).T.to_string())

    # Figure: dev and validation against effective parameters
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), facecolor=SURF)
    for a in ax:
        a.set_facecolor(SURF)
        a.grid(True, color=GRID, linewidth=0.6)
        for s_ in a.spines.values():
            s_.set_visible(False)
        a.tick_params(colors=INK2, labelsize=8)
        a.set_xscale("log")
        a.set_xlim(1, 120)
        a.axhline(0, color=INK2, linewidth=0.8)
        a.set_xlabel("effective parameters (less smoothing →)", color=INK2, fontsize=9)
    for (name, g), col in zip(t.groupby("variant", sort=False), colors):
        g = g[g["df"] >= 1].sort_values("df")
        w = 2.6 if name == "flat (current)" else 1.5
        ax[0].plot(g["df"], g["dev"], color=col, linewidth=w, label=name)
        ax[1].plot(g["df"], g["val"], color=col, linewidth=w, label=name)
    ax[0].set_title("Dev blocks 2003–2010, adjusted Sharpe", color=INK, fontsize=11, loc="left")
    ax[1].set_title("Validation Jun 2011 – Oct 2014, adjusted Sharpe", color=INK, fontsize=11, loc="left")
    ax[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig("YANNICK/output/tikhonov_lagweight.png", dpi=120)
    print("\nwrote YANNICK/output/tikhonov_lagweight.png and .csv")


if __name__ == "__main__":
    main()
