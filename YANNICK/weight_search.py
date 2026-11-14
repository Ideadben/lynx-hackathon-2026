"""Which share of risk should each strategy get?

Strategies: tikhonov_garch, slow_trend, long_stocks_bonds, macro (as in src/portfolio/final.py).

1. Grid: every mix in 10% steps (286). Each strategy's positions are computed once; a mix is their
   weighted sum, risk-scaled once more, exactly as Portfolio does. Scored on dev (adjusted Sharpe,
   and mean Sharpe per calendar year 1997-2010), with a neighbourhood average to find robust regions.
2. Rules that choose weights from past returns every quarter (Portfolio(choose=...)): honest by
   construction.
3. Validation for the candidates. Test only for the final pick (and mixes already seen on test).

Run: uv run python YANNICK/weight_search.py
"""

import itertools
import time

import numpy as np
import pandas as pd

from backtest import Backtest
from backtest.choose import max_sharpe
from backtest.pipeline import risk_scale
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from portfolio import Portfolio, risk_parity, shrunk_max_sharpe
from portfolio.blend import _weighted_sum
from portfolio.final import STRATEGIES

NAMES = list(STRATEGIES)
SHORT = {"tikhonov_garch": "tik", "slow_trend": "trend", "long_stocks_bonds": "long", "macro": "macro"}


def main() -> None:
    pd.set_option("display.width", 220)
    bt = Backtest(PRICES, MACRO)
    dates = pd.DatetimeIndex(pd.to_datetime(bt.prices["AsOfDate"].to_pandas()))
    n = len(dates)
    win = {"dev": (dates[0], dates[int(n * .75)]), "val": (dates[int(n * .75)], dates[int(n * .9)]),
           "test": (dates[int(n * .9)], dates[-1] + pd.Timedelta(days=1))}

    t0 = time.time()
    positions = Portfolio(STRATEGIES).positions(bt.prices, bt.extras)
    print(f"positions for {len(positions)} strategies ({time.time() - t0:.0f}s)")

    def returns_of(frame) -> pd.Series:
        r = bt.run_positions(frame, period="full", check_leakage=False).returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        return r

    def blend(w: dict) -> pd.Series:
        import polars as pl

        rows = np.tile([w.get(k, 0.0) for k in positions], (bt.prices.height, 1))
        mixed = _weighted_sum(positions, rows)
        # An instrument no chosen strategy trades is empty all the way through, and one all-empty
        # column makes risk_scale return nothing. On rows where the blend trades, empty means 0.
        cols = mixed.columns[1:]
        mixed = mixed.with_columns(pl.col(c).fill_nan(None) for c in cols)  # NaN and null differ in polars
        live = pl.any_horizontal(pl.col(c).is_not_null() for c in cols)
        mixed = mixed.with_columns(pl.when(live).then(pl.col(c).fill_null(0.0)).otherwise(None).alias(c) for c in cols)
        return returns_of(risk_scale(mixed, bt.prices))

    def stats(r: pd.Series, windows=("dev", "val")) -> dict:
        out = {}
        for k in windows:
            a, b = win[k]
            x = r[(r.index >= a) & (r.index < b)].dropna().to_numpy()
            out[f"{k} adj"] = float(adjusted_sharpe(x))
        dev = r[(r.index.year >= 1997) & (r.index.year <= 2010)]
        yearly = dev.groupby(dev.index.year).apply(lambda x: float(sharpe(x.dropna().to_numpy())))
        out["dev mean/yr"] = yearly.mean()
        out["dev worst yr"] = yearly.min()
        return out

    # 1. Grid
    t0 = time.time()
    grid = []
    steps = range(11)
    for a, b, c in itertools.product(steps, steps, steps):
        d = 10 - a - b - c
        if d < 0:
            continue
        w = dict(zip(NAMES, (a / 10, b / 10, c / 10, d / 10)))
        r = blend(w)
        grid.append({**{SHORT[k]: v for k, v in w.items()}, **stats(r, ("dev",)), "_r": r})
    print(f"grid of {len(grid)} mixes ({time.time() - t0:.0f}s)")
    g = pd.DataFrame(grid)
    # Robustness: average dev score of each mix and its neighbours (one 10% step away)
    keys = list(SHORT.values())
    W = g[keys].to_numpy()
    nb = [np.where(np.abs(W - W[i]).sum(1) <= 0.2 + 1e-9)[0] for i in range(len(W))]
    g["dev adj, neighbourhood"] = [g["dev adj"].iloc[ix].mean() for ix in nb]
    cols = keys + ["dev adj", "dev adj, neighbourhood", "dev mean/yr", "dev worst yr"]
    print("\nTop 12 mixes by dev adjusted Sharpe:")
    print(g.sort_values("dev adj", ascending=False)[cols].head(12).round(2).to_string(index=False))
    print("\nTop 12 by neighbourhood average (robust regions):")
    print(g.sort_values("dev adj, neighbourhood", ascending=False)[cols].head(12).round(2).to_string(index=False))
    print("\nAverage dev adjusted Sharpe by the share of each strategy (all mixes):")
    for k in keys:
        print(f"  {k:<6} " + "  ".join(f"{int(v*100):>3}%: {m:.2f}" for v, m in g.groupby(k)["dev adj"].mean().items()))

    # 2. Rules from past returns + named mixes
    named = {
        "triple only (0/33/33/33)": {"slow_trend": 1 / 3, "long_stocks_bonds": 1 / 3, "macro": 1 / 3},
        "current final (40/20/20/20)": {"tikhonov_garch": .4, "slow_trend": .2, "long_stocks_bonds": .2, "macro": .2},
        "equal quarters (25 each)": {k: .25 for k in NAMES},
    }
    best_dev = g.sort_values("dev adj", ascending=False).iloc[0]
    best_nb = g.sort_values("dev adj, neighbourhood", ascending=False).iloc[0]
    for label, row in (("best dev mix", best_dev), ("best robust region", best_nb)):
        named[f"{label} ({'/'.join(str(int(round(row[k] * 100))) for k in keys)})"] = {k: row[SHORT[k]] for k in NAMES}
    rows = {}
    for name, w in named.items():
        rows[name] = stats(blend(w))
    t0 = time.time()
    for name, choose in {"risk parity, refit quarterly": risk_parity(), "shrunk max Sharpe, refit quarterly": shrunk_max_sharpe(),
                         "max Sharpe, refit quarterly": max_sharpe()}.items():
        P = Portfolio(STRATEGIES, choose=choose)
        rows[name] = stats(returns_of(bt.run_positions(P, period="full", check_leakage=False).positions))
        if name.startswith("shrunk"):
            wts = P.weights(bt.prices, bt.extras).to_pandas()
            print("\nShrunk max Sharpe: chosen weights at a few dates (past returns only):")
            print(wts.iloc[np.linspace(0, len(wts) - 1, 6).astype(int)].round(2).to_string(index=False))
    print(f"\n(weight rules {time.time() - t0:.0f}s)")
    print("\nCandidates on dev and validation (test not shown):")
    print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
