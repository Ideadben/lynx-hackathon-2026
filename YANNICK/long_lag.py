"""Long-memory momentum: the Tikhonov filter put its weight on returns 100-250 days ago.

Fixed rules (nothing fitted), each a sum of vol-normalised daily returns over a window of past days:
    lagged 100-250   days t-250 .. t-100   (what the learned filter emphasised)
    12-1             days t-250 .. t-21    (classic 'skip the last month' momentum)
    12-0             days t-250 .. t       (plain one-year momentum)
    6-0              days t-125 .. t
Each is used as a sign vote and as a continuous score (sum / sqrt(window), capped at ±2).

Scored on dev, validation (75/15/10) and per calendar year; the test block is not used.
Also tried as the trend leg of equal thirds.

Run: uv run python YANNICK/long_lag.py
"""

from functools import partial

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from lag_tikhonov import z_returns
from macro_strategy import long_stocks_bonds, macro_prior, slow_trend
from portfolio import Portfolio

WINDOWS = {"lagged 100-250": (100, 250), "12-1": (21, 250), "12-0": (0, 250), "6-0": (0, 125)}


def window_momentum(prices: pl.DataFrame, *extras, start: int = 100, end: int = 250, kind: str = "sign") -> pl.DataFrame:
    """Sum of z over days t-end .. t-start (only past returns)."""
    px = prices.to_pandas().set_index("AsOfDate")
    z = z_returns(px)
    s = z.rolling(end - start + 1, min_periods=end - start + 1).sum().shift(start)
    f = np.sign(s) if kind == "sign" else (s / np.sqrt(end - start + 1)).clip(-2, 2)
    return pl.from_pandas(f.rename_axis("AsOfDate").reset_index())


def main() -> None:
    pd.set_option("display.width", 200)
    bt = Backtest(PRICES, MACRO)
    dates = pd.DatetimeIndex(pd.to_datetime(bt.prices["AsOfDate"].to_pandas()))
    n = len(dates)
    dev = (dates[0], dates[int(n * .75)])
    val = (dates[int(n * .75)], dates[int(n * .9)])

    def score(f, is_pos=False):
        res = (bt.run_positions if is_pos else bt.run)(f, period="full")  # leakage-checked
        r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        cut = lambda a, b: r[(r.index >= a) & (r.index < b)].dropna().to_numpy()  # noqa: E731
        yearly = r[(r.index.year >= 1996) & (r.index.year <= 2010)].groupby(r[(r.index.year >= 1996) & (r.index.year <= 2010)].index.year).apply(
            lambda x: float(sharpe(x.dropna().to_numpy())) if x.notna().sum() > 100 else np.nan).dropna()
        return {"dev adj": float(adjusted_sharpe(cut(*dev))), "val adj": float(adjusted_sharpe(cut(*val))),
                "mean/yr (dev)": yearly.mean(), "yrs > 0": f"{(yearly > 0).sum()}/{len(yearly)}",
                "after 2003 /yr": yearly[yearly.index >= 2003].mean(), "hold": res.av_holding}

    rows = {"slow trend 16/32/64 (current)": score(slow_trend)}
    for name, (a, b) in WINDOWS.items():
        for kind in ("sign", "cont"):
            rows[f"{name}, {kind}"] = score(partial(window_momentum, start=a, end=b, kind=kind))
    print("Single strategies (adjusted Sharpe on dev and validation; plain Sharpe per year):")
    print(pd.DataFrame(rows).T.round(2).to_string())

    def thirds(trend_leg):
        return Portfolio({"t": trend_leg, "l": long_stocks_bonds, "m": macro_prior}, {k: 1 / 3 for k in "tlm"})

    rows = {"equal thirds (slow trend leg)": score(thirds(slow_trend), True)}
    for name in ("lagged 100-250", "12-1", "12-0"):
        a, b = WINDOWS[name]
        rows[f"thirds, trend leg = {name} sign"] = score(thirds(partial(window_momentum, start=a, end=b, kind="sign")), True)
    rows["four quarters: thirds + lagged 100-250"] = score(Portfolio(
        {"t": slow_trend, "l": long_stocks_bonds, "m": macro_prior, "x": partial(window_momentum, start=100, end=250, kind="sign")},
        {k: 0.25 for k in "tlmx"}), True)
    print("\nInside equal thirds:")
    print(pd.DataFrame(rows).T.round(2).to_string())


if __name__ == "__main__":
    main()
