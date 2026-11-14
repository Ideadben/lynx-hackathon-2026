"""GARCH volatility timing on equal thirds, using Søren's garch_lib (Søren/garch_lib.py).

Each part's forecast is multiplied by clip((long-run vol / GARCH vol) ** power, 0.5, 2) per asset,
so exposure drops within days when GARCH expects turbulence. The backtest still divides by the
100-day rolling vol afterwards, so the net tilt is roughly rolling vol / GARCH vol.

Settings are Søren's defaults (GARCH(1,1), t errors, refit every 63 days on an expanding window,
power 1, floor 0.5, cap 2); nothing is tuned here. A fast EWMA vol (span 20) in place of GARCH is
the control: if it does as well, GARCH itself is not what helps.

Windows (chronological, as in the 60/20/20 Tikhonov study):
    dev      first 75%            (what the strategies were built on)
    val      60%-80% of dates      (2008-2012)
    test     last 20%              (2012-2016; the thirds strategy has been scored here before)

Run: uv run python YANNICK/garch_timing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "Søren"))
import garch_lib  # noqa: E402
from backtest import Backtest  # noqa: E402
from backtest.stats import adjusted_sharpe, sharpe  # noqa: E402
from data import MACRO, PRICES  # noqa: E402
from macro_strategy import long_stocks_bonds, macro_prior, slow_trend  # noqa: E402
from portfolio import Portfolio  # noqa: E402

_TIMING: dict[tuple, pl.DataFrame] = {}


def timing(prices: pl.DataFrame, method: str = "garch", power: float = 1.0) -> pl.DataFrame:
    """Exposure multiplier per asset and day, from data up to that day only."""
    import hashlib

    # Key on every price: the leakage test builds several altered frames of the same length.
    digest = hashlib.sha1(prices.drop("AsOfDate").to_numpy().tobytes()).hexdigest()
    key = (method, power, prices.height, digest)
    if key in _TIMING:
        return _TIMING[key]
    if method == "garch":
        out = garch_lib.vol_timing(prices, power=power)
    else:  # EWMA control: long-run = expanding mean of the fast vol
        px = prices.to_pandas().set_index("AsOfDate")
        fast = np.sqrt((px.diff() ** 2).ewm(span=20, min_periods=20).mean())
        ref = fast.expanding(min_periods=250).mean()
        mult = ((ref / fast) ** power).clip(0.5, 2.0).fillna(1.0)
        out = pl.from_pandas(mult.rename_axis("AsOfDate").reset_index())
    _TIMING[key] = out
    return out


def timed(f, method: str = "garch", power: float = 1.0):
    def g(prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
        fc = f(prices, *extras)
        m = timing(prices, method, power)
        cols = [c for c in fc.columns if c != "AsOfDate"]
        return fc.select("AsOfDate").with_columns((fc[c] * m[c]).alias(c) for c in cols)

    g.__name__ = f"{getattr(f, '__name__', 'f')}_{method}"
    return g


def thirds(method: str | None = None, power: float = 1.0) -> Portfolio:
    parts = {"t": slow_trend, "l": long_stocks_bonds, "m": macro_prior}
    if method:
        parts = {k: timed(v, method, power) for k, v in parts.items()}
    return Portfolio(parts, {k: 1 / 3 for k in parts})


def window_stats(r: pd.Series, start, end) -> dict:
    x = r[(r.index >= start) & (r.index < end)].dropna()
    v = x.to_numpy()
    return {"adj": float(adjusted_sharpe(v)), "sharpe": float(sharpe(v)), "kurt": float(x.kurt()), "skew": float(x.skew())}


def main() -> None:
    pd.set_option("display.width", 200)
    bt = Backtest(PRICES, MACRO)
    dates = pd.to_datetime(bt.prices["AsOfDate"].to_pandas())
    n = len(dates)
    windows = {"dev": (dates[0], dates[int(n * 0.75)]), "val": (dates[int(n * 0.6)], dates[int(n * 0.8)]),
               "test": (dates[int(n * 0.8)], dates[n - 1] + pd.Timedelta(days=1))}
    for k, (a, b) in windows.items():
        print(f"{k:<5} {a.date()} .. {b.date()}")

    cases = {
        "thirds (100-day rolling vol)": thirds(),
        "thirds x GARCH timing": thirds("garch"),
        "thirds x GARCH timing, power 0.5": thirds("garch", 0.5),
        "thirds x fast EWMA timing (control)": thirds("ewma"),
    }
    rows = []
    for name, P in cases.items():
        res = bt.run_positions(P, period="full", name=name)  # leakage-checked
        r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        for w, (a, b) in windows.items():
            rows.append({"strategy": name, "window": w, **window_stats(r, a, b)})
        print(f"  done: {name}  (holding {res.av_holding:.0f}d)", flush=True)
    t = pd.DataFrame(rows).pivot(index="strategy", columns="window", values=["adj", "sharpe", "kurt", "skew"])
    t = t.reindex(columns=["dev", "val", "test"], level=1).reindex(list(cases))
    print("\nadjusted Sharpe, plain Sharpe, kurtosis (excess) and skew per window:")
    print(t.round(2).to_string())

    print("\nGARCH vs rolling vol as a forecast of next-day |price change| (dev, all assets pooled):")
    gv = garch_lib.garch_vol(bt.prices.head(int(n * 0.75))).to_pandas().set_index("AsOfDate")
    px = bt.prices.head(int(n * 0.75)).to_pandas().set_index("AsOfDate")
    nxt = px.diff().shift(-1).abs()
    roll = np.sqrt((px.diff() ** 2).rolling(100).mean())
    for name, v in {"GARCH(1,1)": gv, "rolling 100d": roll}.items():
        both = pd.concat([v.stack(), nxt.stack()], axis=1).dropna()
        ratio = both.iloc[:, 1] / both.iloc[:, 0]
        print(f"  {name:<13} corr(forecast vol, next |change|) {both.corr().iloc[0, 1]:.3f}   "
              f"spread of |change| / forecast vol {ratio.std():.3f}  (lower = steadier risk)")


if __name__ == "__main__":
    main()
