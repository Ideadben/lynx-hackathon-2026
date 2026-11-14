"""Backtests of three ideas on the dev period, each against the trend baseline.

1. Reversal after extreme days (overreaction).
2. Month-of-year seasonality, estimated only from earlier years.
3. Regime per asset: learn from the past whether a market trends or reverts, and trade it that way.

Run: uv run python YANNICK/edge_tests.py
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from strategies import trend

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "prices" / "example_prices.csv"


def _pd(prices: pl.DataFrame) -> pd.DataFrame:
    return prices.to_pandas().set_index("AsOfDate")


def _pl(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())


def _z(px: pd.DataFrame) -> pd.DataFrame:
    """Daily move in units of the trailing volatility known the day before."""
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    return ret / vol.shift(1)


# 1. Extreme-day reversal ---------------------------------------------------------------------------
def extreme_reversal(prices: pl.DataFrame, k: float = 2.0, hold: int = 5) -> pl.DataFrame:
    """Bet against every move larger than k sigma, for `hold` days."""
    z = _z(_pd(prices))
    shock = z.where(z.abs() > k, 0.0)
    return _pl(-shock.rolling(hold, min_periods=1).sum().fillna(0.0))


def reversal(prices: pl.DataFrame, days: int = 5) -> pl.DataFrame:
    """Plain reversal: bet against the last few days' move, in vol units."""
    return _pl(-_z(_pd(prices)).rolling(days).sum())


# 2. Seasonality -----------------------------------------------------------------------------------
def seasonal(prices: pl.DataFrame, min_years: int = 5, t_cap: float = 3.0) -> pl.DataFrame:
    """Expected move in the coming month, from the same month in earlier years only.

    The forecast is the t-statistic of that month's mean daily move, so months with a consistent
    effect count more than months with one big year. Only years before the current one are used.
    """
    z = _z(_pd(prices))
    # The position decided at T earns the move two trading days later; use the month of T + 3 days.
    earn_month = (z.index + pd.Timedelta(days=3)).month
    sums = z.groupby([z.index.year, z.index.month]).sum()
    sq = (z**2).groupby([z.index.year, z.index.month]).sum()
    cnt = z.notna().groupby([z.index.year, z.index.month]).sum()
    out = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    for year in sorted(set(z.index.year)):
        past = sums.index.get_level_values(0) < year
        if len(set(sums.index.get_level_values(0)[past])) < min_years:
            continue
        s = sums[past].groupby(level=1).sum()
        q = sq[past].groupby(level=1).sum()
        n = cnt[past].groupby(level=1).sum()
        mean = s / n
        se = np.sqrt((q / n - mean**2) / n)
        t = (mean / se).clip(-t_cap, t_cap)
        rows = z.index.year == year
        out.loc[rows] = t.reindex(earn_month[rows]).to_numpy()
    return _pl(out)


# 3. Regime per asset ------------------------------------------------------------------------------
def regime(prices: pl.DataFrame, window: int = 750, horizon: int = 5, scale: float = 10.0) -> pl.DataFrame:
    """Trend where a market has been trending, reversal where it has been reverting.

    The 'who is trading this market' question: where herding traders dominate, moves continue;
    where liquidity providers dominate, they revert. We measure it per asset with the correlation
    between the last `horizon` days' move and the next `horizon` days' move, over a trailing window
    that ends before today, and use it to weight a trend and a reversal forecast.
    """
    px = _pd(prices)
    z = _z(px)
    past_move = z.rolling(horizon).sum()
    next_move = past_move.shift(-horizon)
    # Correlation of past and next move, using only pairs whose 'next' window has already ended.
    autocorr = past_move.shift(horizon).rolling(window, min_periods=window // 2).corr(past_move)
    w_trend = np.tanh(scale * autocorr).clip(lower=0.0).fillna(0.0)
    w_rev = np.tanh(-scale * autocorr).clip(lower=0.0).fillna(0.0)
    del next_move
    tr = _pd(trend.forecast(prices))
    rv = -z.rolling(10).sum() / np.sqrt(10)
    rv = rv.clip(-2, 2) / 2  # same rough scale as the trend vote, -1..1
    return _pl(w_trend * tr + w_rev * rv)


def regime_plus_trend(prices: pl.DataFrame) -> pl.DataFrame:
    """Trend everywhere, plus reversal only where the market has been reverting."""
    px = _pd(prices)
    z = _z(px)
    past_move = z.rolling(5).sum()
    autocorr = past_move.shift(5).rolling(750, min_periods=375).corr(past_move)
    w_rev = np.tanh(-10 * autocorr).clip(lower=0.0).fillna(0.0)
    tr = _pd(trend.forecast(prices))
    rv = (-z.rolling(10).sum() / np.sqrt(10)).clip(-2, 2) / 2
    return _pl((1 - w_rev) * tr + w_rev * rv)


def main() -> None:
    pd.set_option("display.width", 200)
    cols = ["adjusted_sharpe", "sharpe", "sharpe_ex_slippage", "av_holding"]
    for scoring in ("pdf", "lynx"):
        bt = Backtest(PRICES, scoring=scoring)
        print(f"\n===== scoring={scoring}, dev period =====")
        print("\n-- 1. extreme-day reversal sweep")
        print(bt.sweep(extreme_reversal, k=[1.5, 2.0, 2.5], hold=[1, 3, 5, 10]))
        cmp = bt.compare(
            {
                "trend": trend.forecast,
                "reversal 5d": reversal,
                "extreme k2 h5": partial(extreme_reversal, k=2.0, hold=5),
                "seasonal": seasonal,
                "regime": regime,
                "trend+regime rev": regime_plus_trend,
            }
        )
        print("\n-- head to head")
        print(cmp.figures)
        print("\n-- return correlations")
        print(cmp.correlations)
        if scoring == "pdf":
            for name in ("seasonal", "regime", "trend+regime rev"):
                print(f"\n-- {name} by instrument")
                print(cmp.results[name].by_instrument())
            print("\n-- seasonal by year")
            print(cmp.results["seasonal"].by_year())
            print("\n-- seasonal sweep")
            print(bt.sweep(seasonal, min_years=[3, 5, 8], t_cap=[2.0, 3.0]))
            print("\n-- regime sweep")
            print(bt.sweep(regime, window=[500, 750, 1000], horizon=[5, 10]))
    del cols


if __name__ == "__main__":
    main()
