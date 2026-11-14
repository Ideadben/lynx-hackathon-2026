"""Out-of-sample test of the candidate strategies, and of smaller instrument subsets.

Pre-registered: this list was fixed before looking at any holdout number. Do not change the
choice of final strategy based on these results, or the holdout stops being out of sample.

Scored on three windows:
    dev       first 75% of dates (what everything was developed on)
    holdout   last 25% (the backtester's own split)
    last 20%  an 80/20 split: the final 20% of dates, scored from the full run

Subsets of the equal-thirds strategy:
    class-based (not chosen from results): all 27, no FX, stocks+bonds only, commodities only,
        no Callisto (stale closes)
    hindsight: drop instruments that lost money in the thirds strategy on dev (shows the bias)
    walk-forward: each January, keep instruments whose last 3 years made money, using past data only

Run: uv run python YANNICK/oos_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import Backtest  # noqa: E402
from backtest.pipeline import risk_scale  # noqa: E402
from backtest.stats import adjusted_sharpe, sharpe  # noqa: E402
from data import MACRO, PRICES  # noqa: E402
from macro_strategy import long_stocks_bonds, macro_prior, slow_trend  # noqa: E402
from portfolio import Portfolio  # noqa: E402
from strategies import trend  # noqa: E402

ALL = [c for c in pd.read_csv(PRICES, nrows=1).columns if c != "AsOfDate"]
FX = [c for c in ALL if c.endswith("_titan_fx")]
SB = [c for c in ALL if c.endswith(("_stocks", "_bonds"))]
COMM = [c for c in ALL if c not in FX + SB]


def masked(f, keep: list[str]):
    """A strategy that forecasts 0 outside `keep`."""

    def g(prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
        out = f(prices, *extras)
        return out.with_columns(pl.lit(0.0).alias(c) for c in out.columns if c != "AsOfDate" and c not in keep)

    return g


def thirds(keep: list[str] | None = None):
    """Equal thirds. With `keep`, the same positions with the other instruments set to 0, then
    risk-scaled again. (Masking the forecasts instead breaks the pipeline: a column that is 0 in
    every part becomes NaN after normalisation, and one NaN column makes all positions NaN.)"""
    parts = {"trend": slow_trend, "long": long_stocks_bonds, "macro": macro_prior}
    full = Portfolio(parts, {k: 1 / 3 for k in parts})
    if keep is None:
        return full

    def positions(prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
        pos = full(prices, *extras)
        live = pl.any_horizontal(pl.col(c).is_not_null() for c in keep)  # rows the kept part trades
        pos = pos.with_columns(
            pl.when(live).then(0.0).otherwise(None).alias(c) for c in pos.columns if c != "AsOfDate" and c not in keep
        )
        return risk_scale(pos, prices)

    positions.__name__ = "thirds_subset"
    return positions


def walk_forward_selection(window: int = 750):
    """Thirds, but each January drop instruments whose trailing `window`-day P&L Sharpe was <= 0."""
    base = thirds()

    def positions(prices: pl.DataFrame, extras: pl.DataFrame) -> pl.DataFrame:
        pos = base(prices, extras).to_pandas().set_index("AsOfDate")
        px = prices.to_pandas().set_index("AsOfDate")
        pnl = pos.shift(2) * px.diff()  # the return each position earns, as in the scorer
        sr = pnl.rolling(window, min_periods=window).mean() / pnl.rolling(window, min_periods=window).std()
        years = pd.to_datetime(pos.index).year
        keep = pd.DataFrame(True, index=pos.index, columns=pos.columns)
        for y in sorted(set(years)):
            before = np.where(years < y)[0]
            if len(before) == 0:
                continue
            last = sr.iloc[before[-1]]
            keep.loc[years == y] = np.broadcast_to((last > 0) | last.isna(), (int((years == y).sum()), len(last)))
        out = pos.where(keep | pos.isna(), 0.0)  # leave warm-up rows empty, as in the full strategy
        frame = pl.from_pandas(out.rename_axis("AsOfDate").reset_index())
        return risk_scale(frame, prices)

    positions.__name__ = "walk_forward_selection"
    return positions


def scores(bt: Backtest, f, positions: bool) -> dict:
    run = bt.run_positions if positions else bt.run
    dev = run(f, period="dev")
    hold = run(f, period="holdout", check_leakage=False)
    full = run(f, period="full", check_leakage=False)
    r = full.returns.to_pandas().set_index("AsOfDate")["returns"]
    cut = pd.Timestamp(bt.prices["AsOfDate"][int(bt.prices.height * 0.8)])  # 80% of the price dates, not of the returns
    last = r[r.index >= cut].dropna().to_numpy()
    return {
        "dev adj": dev.adjusted_sharpe,
        "holdout adj": hold.adjusted_sharpe,
        "holdout sharpe": hold.sharpe,
        "last20 adj": float(adjusted_sharpe(last)),
        "last20 sharpe": float(sharpe(last)),
        "hold days": hold.av_holding,
    }


def main() -> None:
    pd.set_option("display.width", 200)
    bt = Backtest(PRICES, MACRO)
    print(f"dev ends {bt.prices['AsOfDate'][int(bt.prices.height * 0.75) - 1]}, "
          f"80/20 test starts {bt.prices['AsOfDate'][int(bt.prices.height * 0.8)]}, data ends {bt.prices['AsOfDate'][-1]}")

    # Hindsight subset: instruments that lost money in thirds on dev (dev data only)
    dev_thirds = bt.run_positions(thirds(), period="dev", name="thirds")
    per = dev_thirds.by_instrument().to_pandas().set_index("instrument")["sharpe_ex_slippage"]
    winners = per[per > 0].index.tolist()
    print(f"hindsight subset keeps {len(winners)} of 27; drops {sorted(set(ALL) - set(winners))}\n")

    candidates = {
        "trend.py (default)": (lambda prices, *extras: trend.forecast(prices), False),
        "slow trend": (slow_trend, False),
        "long stocks+bonds": (long_stocks_bonds, False),
        "macro prior": (macro_prior, False),
        "base: slow trend 50 / long 50": (Portfolio({"t": slow_trend, "l": long_stocks_bonds}, {"t": .5, "l": .5}), True),
        "35 / 35 / 30": (Portfolio({"t": slow_trend, "l": long_stocks_bonds, "m": macro_prior}, {"t": .35, "l": .35, "m": .3}), True),
        "THIRDS, all 27": (thirds(), True),
        "thirds, no FX (23)": (thirds([c for c in ALL if c not in FX]), True),
        "thirds, stocks+bonds only (10)": (thirds(SB), True),
        "thirds, commodities only (13)": (thirds(COMM), True),
        "thirds, no Callisto (24)": (thirds([c for c in ALL if not c.startswith("callisto")]), True),
        f"thirds, hindsight winners ({len(winners)})": (thirds(winners), True),
        "thirds, walk-forward selection": (walk_forward_selection(), True),
    }
    rows = {}
    for name, (f, is_pos) in candidates.items():
        rows[name] = scores(bt, f, is_pos)
        print(f"  done: {name}", flush=True)
    table = pd.DataFrame(rows).T.round(2)
    print("\nPDF scoring\n" + table.to_string())

    lynx = Backtest(PRICES, MACRO, scoring="lynx")
    print("\nLynx scoring, holdout adjusted Sharpe:")
    for name in ("trend.py (default)", "base: slow trend 50 / long 50", "THIRDS, all 27", "thirds, no FX (23)", "thirds, walk-forward selection"):
        f, is_pos = candidates[name]
        run = lynx.run_positions if is_pos else lynx.run
        print(f"  {name:<34} {run(f, period='holdout', check_leakage=False).adjusted_sharpe:.2f}")


if __name__ == "__main__":
    main()
