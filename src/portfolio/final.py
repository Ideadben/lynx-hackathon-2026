"""The final portfolio we submit, and the command that writes its positions.

    uv run python -m portfolio.final                       # in-sample data, writes positions.csv
    uv run python -m portfolio.final --prices new.csv --macro new_macro.csv --out positions.csv

Weights (each strategy goes through the risk pipeline on its own, then the blend is
risk-scaled once more, so a weight is a share of risk):

    10%  tikhonov_garch      Tikhonov lag filter (alpha = 46) with GARCH vol timing as risk management
    30%  slow_trend          \
    30%  long_stocks_bonds    > the triple strategy, listed part by part so each is vol-scaled once
    30%  macro               /

The 10% Tikhonov share is the one with the best adjusted Sharpe over dev + validation (1995 - Oct
2014) in a sweep from 0% to 60% in 5% steps, the triple keeping equal thirds of the rest
(YANNICK/portfolio_dashboard.py). The test block was scored once, afterwards.

The whole blend runs through the backtester's leakage test before anything is written. The saved
file has ``AsOfDate`` plus one column per instrument, for every date; warm-up rows are empty.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from portfolio.blend import Portfolio
from strategies import tikhonov, triple

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "prices" / "close_prices_insample.csv"
MACRO = ROOT / "prices" / "macro_data_insample.csv"

STRATEGIES = {
    "tikhonov_garch": tikhonov.forecast_garch,
    "slow_trend": triple.slow_trend,
    "long_stocks_bonds": triple.long_stocks_bonds,
    "macro": triple.macro,
}
WEIGHTS = {"tikhonov_garch": 0.1, "slow_trend": 0.3, "long_stocks_bonds": 0.3, "macro": 0.3}

# Report windows, as fractions of the dates: the 75 / 15 / 10 dev / validation / test split.
WINDOWS = {"dev": (0.0, 0.75), "validation": (0.75, 0.90), "test": (0.90, 1.0)}


def build() -> Portfolio:
    """The final blend, as a positions function for ``Backtest.run_positions``."""
    return Portfolio(STRATEGIES, WEIGHTS)


def _windows(returns: pl.DataFrame, dates: pl.Series) -> dict[str, float]:
    r = returns.to_pandas().set_index(returns.columns[0])[returns.columns[1]]
    r.index = pd.to_datetime(r.index)
    d = pd.to_datetime(dates.to_pandas())
    out = {}
    for name, (a, b) in WINDOWS.items():
        lo, hi = d.iloc[int(len(d) * a)], d.iloc[min(int(len(d) * b), len(d) - 1)]
        x = r[(r.index >= lo) & ((r.index < hi) if b < 1 else (r.index <= hi))].dropna().to_numpy()
        out[f"{name} adj"] = float(adjusted_sharpe(x)) if len(x) > 50 else np.nan
        out[f"{name} sharpe"] = float(sharpe(x)) if len(x) > 50 else np.nan
    return out


def report(bt: Backtest, final) -> None:
    """Adjusted and plain Sharpe per window for the blend, the triple, and each strategy."""
    rows = {"FINAL (40 / 20 / 20 / 20)": final}
    rows["triple alone"] = bt.run_positions(triple.portfolio(), period="full", check_leakage=False, name="triple")
    for name, f in STRATEGIES.items():
        rows[name] = bt.run(f, period="full", check_leakage=False, name=name)
    rows["tikhonov without GARCH"] = bt.run(tikhonov.forecast, period="full", check_leakage=False, name="tikhonov")
    table = pd.DataFrame({k: {**_windows(r.returns, bt.prices["AsOfDate"]), "holding (days)": r.av_holding}
                          for k, r in rows.items()}).T
    pd.set_option("display.width", 200)
    print("\nAdjusted and plain Sharpe per window (75 / 15 / 10 split; PDF costs):")
    print(table.round(2).to_string())
    rets = pd.DataFrame({k: r.returns.to_pandas().set_index("AsOfDate")["returns"] for k, r in rows.items() if k in STRATEGIES})
    print("\nCorrelation of the strategies' daily returns:")
    print(rets.corr().round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the final positions.")
    parser.add_argument("--prices", default=str(PRICES), help="Close prices csv (AsOfDate + one column per instrument).")
    parser.add_argument("--macro", default=str(MACRO), help="Macro csv (GDP and CPI per planet).")
    parser.add_argument("--out", default="positions.csv", help="Where to write the positions.")
    parser.add_argument("--no-report", action="store_true", help="Skip the per-strategy report.")
    args = parser.parse_args()

    bt = Backtest(args.prices, args.macro)
    final = bt.run_positions(build(), period="full", name="final")  # runs the leakage test on the whole blend
    final.to_csv(args.out)
    print(f"Wrote {args.out}: {final.positions.height} rows x {final.positions.width - 1} instruments. Leakage test passed.")
    print(final)
    if not args.no_report:
        report(bt, final)


if __name__ == "__main__":
    main()
