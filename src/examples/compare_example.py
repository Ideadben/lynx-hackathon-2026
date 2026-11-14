"""Putting strategies side by side, and testing whether one really beats another.

Run it with: uv run python src/examples/compare_example.py
"""

from pathlib import Path

import polars as pl

from backtest import Backtest, paired_bootstrap
from backtest.pipeline import instruments
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "prices_old_example" / "example_prices.csv"


def reversal(prices: pl.DataFrame) -> pl.DataFrame:
    """Bet against the move of the last 5 days."""
    return prices.with_columns(-pl.col(instruments(prices)).diff(5))


def long_only(prices: pl.DataFrame) -> pl.DataFrame:
    """Always long everything. A simple baseline."""
    return prices.with_columns(pl.lit(1.0).alias(c) for c in instruments(prices))


def main() -> None:
    bt = Backtest(PRICES)
    comparison = bt.compare(
        {"trend": trend.forecast, "reversal": reversal, "long only": long_only}
    )

    # Key figures for each strategy, and how correlated their daily returns are.
    print(comparison)

    # Overlapping intervals say little. Compare on the same resampled days instead.
    trend_returns = comparison.results["trend"].returns
    for other in ("reversal", "long only"):
        gap = paired_bootstrap(trend_returns, comparison.results[other].returns)
        print(f"\ntrend minus {other}: {gap}")


if __name__ == "__main__":
    main()
