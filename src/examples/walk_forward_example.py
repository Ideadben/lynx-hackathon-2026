"""Tests choosing a parameter from past data only.

Run it with: uv run python src/examples/walk_forward_example.py
"""

from datetime import date
from pathlib import Path

import polars as pl

from backtest import Backtest
from backtest.pipeline import instruments
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "example_prices.csv"
OUTPUT = HERE / "output"


def single_speed(prices: pl.DataFrame, span: int) -> pl.DataFrame:
    """Trend at one speed. Walk-forward passes the parameter as a keyword argument."""
    return trend.forecast(prices, spans=(span,))


def only_between(positions: pl.DataFrame, first: date, last: date) -> pl.DataFrame:
    """Keeps positions between two dates, so strategies can be scored on the same days."""
    inside = pl.col("AsOfDate").is_between(first, last)
    return positions.with_columns(
        pl.when(inside).then(pl.col(c)).alias(c) for c in instruments(positions)
    )


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    bt = Backtest(PRICES)

    # Use a wide grid: choosing the grid after seeing results brings hindsight back.
    spans = [4, 8, 12, 16, 20, 24, 32, 48, 64]
    walk = bt.walk_forward(single_speed, span=spans)

    # The honest score: every traded day used a span chosen from earlier data.
    print(walk.result)

    # What was chosen before each block. If this jumps around, the choice is mostly noise.
    print(walk.choices)

    # Each span held fixed over the same days. The best row is what hindsight would pick.
    print(walk.fixed.select("span", "adjusted_sharpe", "sharpe", "av_holding"))

    hindsight = walk.fixed["adjusted_sharpe"].max()
    honest = walk.result.adjusted_sharpe
    print(f"\nHindsight:    {hindsight:.2f}")
    print(f"Walk-forward: {honest:.2f}")
    print(f"Flattered by: {hindsight - honest:.2f}")

    # Compare with a strategy that never chooses a span, on the same days.
    first, last = walk.choices["first_day"].min(), walk.choices["last_day"].max()
    averaged = bt.run(trend.forecast).positions
    same_days = bt.run_positions(only_between(averaged, first, last), name="trend")
    print(f"Averaged spans, same days: {same_days.adjusted_sharpe:.2f}")

    # The stitched strategy is a normal result, so every report works on it.
    print(walk.result.by_year())

    # Smaller blocks choose more often. A larger first_train gives more history to start.
    frequent = bt.walk_forward(single_speed, span=spans, first_train=0.5, blocks=24)
    print(f"\nHalf the data to start, 24 blocks: {frequent.result.adjusted_sharpe:.2f}")

    walk.result.plot().savefig(OUTPUT / "walk_forward_dashboard.png")
    print(f"\nSaved the dashboard to {OUTPUT}")


if __name__ == "__main__":
    main()
