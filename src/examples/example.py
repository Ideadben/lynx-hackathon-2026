"""A tour of the backtester. Run it with: uv run python src/examples/example.py"""

from functools import partial
from pathlib import Path

import polars as pl

from backtest import Backtest
from backtest.pipeline import instruments, risk_scale, to_positions
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "example_prices.csv"
OUTPUT = HERE / "output"


def reversal(prices: pl.DataFrame, days: int = 5) -> pl.DataFrame:
    """A simple forecast: bet against the move of the last few days."""
    return prices.with_columns(-pl.col(instruments(prices)).diff(days))


def single_speed(prices: pl.DataFrame, span: int) -> pl.DataFrame:
    """Trend at one speed, with a keyword parameter so that it works with sweep."""
    return trend.forecast(prices, spans=(span,))


def blend(prices: pl.DataFrame) -> pl.DataFrame:
    """Final positions: half trend, half reversal, with risk rescaled once at the end."""
    first = to_positions(trend.forecast(prices), prices)
    second = to_positions(reversal(prices), prices)
    mixed = first.with_columns((pl.col(c) + second[c]) / 2 for c in instruments(prices))
    return risk_scale(mixed, prices)


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)

    # One Backtest per data source. It scores the development period by default.
    bt = Backtest(PRICES)

    # Run one strategy, and read its numbers.
    result = bt.run(trend.forecast, name="trend")
    print(result)
    print(f"\nAdjusted Sharpe: {result.adjusted_sharpe:.2f}")

    # Break the result down.
    print(result.by_year())
    print(result.by_instrument())

    # Set a parameter.
    slower = bt.run(partial(single_speed, span=32), name="span 32")
    print(f"\nSharpe with span 32: {slower.sharpe:.2f}")

    # Try a range of parameters, to see if the result is stable.
    print(bt.sweep(single_speed, span=[8, 16, 32]))

    # Test choosing the span from the past, as live trading would have to.
    walk = bt.walk_forward(single_speed, span=[8, 12, 16, 20, 32])
    print(walk)

    # Put strategies side by side, and see how correlated they are.
    print(bt.compare({"trend": trend.forecast, "reversal": reversal}))

    # Score final positions, like the blend. Passing a function runs the leakage test.
    blended = bt.run_positions(blend)
    print(f"\n{blended}")

    # Check a fast strategy under Lynx's costs too.
    lynx = Backtest(PRICES, scoring="lynx")
    print(f"\nReversal Sharpe, PDF costs:  {bt.run(reversal).sharpe:.2f}")
    print(f"Reversal Sharpe, Lynx costs: {lynx.run(reversal).sharpe:.2f}")

    # Save the dashboard and the positions.
    result.plot().savefig(OUTPUT / "trend_dashboard.png")
    result.to_csv(OUTPUT / "trend_positions.csv")
    print(f"\nSaved the dashboard and positions to {OUTPUT}")


if __name__ == "__main__":
    main()
