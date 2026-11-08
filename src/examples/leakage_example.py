"""What happens when a strategy uses future data by mistake.

Run it with: uv run python src/examples/leakage_example.py
"""

from pathlib import Path

import polars as pl

from backtest import Backtest
from backtest.leakage import LeakageError
from backtest.pipeline import instruments

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "example_prices.csv"


def peeks_ahead(prices: pl.DataFrame) -> pl.DataFrame:
    """Wrong: the shift(-1) uses tomorrow's price change today."""
    return prices.with_columns(pl.col(instruments(prices)).diff().shift(-1).sign())


def uses_whole_history(prices: pl.DataFrame) -> pl.DataFrame:
    """Wrong: the average over all dates includes the future."""
    return prices.with_columns(
        pl.col(c) - pl.col(c).mean() for c in instruments(prices)
    )


def honest(prices: pl.DataFrame) -> pl.DataFrame:
    """Right: an average of the last 50 days only."""
    return prices.with_columns(
        pl.col(c) - pl.col(c).rolling_mean(50) for c in instruments(prices)
    )


def main() -> None:
    bt = Backtest(PRICES)
    for strategy in (peeks_ahead, uses_whole_history, honest):
        try:
            result = bt.run(strategy)
            print(
                f"{strategy.__name__:<20} passed, adjusted Sharpe {result.adjusted_sharpe:.2f}"
            )
        except LeakageError as error:
            print(f"{strategy.__name__:<20} caught: {error}")


if __name__ == "__main__":
    main()
