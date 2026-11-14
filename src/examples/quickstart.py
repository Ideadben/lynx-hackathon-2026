"""The shortest useful script: write a forecast, backtest it, look at the result.

Run it with: uv run python src/examples/quickstart.py
"""

from pathlib import Path

import polars as pl

from backtest import Backtest

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "prices_old_example" / "example_prices.csv"


def moving_average(prices: pl.DataFrame) -> pl.DataFrame:
    """Long when the last ~16 days are above the last ~64 days, short when below."""
    assets = [c for c in prices.columns if c != "AsOfDate"]
    fast = pl.col(assets).ewm_mean(span=16, adjust=False, min_samples=16)
    slow = pl.col(assets).ewm_mean(span=64, adjust=False, min_samples=64)
    return prices.with_columns((fast - slow).sign())


def main() -> None:
    bt = Backtest(PRICES)
    result = bt.run(moving_average)

    print(result)
    print(f"\nThe competition score: {result.adjusted_sharpe:.2f}")

    (HERE / "output").mkdir(exist_ok=True)
    result.plot().savefig(HERE / "output" / "quickstart_dashboard.png")


if __name__ == "__main__":
    main()
