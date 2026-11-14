"""Setting parameters, and checking whether a result depends on one lucky setting.

Run it with: uv run python src/examples/parameters_example.py
"""

from functools import partial
from pathlib import Path

import polars as pl

from backtest import Backtest
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "prices_old_example" / "example_prices.csv"


def crossover(prices: pl.DataFrame, fast: int, slow: int) -> pl.DataFrame:
    """Long when the fast average is above the slow one."""
    assets = [c for c in prices.columns if c != "AsOfDate"]
    quick = pl.col(assets).ewm_mean(span=fast, adjust=False, min_samples=fast)
    slow_average = pl.col(assets).ewm_mean(span=slow, adjust=False, min_samples=slow)
    return prices.with_columns((quick - slow_average).sign())


def main() -> None:
    bt = Backtest(PRICES)

    # One setting: fix the parameters with partial.
    result = bt.run(partial(crossover, fast=16, slow=64), name="16 / 64")
    print(f"16 / 64: {result.adjusted_sharpe:.2f}")

    # Many settings: sweep tries every combination, 3 x 3 = 9 runs.
    table = bt.sweep(crossover, fast=[8, 16, 32], slow=[64, 128, 256])
    print(table.select("fast", "slow", "adjusted_sharpe", "av_holding"))

    # Neighbours scoring alike suggests a real effect. One standout suggests luck.
    print(table.pivot("slow", index="fast", values="adjusted_sharpe"))

    # Strategies in src/strategies take parameters the same way.
    print(f"\ntrend at spans 8, 16, 32: {bt.run(trend.forecast).adjusted_sharpe:.2f}")
    print(
        f"trend at span 16 only:   "
        f"{bt.run(partial(trend.forecast, spans=(16,))).adjusted_sharpe:.2f}"
    )


if __name__ == "__main__":
    main()
