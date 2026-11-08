"""Combining strategies into one set of positions, as the final mixer will.

Run it with: uv run python src/examples/blend_example.py
"""

from pathlib import Path

import polars as pl

from backtest import Backtest
from backtest.pipeline import instruments, risk_scale, to_positions
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "example_prices.csv"


def slow_trend(prices: pl.DataFrame) -> pl.DataFrame:
    return trend.forecast(prices, spans=(64,))


def blend(prices: pl.DataFrame, weights: dict) -> pl.DataFrame:
    """Each strategy through the pipeline, a weighted sum, then risk rescaled once."""
    positions = {
        name: to_positions(f(prices), prices) for name, f in STRATEGIES.items()
    }
    first = next(iter(positions.values()))
    mixed = first.with_columns(
        sum(positions[name][c] * w for name, w in weights.items()).alias(c)
        for c in instruments(prices)
    )
    return risk_scale(mixed, prices)


STRATEGIES = {"trend": trend.forecast, "slow trend": slow_trend}


def main() -> None:
    bt = Backtest(PRICES)
    for name, strategy in STRATEGIES.items():
        print(f"{name:<11} {bt.run(strategy).adjusted_sharpe:.2f}")

    # Slow trend loses money on its own, so giving it weight drags the blend down: the weights
    # matter. Pass a function, not a finished frame, so the blend gets the leakage test too.
    mixed = bt.run_positions(
        lambda prices: blend(prices, {"trend": 0.7, "slow trend": 0.3}), name="blend"
    )
    print(f"{'blend':<11} {mixed.adjusted_sharpe:.2f}")

    # Save the submission format: positions for every date.
    (HERE / "output").mkdir(exist_ok=True)
    mixed.to_csv(HERE / "output" / "blend_positions.csv")


if __name__ == "__main__":
    main()
