"""How trading costs change the picture, under the PDF's and Lynx's cost models.

Run it with: uv run python src/examples/costs_example.py
"""

from pathlib import Path

import polars as pl

from backtest import Backtest
from backtest.pipeline import instruments
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "prices_old_example" / "example_prices.csv"


def reversal(prices: pl.DataFrame) -> pl.DataFrame:
    """A fast strategy: bet against the move of the last 5 days."""
    return prices.with_columns(-pl.col(instruments(prices)).diff(5))


def main() -> None:
    pdf = Backtest(PRICES)
    lynx = Backtest(PRICES, scoring="lynx")

    # A slow strategy barely notices costs. A fast one can flip from profit to loss.
    for name, strategy in (("trend", trend.forecast), ("reversal", reversal)):
        before = pdf.run(strategy).sharpe_ex_slippage
        after_pdf = pdf.run(strategy).sharpe
        after_lynx = lynx.run(strategy).sharpe
        holding = pdf.run(strategy).av_holding
        print(
            f"{name:<9} holds {holding:4.0f} days. Sharpe before costs {before:5.2f}, "
            f"PDF costs {after_pdf:5.2f}, Lynx costs {after_lynx:5.2f}"
        )

    # How high can the flat PDF cost go before the fast strategy stops making money?
    print("\nReversal Sharpe by flat cost per unit traded:")
    for delta in (0.0, 0.0002, 0.001, 0.005, 0.02):
        result = Backtest(PRICES, delta=delta).run(reversal)
        print(f"  delta {delta:<7} {result.sharpe:5.2f}")


if __name__ == "__main__":
    main()
