"""Loading data in different forms, and giving a strategy extra series.

Run it with: uv run python src/examples/data_example.py
"""

from pathlib import Path

import polars as pl

from backtest import Backtest
from backtest.pipeline import instruments, load_data

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "example_prices.csv"


def trend_when_calm(prices: pl.DataFrame, extras: pl.DataFrame) -> pl.DataFrame:
    """Trend, at half size on days the extra 'stress' series is high.

    With extras, the forecast gets them as a second argument. They are not lined up with the
    price dates, so join them on the date first.
    """
    assets = instruments(prices)
    fast = pl.col(assets).ewm_mean(span=16, adjust=False, min_samples=16)
    slow = pl.col(assets).ewm_mean(span=64, adjust=False, min_samples=64)
    joined = prices.join(extras, on="AsOfDate", how="left")
    size = pl.when(pl.col("stress") > 1.5).then(0.5).otherwise(1.0)
    return joined.select("AsOfDate", ((fast - slow).sign() * size).name.keep())


def main() -> None:
    # Any of these work: a csv path, a polars frame, or a pandas frame.
    frame = pl.read_csv(PRICES, try_parse_dates=True)
    from_path = Backtest(PRICES)
    from_polars = Backtest(frame)
    from_pandas = Backtest(frame.to_pandas().set_index("dates"))
    print(f"Same data three ways: {from_path.prices.equals(from_pandas.prices)}")
    print(f"Assets: {instruments(from_polars.prices)}")

    # A made-up extra series for the demo: recent volatility of all assets, relative to its
    # long-run level. Real extras, like macro data, would come from their own csv.
    prices = load_data(PRICES)
    moves = prices.select(
        pl.mean_horizontal(pl.col(instruments(prices)).pct_change().abs())
    )
    level = moves.to_series()
    stress = level.rolling_mean(20) / level.rolling_mean(250)
    extras = pl.DataFrame({"date": prices["AsOfDate"], "stress": stress})

    bt = Backtest(PRICES, extras)
    result = bt.run(trend_when_calm)
    print(f"\nTrend, halved when stressed: {result.adjusted_sharpe:.2f}")


if __name__ == "__main__":
    main()
