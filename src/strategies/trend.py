"""Trend following: go long what has been rising and short what has been falling.

For each speed, compare a recent average price with a longer one, four times as slow. If the recent
average is higher, prices have been rising, so that speed votes +1, otherwise -1. The forecast is
the average vote, so it runs from -1 when all speeds say short to +1 when all say long.

Several speeds are used because the result depends strongly on any single one. On the development
period the edge is around spans of 8 to 32, and it fades outside that range.
"""

import polars as pl

SPANS = (8, 16, 32)
SLOW_MULTIPLE = 4


def forecast(prices: pl.DataFrame, spans: tuple[int, ...] = SPANS) -> pl.DataFrame:
    """The average vote of the moving average crossovers, per asset and day."""
    assets = [column for column in prices.columns if column != "AsOfDate"]
    # Added up rather than averaged with a null-skipping mean, so a day stays null until every
    # speed has warmed up.
    return prices.with_columns(
        (sum(_vote(asset, span) for span in spans) / len(spans)).alias(asset)
        for asset in assets
    )


def _vote(asset: str, span: int) -> pl.Expr:
    slow_span = SLOW_MULTIPLE * span
    fast = pl.col(asset).ewm_mean(span=span, adjust=False, min_samples=span)
    slow = pl.col(asset).ewm_mean(span=slow_span, adjust=False, min_samples=slow_span)
    return (fast - slow).sign()
