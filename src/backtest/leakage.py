"""Checks that a strategy never uses future data, as required by the instructions."""

from collections.abc import Callable

import numpy as np
import polars as pl

from backtest.evaluation import DATE
from backtest.pipeline import instruments

CUT_FRACTIONS = (0.5, 0.75)
LAST_DAYS_REMOVED = 20


class LeakageError(AssertionError):
    """The strategy's positions depend on data from after their date."""


def check_no_leakage(
    positions_fn: Callable[..., pl.DataFrame],
    prices: pl.DataFrame,
    extras: pl.DataFrame | None = None,
) -> None:
    """Raises ``LeakageError`` if positions up to a date change when later data changes.

    Runs both tests from the instructions at several dates: removing the data after the date, and
    altering it.

    Args:
        positions_fn: Called as ``positions_fn(prices)``, or ``positions_fn(prices, extras)`` if
            extras are given. Must return positions with one row per price date.
        prices: ``AsOfDate`` plus one price column per instrument.
        extras: Optional frame of non-price series with an ``AsOfDate`` column.
    """

    def run(prices: pl.DataFrame, extras: pl.DataFrame | None) -> pl.DataFrame:
        return positions_fn(prices) if extras is None else positions_fn(prices, extras)

    full = run(prices, extras)
    cuts = [int(prices.height * fraction) for fraction in CUT_FRACTIONS] + [
        prices.height - LAST_DAYS_REMOVED
    ]

    for cut in cuts:
        date = prices[DATE][cut - 1]
        expected = full.head(cut)

        shorter = run(
            _up_to(prices, date), None if extras is None else _up_to(extras, date)
        )
        _compare(shorter, expected, f"removing the data after {date}")

        altered = run(
            _alter_after(prices, date),
            None if extras is None else _alter_after(extras, date),
        )
        _compare(altered.head(cut), expected, f"altering the data after {date}")


def _up_to(frame: pl.DataFrame, date: object) -> pl.DataFrame:
    return frame.filter(pl.col(DATE) <= date)


def _alter_after(frame: pl.DataFrame, date: object) -> pl.DataFrame:
    is_future = pl.col(DATE) > date
    return frame.with_columns(
        pl.when(is_future).then(pl.col(c) * 1.5 + 100).otherwise(pl.col(c)).alias(c)
        for c in instruments(frame)
    )


def _compare(actual: pl.DataFrame, expected: pl.DataFrame, change: str) -> None:
    if actual.height != expected.height:
        raise LeakageError(
            f"Expected {expected.height} rows after {change}, got {actual.height}."
        )
    columns = instruments(expected)
    actual_values = actual.select(columns).to_numpy().astype(float)
    expected_values = expected.select(columns).to_numpy().astype(float)
    if not np.allclose(
        actual_values, expected_values, rtol=1e-9, atol=1e-12, equal_nan=True
    ):
        raise LeakageError(f"Earlier positions changed after {change}.")
