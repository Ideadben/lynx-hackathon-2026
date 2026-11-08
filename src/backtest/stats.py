"""Statistics on daily returns: Sharpe ratios, and bootstrap intervals for them.

``sharpe`` and ``adjusted_sharpe`` follow equations (4) and (5) of the PDF. They take one series of
returns, or a 2D array with one column per series, and then score every column at once.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from arch.bootstrap import StationaryBootstrap, optimal_block_length

DAYS_PER_YEAR = 252

Returns = np.ndarray | pl.Series | pl.DataFrame
"""Daily returns: an array, a polars series, or a frame with a ``returns`` column."""
Statistic = Callable[[np.ndarray], float]
"""Turns an array of daily returns into a number, like ``adjusted_sharpe``."""


def sharpe(returns: Returns, days_per_year: int = DAYS_PER_YEAR) -> float | np.ndarray:
    """Equation (4): the yearly Sharpe ratio of daily returns.

    Args:
        returns: Daily returns: an array, a polars series, or a frame with a ``returns`` column.
            A 2D array is scored column by column. NaN days are dropped from 1D input.
        days_per_year: Trading days per year. 252 for the PDF, 260 for ``evaluation.py``.

    Returns:
        The Sharpe ratio, or one per column for a 2D array. NaN with fewer than 2 returns or no
        variation.

    Raises:
        ValueError: If a 2D array contains NaN.
    """
    values = _values(returns)
    count = values.shape[0]
    if count < 2:
        return _nan_like(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (
            values.mean(axis=0) / values.std(axis=0, ddof=1) * np.sqrt(days_per_year)
        )
    return _finite_or_nan(ratio)


def adjusted_sharpe(
    returns: Returns, days_per_year: int = DAYS_PER_YEAR
) -> float | np.ndarray:
    """Equation (5): Sharpe ratio penalised for negative skew and excess kurtosis.

    Skewness and excess kurtosis are bias-corrected, as in ``evaluation.py``. This is the
    competition score.

    Args:
        returns: Daily returns: an array, a polars series, or a frame with a ``returns`` column.
            A 2D array is scored column by column. NaN days are dropped from 1D input.
        days_per_year: Trading days per year. 252 for the PDF, 260 for ``evaluation.py``.

    Returns:
        The Adjusted Sharpe ratio, or one per column for a 2D array. NaN with fewer than 4
        returns or no variation.

    Raises:
        ValueError: If a 2D array contains NaN.
    """
    values = _values(returns)
    count = values.shape[0]
    if count < 4:
        return _nan_like(values)
    ratio = sharpe(values, days_per_year)
    centred = values - values.mean(axis=0)
    m2 = (centred**2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = (centred**3).mean(axis=0) / m2**1.5
        g2 = (centred**4).mean(axis=0) / m2**2 - 3
    skewness = g1 * np.sqrt(count * (count - 1)) / (count - 2)
    excess_kurtosis = ((count + 1) * g2 + 6) * (count - 1) / ((count - 2) * (count - 3))
    adjusted = ratio * (1 + ratio / 6 * skewness - ratio**2 / 24 * excess_kurtosis)
    return _finite_or_nan(adjusted)


@dataclass(frozen=True)
class Bootstrap:
    """A statistic, its 95% interval, and how often it came out at or below zero."""

    estimate: float
    """The statistic on the original returns."""
    low: float
    """The 2.5th percentile of the resampled statistic."""
    high: float
    """The 97.5th percentile of the resampled statistic."""
    p_at_most_zero: float
    """Share of resampled values at or below 0."""
    block: float
    """Average block length used, in days."""
    draws: np.ndarray
    """Every resampled value of the statistic."""

    def __str__(self) -> str:
        return (
            f"{self.estimate:.2f}, 95% interval [{self.low:.2f}, {self.high:.2f}], "
            f"P(<= 0) = {self.p_at_most_zero:.1%}, average block {self.block:.0f} days"
        )


def bootstrap(
    returns: Returns,
    statistic: Statistic = adjusted_sharpe,
    *,
    block: float | None = None,
    samples: int = 10_000,
    seed: int = 0,
) -> Bootstrap:
    """Resamples daily returns in blocks, to show how uncertain a statistic is.

    Uses a stationary bootstrap: blocks of random length, ``block`` days on average, so runs of
    related days stay together. Run it on out-of-sample returns, like ``walk.result.returns``.

    Args:
        returns: Daily returns: an array, a polars series, or a frame with a ``returns`` column.
        statistic: Turns an array of returns into a number. Defaults to ``adjusted_sharpe``.
        block: Average block length in days. By default it is estimated from how related the
            daily returns are (Politis and White).
        samples: Number of resampled histories.
        seed: Random seed. The same seed gives the same result.

    Returns:
        The statistic, its 95% interval, the share of draws at or below 0, the block length, and
        every draw. ``print`` it for a one-line summary.

    Example:
        >>> print(bootstrap(result.returns))
        0.72, 95% interval [0.24, 1.16], P(<= 0) = 0.1%, average block 1 days
    """
    values = _values(returns)
    block = _block(values) if block is None else block
    draws = (
        StationaryBootstrap(block, values, seed=seed).apply(statistic, samples).ravel()
    )
    return _summary(float(statistic(values)), draws, block)


def paired_bootstrap(
    first: Returns,
    second: Returns,
    statistic: Statistic = adjusted_sharpe,
    *,
    block: float | None = None,
    samples: int = 10_000,
    seed: int = 0,
) -> Bootstrap:
    """Bootstraps ``statistic(first) - statistic(second)`` on the same resampled days.

    Use it to test whether one strategy beats another. Overlapping separate intervals say little,
    because the strategies share the same days.

    Args:
        first: Daily returns of the first strategy.
        second: Daily returns of the second strategy. Frames with an ``AsOfDate`` column are
            matched on date, keeping the days both have. Arrays must have the same length.
        statistic: Turns an array of returns into a number. Defaults to ``adjusted_sharpe``.
        block: Average block length in days. By default it is estimated from the difference.
        samples: Number of resampled histories.
        seed: Random seed. The same seed gives the same result.

    Returns:
        The difference, its 95% interval, and ``p_at_most_zero``: how often ``first`` didn't beat
        ``second``.

    Raises:
        ValueError: If arrays differ in length.
    """
    a, b = _paired_values(first, second)
    block = _block(a - b) if block is None else block

    def difference(a: np.ndarray, b: np.ndarray) -> float:
        return float(statistic(a)) - float(statistic(b))

    draws = (
        StationaryBootstrap(block, a, b, seed=seed).apply(difference, samples).ravel()
    )
    return _summary(difference(a, b), draws, block)


def _summary(estimate: float, draws: np.ndarray, block: float) -> Bootstrap:
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5])
    p = float(np.mean(finite <= 0))
    return Bootstrap(estimate, float(low), float(high), p, float(block), draws)


def _block(values: np.ndarray) -> float:
    """The estimated average block length, or 1 day if it can't be estimated."""
    if np.std(values) == 0:
        return 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        length = float(optimal_block_length(values)["stationary"].iloc[0])
    return max(1.0, length) if np.isfinite(length) else 1.0


def _values(returns: Returns) -> np.ndarray:
    """A float array without missing days. A frame uses its ``returns`` column."""
    if isinstance(returns, pl.DataFrame):
        returns = returns["returns"]
    values = np.asarray(
        returns.to_numpy() if isinstance(returns, pl.Series) else returns, dtype=float
    )
    if values.ndim == 1:
        return values[~np.isnan(values)]
    if np.isnan(values).any():
        raise ValueError("A 2D array of returns must not contain NaN.")
    return values


def _paired_values(first: Returns, second: Returns) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(first, pl.DataFrame) and isinstance(second, pl.DataFrame):
        joined = first.join(second, on="AsOfDate", suffix="_second").drop_nulls()
        return _values(joined["returns"]), _values(joined["returns_second"])
    a, b = _values(first), _values(second)
    if a.shape != b.shape:
        raise ValueError(
            "Paired returns need the same length, or frames to match on date."
        )
    return a, b


def _nan_like(values: np.ndarray) -> float | np.ndarray:
    return float("nan") if values.ndim == 1 else np.full(values.shape[1], np.nan)


def _finite_or_nan(values: np.ndarray) -> float | np.ndarray:
    cleaned = np.where(np.isfinite(values), values, np.nan)
    return float(cleaned) if cleaned.ndim == 0 else cleaned
