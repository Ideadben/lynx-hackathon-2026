"""Scores positions, either by the formulas in the PDF or with Lynx's ``evaluation.py``.

The two differ in costs and annualisation:

* ``"pdf"``: a flat cost of ``delta`` per unit traded (equation 1) and 252 days a year (equation 4).
* ``"lynx"``: a cost of ``costs`` times the asset's volatility per unit traded, and 260 days a year.

Both use excess kurtosis in the Adjusted Sharpe, as the PDF's reference (Maillard, 2018) does.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal, cast

import numpy as np
import polars as pl
from matplotlib import gridspec
from matplotlib.figure import Figure

from backtest.evaluation import (
    ANNUALIZATION,
    DATE,
    POSITION_LAG,
    _excess_kurtosis,
    _plot_cumulative,
    _plot_instrument_bars,
    _plot_key_figure_table,
    _plot_rolling,
    _skew,
    calc_key_figures,
    plot_key_figures,
)
from backtest.pipeline import instruments

PDF_DELTA = 0.0002
PDF_DAYS_PER_YEAR = 252
LYNX_COSTS = -0.02

ScoringKind = Literal["pdf", "lynx"]


@dataclass(frozen=True)
class Scoring:
    """Which scoring to use, and its cost setting."""

    kind: ScoringKind = "pdf"
    delta: float = PDF_DELTA
    costs: float = LYNX_COSTS

    def __post_init__(self) -> None:
        if self.kind not in ("pdf", "lynx"):
            raise ValueError(f"Unknown scoring {self.kind!r}; use 'pdf' or 'lynx'.")

    @property
    def annualization(self) -> float:
        return (
            float(np.sqrt(PDF_DAYS_PER_YEAR)) if self.kind == "pdf" else ANNUALIZATION
        )

    def figures(
        self, positions: pl.DataFrame, prices: pl.DataFrame, names: Sequence[str]
    ) -> dict[str, object]:
        """The key figures in ``names``, in the same format as ``calc_key_figures``."""
        if self.kind == "lynx":
            return calc_key_figures(
                positions, prices, costs=self.costs, key_figures=names
            )
        figures = _pdf_figures(positions, prices, self.delta)
        return {name: figures[name] for name in names}

    def plot(self, positions: pl.DataFrame, prices: pl.DataFrame) -> Figure:
        """The four-panel dashboard."""
        if self.kind == "lynx":
            return plot_key_figures(positions, prices, costs=self.costs)

        figures = _pdf_figures(positions, prices, self.delta)
        xlims = (
            cast(date, positions[DATE].min()),
            cast(date, positions[DATE].max()),
        )
        fig = Figure(figsize=(16, 10))
        grid = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25)
        _plot_cumulative(fig.add_subplot(grid[0, 0]), figures, xlims)
        _plot_key_figure_table(fig.add_subplot(grid[0, 1]), figures)
        _plot_rolling(fig, grid[1, 0], figures, xlims, PDF_DAYS_PER_YEAR)
        _plot_instrument_bars(fig, grid[1, 1], figures)
        return fig


def _pdf_figures(
    positions: pl.DataFrame, prices: pl.DataFrame, delta: float
) -> dict[str, object]:
    """Equation (1) for returns, then the figures from equations (4), (5) and (6)."""
    columns = instruments(positions)
    start, end = positions[DATE][0], positions[DATE][-1]
    price_changes = prices.select(DATE, pl.col(columns).diff()).filter(
        pl.col(DATE).is_between(start, end)
    )
    dates = price_changes[DATE]
    changes = price_changes.select(columns).to_numpy().astype(float)
    held = (
        positions.select(pl.col(columns).shift(POSITION_LAG)).to_numpy().astype(float)
    )

    trades = np.abs(_diff(held))
    # The first day of a position has no earlier position to trade from, so it has no net return.
    # Gross returns leave it out too, so that net and gross cover the same days.
    gross = np.where(np.isnan(trades), np.nan, held * changes)
    net = gross - delta * trades
    held_later = _shift(held)
    net_later = held_later * changes - delta * np.abs(_diff(held_later))

    total = _row_sum(net)
    series = pl.Series("returns", total, nan_to_null=True)
    window = PDF_DAYS_PER_YEAR
    rolling_std = series.rolling_std(window, min_samples=window // 2)
    rolling_mean = series.rolling_mean(window, min_samples=window // 2)
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(PDF_DAYS_PER_YEAR)
    slippages = pl.Series(-_row_sum(delta * trades), nan_to_null=True)
    sharpe = _sharpe(total)
    skewness = _skew(series)
    excess_kurtosis = _excess_kurtosis(series)

    no_costs = calc_key_figures(
        positions,
        prices,
        costs=0.0,
        key_figures=["position_bias", "rolling_position_bias"],
    )
    return no_costs | {
        "adjusted_sharpe": sharpe
        * (1 + sharpe / 6 * skewness - sharpe**2 / 24 * excess_kurtosis),
        "sharpe": sharpe,
        "sharpe_ex_slippage": _sharpe(_row_sum(gross)),
        "sharpe_1d_lag": _sharpe(_row_sum(net_later)),
        "skewness": skewness,
        "kurtosis": excess_kurtosis + 3,
        "av_holding": _holding(held, trades),
        "returns": pl.DataFrame({DATE: dates, "returns": series}),
        "slippages": pl.DataFrame({DATE: dates, "slippages": slippages}),
        "rolling_std": pl.DataFrame({DATE: dates, "rolling_std": rolling_std}),
        "rolling_sharpe": pl.DataFrame({DATE: dates, "rolling_sharpe": rolling_sharpe}),
        "sharpe_per_instrument": {
            c: _sharpe(gross[:, i]) for i, c in enumerate(columns)
        },
        "av_holding_per_instrument": {
            c: _holding(held[:, i], trades[:, i]) for i, c in enumerate(columns)
        },
        "position_bias_per_instrument": {
            c: _bias(held[:, i]) for i, c in enumerate(columns)
        },
    }


def _shift(values: np.ndarray) -> np.ndarray:
    return np.vstack([np.full((1, values.shape[1]), np.nan), values[:-1]])


def _diff(values: np.ndarray) -> np.ndarray:
    return values - _shift(values)


def _row_sum(values: np.ndarray) -> np.ndarray:
    """Sum over instruments per day; a day without any value is NaN, not zero."""
    return np.where(np.isnan(values).all(axis=1), np.nan, np.nansum(values, axis=1))


def _sharpe(returns: np.ndarray) -> float:
    """Equation (4). NaN when there are too few returns."""
    values = returns[~np.isnan(returns)]
    if values.size < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(values.mean() / values.std(ddof=1) * np.sqrt(PDF_DAYS_PER_YEAR))


def _holding(held: np.ndarray, trades: np.ndarray) -> float:
    """Equation (6). NaN when nothing was traded."""
    traded = np.nansum(trades)
    return float(2 * np.nansum(np.abs(held)) / traded) if traded else float("nan")


def _bias(held: np.ndarray) -> float:
    size = np.nansum(np.abs(held))
    return float(np.nansum(held) / size) if size else float("nan")
