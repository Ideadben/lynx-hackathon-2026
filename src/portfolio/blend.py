"""Combines the team's strategies into the one set of positions we submit.

Each strategy goes through the shared risk pipeline on its own, so each has about the same risk.
The weighted sum is then risk-scaled once more, because strategies that don't move together
partly cancel out, and the blend would otherwise take less risk than intended.
"""

from collections.abc import Mapping
from typing import cast

import numpy as np
import polars as pl

from backtest.choose import Choose, State, Weights
from backtest.evaluation import DATE
from backtest.pipeline import (
    FORECAST_CAP,
    RISK_WINDOW,
    instruments,
    risk_scale,
    to_positions,
)
from backtest.runner import MIN_TRAIN_DAYS, Strategy
from backtest.scoring import Scoring

REFIT_EVERY = 63  # About a quarter. Weights estimated more often mostly chase noise.


class Portfolio:
    """A weighted blend of strategies, usable wherever a positions function is.

    With ``weights``, the weights stay fixed. With ``choose``, they are picked again every
    ``refit_every`` days from each strategy's net returns up to that day, so every position
    only uses the past. Until there are ``min_train`` days of returns, the weights are equal.

    Pass it to ``Backtest.run_positions``, not ``Backtest.run``: it returns final positions.

    Args:
        strategies: Forecast functions by name, as written by the researchers. Each is called as
            ``forecast(prices)``, or ``forecast(prices, extras)`` when the backtest has extras.
        weights: Fixed weights by strategy name. Missing strategies get 0.
        choose: A choosing function, from ``backtest.choose`` or ``portfolio.weights``.
            Give ``weights`` or ``choose``, not both. With neither, the weights are equal.
        refit_every: Days between choosing new weights.
        min_train: Days of returns, where every strategy has one, needed before choosing.
        lookback: Choose from only the last this many days. ``None`` uses all earlier days.
        scoring: How to calculate the strategies' net returns for ``choose``.
        vol_window: As in ``Backtest``. Keep these four the same as the backtest's.
        risk_window: As in ``Backtest``.
        cap: As in ``Backtest``.
        normalize: As in ``Backtest``.

    Raises:
        ValueError: If the arguments are invalid.

    Example:
        >>> portfolio = Portfolio({"trend": trend.forecast, "carry": carry}, choose=risk_parity())
        >>> result = bt.run_positions(portfolio, name="portfolio")
        >>> portfolio.weights(bt.prices)
    """

    def __init__(
        self,
        strategies: Mapping[str, Strategy],
        weights: Weights | None = None,
        *,
        choose: Choose | None = None,
        refit_every: int = REFIT_EVERY,
        min_train: int = MIN_TRAIN_DAYS,
        lookback: int | None = None,
        scoring: Scoring | None = None,
        vol_window: int = RISK_WINDOW,
        risk_window: int = RISK_WINDOW,
        cap: float | None = FORECAST_CAP,
        normalize: bool = True,
    ) -> None:
        if not strategies:
            raise ValueError("Give at least one strategy.")
        if weights is not None and choose is not None:
            raise ValueError("Give fixed weights or a choosing function, not both.")
        if refit_every < 1 or min_train < 2:
            raise ValueError(
                "refit_every must be at least 1, and min_train at least 2."
            )
        if lookback is not None and lookback < min_train:
            raise ValueError("lookback must be at least min_train.")
        self.strategies = dict(strategies)
        self.names = list(self.strategies)
        self.fixed = None if weights is None else _checked(weights, self.names)
        self.choose = choose
        self.refit_every = refit_every
        self.min_train = min_train
        self.lookback = lookback
        self.scoring = scoring or Scoring()
        self.vol_window = vol_window
        self.risk_window = risk_window
        self.cap = cap
        self.normalize = normalize
        self.__name__ = "portfolio"

    def __call__(self, prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
        """The blended, risk-scaled positions.

        Args:
            prices: ``AsOfDate`` plus one price column per asset.
            *extras: The backtest's extras, if it has any.

        Returns:
            Final positions, one row per price date.
        """
        positions = self.positions(prices, *extras)
        rows = self._weight_rows(prices, positions)
        return risk_scale(_weighted_sum(positions, rows), prices, self.risk_window)

    def weights(self, prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
        """The weights in use, from each day they were chosen.

        Args:
            prices: ``AsOfDate`` plus one price column per asset.
            *extras: The backtest's extras, if it has any.

        Returns:
            ``AsOfDate`` plus one column per strategy, one row per day the weights changed.
            The first row is the start of the data.
        """
        rows = self._weight_rows(prices, self.positions(prices, *extras))
        frame = pl.DataFrame(rows, schema=self.names, orient="row")
        frame = frame.with_columns(prices[DATE]).select(DATE, *self.names)
        changed = pl.any_horizontal(pl.col(n) != pl.col(n).shift(1) for n in self.names)
        return frame.filter(changed.fill_null(True))

    def positions(
        self, prices: pl.DataFrame, *extras: pl.DataFrame
    ) -> dict[str, pl.DataFrame]:
        """Each strategy's positions through the risk pipeline, before blending.

        Args:
            prices: ``AsOfDate`` plus one price column per asset.
            *extras: The backtest's extras, if it has any.

        Returns:
            Positions by strategy name.
        """
        return {
            name: to_positions(
                forecast(prices, *extras),
                prices,
                self.vol_window,
                self.risk_window,
                self.cap,
                self.normalize,
            )
            for name, forecast in self.strategies.items()
        }

    def _weight_rows(
        self, prices: pl.DataFrame, positions: dict[str, pl.DataFrame]
    ) -> np.ndarray:
        """The weights for every row, one column per strategy."""
        count = len(self.names)
        if self.fixed is not None:
            fixed = [self.fixed[name] for name in self.names]
            return np.tile(fixed, (prices.height, 1))
        if self.choose is None:
            return np.full((prices.height, count), 1 / count)

        returns = self._returns(prices, positions)
        rows = np.full((prices.height, count), 1 / count)
        current: Weights | None = None
        state: State = None
        # Refit days are counted from the first row, so cutting off later data can't move them.
        for start in range(self.refit_every, prices.height, self.refit_every):
            # Returns up to and including this day are known when its positions are decided.
            train = returns.head(start + 1).drop_nulls()
            if self.lookback is not None:
                train = train.tail(self.lookback)
            if train.height >= self.min_train:
                weights, state = self.choose(train, state)
                current = _checked(weights, self.names)
            if current is not None:
                rows[start:] = [current[name] for name in self.names]
        return rows

    def _returns(
        self, prices: pl.DataFrame, positions: dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """Each strategy's daily net returns, one column per strategy, one row per price date."""
        columns = {}
        for name, frame in positions.items():
            figures = self.scoring.figures(frame, prices, ["returns"])
            columns[name] = cast(pl.DataFrame, figures["returns"])["returns"]
        return pl.DataFrame(columns)


def _checked(weights: Weights, names: list[str]) -> Weights:
    unknown = set(weights) - set(names)
    if unknown:
        raise ValueError(f"Weights for unknown strategies: {sorted(unknown)}")
    full = {name: float(weights.get(name, 0.0)) for name in names}
    if not all(np.isfinite(list(full.values()))):
        raise ValueError(f"Weights must be finite numbers: {full}")
    return full


def _weighted_sum(positions: dict[str, pl.DataFrame], rows: np.ndarray) -> pl.DataFrame:
    """Sum of weight times positions, row by row. Null only where every weighted strategy is."""
    first = next(iter(positions.values()))
    columns = instruments(first)
    stacked = np.stack(
        [frame.select(columns).to_numpy().astype(float) for frame in positions.values()]
    )  # strategies, days, assets
    weighted = stacked * rows.T[:, :, None]
    used = (rows.T != 0)[:, :, None] & ~np.isnan(stacked)
    total = np.where(
        used.any(axis=0), np.nansum(np.where(used, weighted, 0), axis=0), np.nan
    )
    return (
        pl.DataFrame(total, schema=columns, orient="row")
        .with_columns(first[DATE])
        .select(DATE, *columns)
    )
