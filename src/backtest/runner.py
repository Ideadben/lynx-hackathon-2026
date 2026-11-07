"""The main entry point: data in once, then forecast in, key figures and plots out."""

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Literal, cast

import polars as pl
from matplotlib.figure import Figure

from backtest.evaluation import DATE, POSITION_LAG
from backtest.leakage import check_no_leakage
from backtest.pipeline import (
    FORECAST_CAP,
    RISK_WINDOW,
    Data,
    check_frame,
    instruments,
    load_data,
    to_positions,
)
from backtest.scoring import LYNX_COSTS, PDF_DELTA, Scoring, ScoringKind

DEV_FRACTION = 0.75  # Same split as Benjamin's LSTM work.
KEY_FIGURES = [
    "adjusted_sharpe",
    "sharpe",
    "sharpe_ex_slippage",
    "sharpe_1d_lag",
    "skewness",
    "kurtosis",
    "av_holding",
    "position_bias",
]

Period = Literal["dev", "holdout", "full"]
Strategy = Callable[..., pl.DataFrame]


@dataclass(frozen=True)
class Result:
    """The outcome of ``Backtest.run``.

    Every key figure is an attribute, for example ``result.sharpe``, and they are all in the
    ``figures`` dict. They cover ``period``, while ``positions`` covers all dates.
    """

    name: str
    period: Period
    figures: dict[str, float]
    positions: pl.DataFrame
    prices: pl.DataFrame
    scoring: Scoring

    def __getattr__(self, name: str) -> float:
        if name in KEY_FIGURES:
            return self.figures[name]
        raise AttributeError(
            f"Result has no attribute {name!r}. Key figures: {KEY_FIGURES}"
        )

    @property
    def returns(self) -> pl.DataFrame:
        """Daily net returns for this period: ``AsOfDate`` and ``returns``."""
        figures = self._figures("returns")
        return cast(pl.DataFrame, figures["returns"]).drop_nulls()

    def by_year(self) -> pl.DataFrame:
        """Sharpe after costs, total return and number of scored days, per calendar year."""
        returns = pl.col("returns")
        return self.returns.group_by(
            pl.col(DATE).dt.year().alias("year"), maintain_order=True
        ).agg(
            days=pl.len(),
            total_return=returns.sum(),
            sharpe=returns.mean() / returns.std() * self.scoring.annualization,
        )

    def by_instrument(self) -> pl.DataFrame:
        """Sharpe before costs, holding period and position bias, per instrument."""
        figures = self._figures(
            "sharpe_per_instrument",
            "av_holding_per_instrument",
            "position_bias_per_instrument",
        )
        sharpe = cast(dict[str, float], figures["sharpe_per_instrument"])
        holding = cast(dict[str, float], figures["av_holding_per_instrument"])
        bias = cast(dict[str, float], figures["position_bias_per_instrument"])
        return pl.DataFrame(
            {
                "instrument": list(sharpe),
                "sharpe_ex_slippage": list(sharpe.values()),
                "av_holding": list(holding.values()),
                "position_bias": list(bias.values()),
            }
        )

    def to_csv(self, path: str | Path) -> None:
        """Saves positions for all dates, whatever the period. Warm-up nulls become empty cells."""
        self.positions.write_csv(path)

    def plot(self) -> Figure:
        """The four-panel dashboard for this period."""
        return self.scoring.plot(self._scored(), self.prices)

    def __str__(self) -> str:
        lines = [f"{self.name} ({self.period})"]
        lines += [f"  {name:<20}{value:>8.3f}" for name, value in self.figures.items()]
        return "\n".join(lines)

    def _scored(self) -> pl.DataFrame:
        return _in_period(self.positions, self.prices, self.period)

    def _figures(self, *names: str) -> dict[str, object]:
        return self.scoring.figures(self._scored(), self.prices, names)


@dataclass(frozen=True)
class Comparison:
    """The outcome of ``Backtest.compare``.

    ``figures`` has one row per strategy. ``correlations`` holds the correlations of their daily
    net returns, and its ``strategy`` column names the rows.
    """

    results: dict[str, Result]
    figures: pl.DataFrame
    correlations: pl.DataFrame

    def __str__(self) -> str:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True):
            return f"{self.figures}\n\nCorrelation of returns\n{self.correlations}"


class Backtest:
    """Holds one data source and its settings, and tests any number of strategies on it.

    Args:
        data: Prices as a csv path, or a polars or pandas frame. Needs a date column, and every
            other column is an instrument.
        extras: Optional non-price series, in the same forms as ``data``.
        scoring: ``"pdf"`` for the formulas in the instructions, or ``"lynx"`` for ``evaluation.py``.
        delta: Cost per unit traded for ``"pdf"`` scoring.
        costs: Cost per unit traded for ``"lynx"`` scoring, in daily standard deviations.
        vol_window: Window for asset volatility, in days.
        risk_window: Window for strategy risk, in days.
        cap: Largest absolute forecast after scaling to an average size of 10. ``None`` for no cap.
        normalize: Scales each asset to an average absolute forecast of 10. If False, forecasts are
            used as they are and ``cap`` has no effect, so sizes must be comparable across assets.
    """

    def __init__(
        self,
        data: Data,
        extras: Data | None = None,
        *,
        scoring: ScoringKind = "pdf",
        delta: float = PDF_DELTA,
        costs: float = LYNX_COSTS,
        vol_window: int = RISK_WINDOW,
        risk_window: int = RISK_WINDOW,
        cap: float | None = FORECAST_CAP,
        normalize: bool = True,
    ) -> None:
        self.prices = load_data(data)
        self.extras = None if extras is None else load_data(extras)
        self.scoring = Scoring(scoring, delta, costs)
        self.vol_window = vol_window
        self.risk_window = risk_window
        self.cap = cap
        self.normalize = normalize

    def run(
        self,
        forecast: Strategy,
        *,
        period: Period = "dev",
        check_leakage: bool = True,
        name: str | None = None,
    ) -> Result:
        """Backtests a forecast function through the shared risk pipeline.

        Args:
            forecast: Called as ``forecast(prices)``, or ``forecast(prices, extras)`` if this
                backtest has extras. Returns ``AsOfDate`` plus one column per instrument, one row
                per price date. Row T may only use data up to and including T. Use a lambda or
                ``functools.partial`` to set parameters.
            period: ``"dev"`` for the first 75% of dates, ``"holdout"`` for the rest, or ``"full"``.
                Use the holdout sparingly, because every look at it makes it less trustworthy.
            check_leakage: Reruns the strategy on changed data, and raises ``LeakageError`` on
                failure. Only turn this off for slow strategies during a parameter search.
            name: The name shown in results. Defaults to the function's name.

        Returns:
            The key figures and positions: ``result.sharpe``, ``result.figures``, ``result.plot()``.
        """

        def positions_fn(prices: pl.DataFrame, *extra: pl.DataFrame) -> pl.DataFrame:
            return to_positions(
                forecast(prices, *extra),
                prices,
                self.vol_window,
                self.risk_window,
                self.cap,
                self.normalize,
            )

        return self.run_positions(
            positions_fn,
            period=period,
            check_leakage=check_leakage,
            name=name or _name(forecast),
        )

    def run_positions(
        self,
        positions: pl.DataFrame | Strategy,
        *,
        period: Period = "dev",
        check_leakage: bool = True,
        name: str | None = None,
    ) -> Result:
        """Scores positions as they are, without the risk pipeline.

        Args:
            positions: A function called like a forecast, which returns final positions, or a
                finished frame of positions. A frame can't be rerun on changed data, so it skips
                the leakage test. Pass a function for anything we submit.
            period: As in ``run``.
            check_leakage: As in ``run``. Only applies to a function.
            name: The name shown in results.

        Returns:
            The key figures and positions, as from ``run``.
        """
        if callable(positions):
            if check_leakage:
                check_no_leakage(positions, self.prices, self.extras)
            extra = () if self.extras is None else (self.extras,)
            frame = positions(self.prices, *extra)
            name = name or _name(positions)
        else:
            frame = positions
            name = name or "positions"

        check_frame(frame, self.prices)
        frame = frame.select(DATE, *instruments(self.prices))
        scored = _in_period(frame, self.prices, period)
        figures = self.scoring.figures(scored, self.prices, KEY_FIGURES)
        values = {key: float(cast(float, figures[key])) for key in KEY_FIGURES}
        return Result(name, period, values, frame, self.prices, self.scoring)

    def compare(
        self,
        strategies: Iterable[Strategy] | Mapping[str, Strategy],
        *,
        period: Period = "dev",
        check_leakage: bool = True,
    ) -> Comparison:
        """Runs several forecast functions and puts them side by side.

        Args:
            strategies: A list of functions, or a dict of ``{name: function}``. Use a dict if
                functions share a name, like two ``partial`` versions of one function.
            period: As in ``run``.
            check_leakage: As in ``run``.

        Returns:
            ``figures`` with one row per strategy, ``correlations`` of their daily net returns, and
            each ``Result`` in ``results``.
        """
        if isinstance(strategies, Mapping):
            named = dict(strategies)
        else:
            functions = list(strategies)
            named = {_name(function): function for function in functions}
            if len(named) != len(functions):
                raise ValueError("Strategies share a name. Pass {name: function}.")
        if not named:
            raise ValueError("There are no strategies to compare.")

        results = {
            name: self.run(
                function, period=period, check_leakage=check_leakage, name=name
            )
            for name, function in named.items()
        }
        figures = pl.DataFrame(
            [{"strategy": name, **r.figures} for name, r in results.items()]
        )
        return Comparison(results, figures, _correlations(results))

    def sweep(
        self,
        forecast: Strategy,
        /,
        *,
        period: Period = "dev",
        check_leakage: bool = True,
        **grid: Iterable[object],
    ) -> pl.DataFrame:
        """Runs a forecast function for every combination of parameters.

        Args:
            forecast: A forecast function that takes the parameters as keyword arguments.
            period: As in ``run``.
            check_leakage: As in ``run``.
            **grid: The values to try per parameter, like ``span=[8, 16, 32]``.

        Returns:
            One row per combination: the parameters, then the key figures.
        """
        if not grid:
            raise ValueError("Give at least one parameter, like span=[8, 16, 32].")
        rows = []
        for values in itertools.product(*grid.values()):
            parameters = dict(zip(grid, values, strict=True))
            result = self.run(
                partial(forecast, **parameters),
                period=period,
                check_leakage=check_leakage,
            )
            rows.append(parameters | result.figures)
        return pl.DataFrame(rows)


def dev_end(prices: pl.DataFrame) -> date:
    """The date of the last development position. Its return is earned two days later."""
    return prices[DATE][int(prices.height * DEV_FRACTION) - 1 - POSITION_LAG]


def _name(strategy: Strategy) -> str:
    return getattr(strategy, "__name__", type(strategy).__name__)


def _in_period(
    positions: pl.DataFrame, prices: pl.DataFrame, period: Period
) -> pl.DataFrame:
    """Selects positions, not prices, so volatility estimates still see the earlier data."""
    if period == "full":
        return positions
    if period == "dev":
        return positions.filter(pl.col(DATE) <= dev_end(prices))
    if period == "holdout":
        return positions.filter(pl.col(DATE) > dev_end(prices))
    raise ValueError(f"Unknown period {period!r}; use 'dev', 'holdout' or 'full'.")


def _correlations(results: dict[str, Result]) -> pl.DataFrame:
    """Correlations of daily net returns, on the dates every strategy has a return."""
    joined = None
    for name, result in results.items():
        returns = result.returns.rename({"returns": name})
        joined = returns if joined is None else joined.join(returns, on=DATE)
    assert joined is not None
    return (
        joined.drop(DATE).corr().insert_column(0, pl.Series("strategy", list(results)))
    )
