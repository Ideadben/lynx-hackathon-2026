"""The main entry point: data in once, then forecast in, key figures and plots out."""

import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from functools import partial
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
from matplotlib.figure import Figure

from backtest.choose import Choose, Weights, best_single
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
# Walk-forward blocks are fractions of the data, so they adapt to any length of history.
FIRST_TRAIN_FRACTION = 0.3
WALK_FORWARD_BLOCKS = 12
MIN_TRAIN_DAYS = 126  # A parameter can't be judged on less than about half a year.
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
"""Which dates to score: the first 75% (``"dev"``), the rest (``"holdout"``), or all (``"full"``)."""
Strategy = Callable[..., pl.DataFrame]
"""A forecast function: ``forecast(prices)`` or ``forecast(prices, extras)`` returns direction and size per date and asset."""


@dataclass(frozen=True)
class Result:
    """The outcome of ``Backtest.run``.

    Every key figure is an attribute, for example ``result.sharpe``, and they are all in the
    ``figures`` dict. They cover ``period``, while ``positions`` covers all dates.
    """

    name: str
    """The strategy's name, from its function or the ``name`` argument."""
    period: Period
    """Which dates the figures cover: ``"dev"``, ``"holdout"`` or ``"full"``."""
    figures: dict[str, float]
    """Every key figure by name. Each is also an attribute, like ``result.sharpe``."""
    positions: pl.DataFrame
    """Final positions for all dates, not just ``period``."""
    prices: pl.DataFrame
    """The prices the strategy ran on."""
    scoring: Scoring
    """How the positions were scored."""

    @property
    def adjusted_sharpe(self) -> float:
        """The competition score: yearly Sharpe after costs, penalised for negative skew and fat tails."""
        return self.figures["adjusted_sharpe"]

    @property
    def sharpe(self) -> float:
        """Yearly Sharpe ratio after costs."""
        return self.figures["sharpe"]

    @property
    def sharpe_ex_slippage(self) -> float:
        """Yearly Sharpe ratio before costs."""
        return self.figures["sharpe_ex_slippage"]

    @property
    def sharpe_1d_lag(self) -> float:
        """Sharpe with positions one day later. A big drop means a short-lived signal."""
        return self.figures["sharpe_1d_lag"]

    @property
    def skewness(self) -> float:
        """Skew of daily returns. Negative skew lowers the adjusted Sharpe."""
        return self.figures["skewness"]

    @property
    def kurtosis(self) -> float:
        """Kurtosis of daily returns. Normal returns give 3. Fat tails lower the adjusted Sharpe."""
        return self.figures["kurtosis"]

    @property
    def av_holding(self) -> float:
        """Average number of days a position is held."""
        return self.figures["av_holding"]

    @property
    def position_bias(self) -> float:
        """Long/short tilt, from -1 (short only) to 1 (long only)."""
        return self.figures["position_bias"]

    def __getattr__(self, name: str) -> float:
        if name in KEY_FIGURES:
            return self.figures[name]
        raise AttributeError(
            f"Result has no attribute {name!r}. Key figures: {KEY_FIGURES}"
        )

    @property
    def returns(self) -> pl.DataFrame:
        """Daily net returns for this period: ``AsOfDate`` and ``returns``.

        Warm-up days are left out. Pass it to ``bootstrap`` or ``paired_bootstrap``.
        """
        figures = self._figures("returns")
        return cast(pl.DataFrame, figures["returns"]).drop_nulls()

    def by_year(self) -> pl.DataFrame:
        """Results per calendar year in this period.

        Returns:
            One row per year: ``year``, ``days`` scored, ``total_return`` and ``sharpe`` after costs.
            The first and last years can be partial, so check ``days``.
        """
        returns = pl.col("returns")
        return self.returns.group_by(
            pl.col(DATE).dt.year().alias("year"), maintain_order=True
        ).agg(
            days=pl.len(),
            total_return=returns.sum(),
            sharpe=returns.mean() / returns.std() * self.scoring.annualization,
        )

    def by_instrument(self) -> pl.DataFrame:
        """Results per asset in this period.

        Returns:
            One row per asset: ``instrument``, ``sharpe_ex_slippage`` (before costs, the same as the
            dashboard), ``av_holding`` and ``position_bias``. NaN for an asset with no data here.
        """
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
        """Saves the positions for all dates to a csv, whatever the period was.

        The submission needs the full run. Warm-up rows have empty cells, the same as the Lynx
        Quickstart notebook.

        Args:
            path: Where to write the csv.
        """
        self.positions.write_csv(path)

    def plot(self) -> Figure:
        """The four-panel dashboard for this period.

        Returns:
            A matplotlib ``Figure``. In a notebook it shows when it's the last line of a cell.
            In a script, save it with ``result.plot().savefig(path)``.
        """
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
    """Each strategy's full ``Result``, by name."""
    figures: pl.DataFrame
    """One row per strategy, with every key figure."""
    correlations: pl.DataFrame
    """Correlations of daily net returns, on the dates every strategy shares."""

    def __str__(self) -> str:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True):
            return f"{self.figures}\n\nCorrelation of returns\n{self.correlations}"


@dataclass(frozen=True)
class WalkForward:
    """The outcome of ``Backtest.walk_forward``.

    ``result`` scores the stitched positions, so every traded day used weights chosen from
    earlier data. ``choices`` has the weights per block. ``fixed`` scores each rule on its own over
    the same traded days, so its best row is what hindsight would have picked.
    """

    result: Result
    """The stitched positions, scored. The honest number."""
    choices: pl.DataFrame
    """The weights chosen before each block, with its first and last day."""
    fixed: pl.DataFrame
    """Each rule on its own over the same days. The best row is what hindsight would pick."""

    def __str__(self) -> str:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True):
            return f"{self.result}\n\nChoices\n{self.choices}\n\nFixed\n{self.fixed}"


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

    def walk_forward(
        self,
        rules: Mapping[str, Strategy],
        /,
        *,
        choose: Choose | None = None,
        period: Period = "dev",
        first_train: float = FIRST_TRAIN_FRACTION,
        blocks: int = WALK_FORWARD_BLOCKS,
        check_leakage: bool = True,
    ) -> WalkForward:
        """Tests choosing between rules using only the past, as live trading would have to.

        Before each block, ``choose`` sees each rule's daily net returns on all earlier days and
        returns a weight per rule. The block then trades the weighted sum of the rules' positions.
        Blocks are fractions of the rows in ``period``, so this works on any length of history.

        Args:
            rules: Forecast functions by name. ``grid`` makes them from parameter values.
            choose: A choosing function from ``backtest.choose``. Defaults to ``best_single()``.
            period: ``"dev"`` or ``"full"``. ``"holdout"`` alone has no history to train on.
            first_train: The fraction of the period used only for training, at the start.
            blocks: The number of blocks the rest of the period is split into.
            check_leakage: As in ``run``, for each rule.

        Returns:
            ``result`` for the stitched positions, the ``choices`` per block, and ``fixed`` for
            each rule on its own over the same days.

        Raises:
            ValueError: If the arguments are invalid, ``choose`` returns unknown rules or
                non-finite weights, or there are too few training days.
        """
        if period not in ("dev", "full"):
            raise ValueError("walk_forward needs period 'dev' or 'full'.")
        if not rules:
            raise ValueError("Give at least one rule.")
        if not 0 < first_train < 1:
            raise ValueError("first_train must be between 0 and 1.")
        rows = _in_period(self.prices, self.prices, period).height
        edges = np.linspace(int(rows * first_train), rows, blocks + 1).astype(int)
        if blocks < 1 or (np.diff(edges) < 1).any():
            raise ValueError(f"Can't split {rows} days into {blocks} blocks.")
        choose = best_single() if choose is None else choose

        names = list(rules)
        positions = {
            name: self.run(
                rule, period="full", check_leakage=check_leakage, name=name
            ).positions
            for name, rule in rules.items()
        }
        returns = self._rule_returns(positions)

        pieces = [_blank(positions[names[0]]).head(int(edges[0]))]
        choices = []
        state = None
        for start, end in zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True):
            train = returns.filter(pl.col(DATE) < self.prices[DATE][start])
            train = train.drop(DATE).drop_nulls()
            if train.height < MIN_TRAIN_DAYS:
                raise ValueError(
                    f"Only {train.height} training days before {self.prices[DATE][start]} "
                    f"where every rule has a return; {MIN_TRAIN_DAYS} are needed. "
                    "Use a larger first_train."
                )
            weights, state = choose(train, state)
            weights = _checked_weights(weights, names)
            block = {
                name: frame.slice(start, end - start)
                for name, frame in positions.items()
            }
            pieces.append(_weighted_sum(block, weights))
            choices.append(
                {
                    "first_day": self.prices[DATE][start],
                    "last_day": self.prices[DATE][end - 1],
                    **weights,
                }
            )
        pieces.append(_blank(positions[names[0]]).slice(rows))

        result = self.run_positions(
            pl.concat(pieces), period=period, check_leakage=False, name="walk_forward"
        )
        fixed = [
            {"rule": name}
            | self.run_positions(
                _only_rows(frame, int(edges[0]), rows),
                period=period,
                check_leakage=False,
            ).figures
            for name, frame in positions.items()
        ]
        return WalkForward(result, pl.DataFrame(choices), pl.DataFrame(fixed))

    def _rule_returns(self, positions: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """Each rule's daily net returns, one column per rule."""
        columns = {}
        for name, frame in positions.items():
            figures = self.scoring.figures(frame, self.prices, ["returns"])
            series = cast(pl.DataFrame, figures["returns"])
            columns[DATE] = series[DATE]
            columns[name] = series["returns"]
        return pl.DataFrame(columns)


def dev_end(prices: pl.DataFrame) -> date:
    """The date of the last development position. Its return is earned two days later.

    Args:
        prices: The prices of the backtest.

    Returns:
        The last date scored when ``period="dev"``.
    """
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


def grid(forecast: Strategy, /, **values: Iterable[object]) -> dict[str, Strategy]:
    """One rule per combination of parameter values, for ``Backtest.walk_forward``.

    Args:
        forecast: A forecast function that takes the parameters as keyword arguments.
        **values: The values to try per parameter, like ``span=[8, 16, 32]``.

    Returns:
        The rules, named like ``"span=8"``, or ``"fast=8, slow=32"`` for several parameters.

    Raises:
        ValueError: If no parameters are given.

    Example:
        >>> rules = grid(single_speed, span=[4, 8, 16, 32, 64])
        >>> walk = bt.walk_forward(rules, choose=max_sharpe())
    """
    if not values:
        raise ValueError("Give at least one parameter, like span=[8, 16, 32].")
    rules = {}
    for combo in itertools.product(*values.values()):
        parameters = dict(zip(values, combo, strict=True))
        name = ", ".join(f"{key}={value}" for key, value in parameters.items())
        rules[name] = partial(forecast, **parameters)
    return rules


def _checked_weights(weights: Weights, names: list[str]) -> Weights:
    unknown = set(weights) - set(names)
    if unknown:
        raise ValueError(
            f"choose returned weights for unknown rules: {sorted(unknown)}"
        )
    full = {name: float(weights.get(name, 0.0)) for name in names}
    if not all(np.isfinite(list(full.values()))):
        raise ValueError(f"choose returned weights that aren't finite: {full}")
    return full


def _weighted_sum(positions: dict[str, pl.DataFrame], weights: Weights) -> pl.DataFrame:
    """Sum of weight times positions. A day is null only if every weighted rule is null there."""
    used = [name for name, weight in weights.items() if weight != 0]
    first = next(iter(positions.values()))
    if not used:
        return _blank(first)
    columns = instruments(first)
    terms = {c: [positions[n][c] * weights[n] for n in used] for c in columns}
    return first.select(
        DATE,
        *(
            pl.when(pl.any_horizontal(t.is_not_null() for t in terms[c]))
            .then(pl.sum_horizontal(terms[c]))
            .alias(c)
            for c in columns
        ),
    )


def _blank(positions: pl.DataFrame) -> pl.DataFrame:
    """The same frame without any positions."""
    return positions.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(column)
        for column in instruments(positions)
    )


def _only_rows(positions: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    """The same frame with positions kept only in rows ``start`` up to, not including, ``end``."""
    return pl.concat(
        [
            _blank(positions).head(start),
            positions.slice(start, end - start),
            _blank(positions).slice(end),
        ]
    )
