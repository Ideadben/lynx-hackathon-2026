# Library: backtest

- [Types](#types)
- [Backtest](#backtest-1)
- [Result](#result)
- [Comparison](#comparison)
- [WalkForward](#walkforward)
- [Scoring](#scoring)
- [Pipeline](#pipeline)
- [Leakage](#leakage)
- [Trend](#trend)

## Module backtest

```python
from backtest import Backtest, Comparison, Result, WalkForward
```

Test one trading strategy at a time. How it behaves is in `explanation.md`.

---

## Types

```python
Data = str | Path | pl.DataFrame | pd.DataFrame
```

A data source: a csv path, or a polars or pandas frame.

```python
Strategy = Callable[..., pl.DataFrame]
```

A forecast function. `forecast(prices)` returns the desired direction and size per date and asset.

```python
Period = Literal["dev", "holdout", "full"]
```

The dates to score: the first 75%, the rest, or all of them.

```python
ScoringKind = Literal["pdf", "lynx"]
```

The PDF's formulas, or Lynx's `evaluation.py`.

```python
class LeakageError(AssertionError)
```

Positions depend on future data. From `backtest.leakage`.

## Backtest

```python
class Backtest(
    data: Data,
    extras: Data | None = None,
    *,
    scoring: ScoringKind = "pdf",
    delta: float = 0.0002,
    costs: float = -0.02,
    vol_window: int = 100,
    risk_window: int = 100,
    cap: float | None = 20.0,
    normalize: bool = True,
)
```

One data source and its settings.

- **Raises** `ValueError` if the data has no single date column, or a column isn't numeric.
- **Raises** `TypeError` if `data` isn't a path or frame.

```python
def run(
    self,
    forecast: Strategy,
    *,
    period: Period = "dev",
    check_leakage: bool = True,
    name: str | None = None,
) -> Result
```

`bt.run(f)` turns the forecast `f` into positions and scores them.

- **Raises** `LeakageError` if `f` uses future data.
- **Raises** `ValueError` if the forecast doesn't line up with the prices, or `period` is unknown.

```python
def run_positions(
    self,
    positions: pl.DataFrame | Strategy,
    *,
    period: Period = "dev",
    check_leakage: bool = True,
    name: str | None = None,
) -> Result
```

`bt.run_positions(p)` scores the positions `p` as they are. `p` is a function returning positions, or a frame.

- **Raises** `LeakageError` if a function `p` uses future data.
- **Raises** `ValueError` if the positions don't line up with the prices.

```python
def compare(
    self,
    strategies: Iterable[Strategy] | Mapping[str, Strategy],
    *,
    period: Period = "dev",
    check_leakage: bool = True,
) -> Comparison
```

`bt.compare(l)` runs every forecast in `l`, a list of functions or a dict of names to functions.

- **Raises** `LeakageError` if a forecast uses future data.
- **Raises** `ValueError` if `l` is empty, or two functions share a name.

```python
def sweep(
    self,
    forecast: Strategy,
    /,
    *,
    period: Period = "dev",
    check_leakage: bool = True,
    **grid: Iterable[object],
) -> pl.DataFrame
```

`bt.sweep(f, span=[8, 16])` runs `f` for every combination of parameters, and returns one row per combination.

- **Raises** `LeakageError` if `f` uses future data.
- **Raises** `ValueError` if no parameters are given.

```python
def walk_forward(
    self,
    forecast: Strategy,
    /,
    *,
    period: Period = "dev",
    first_train: float = 0.3,
    blocks: int = 12,
    check_leakage: bool = True,
    **grid: Iterable[object],
) -> WalkForward
```

`bt.walk_forward(f, span=[8, 16])` picks the best parameters from earlier days before each block, then trades them
through that block.

- **Raises** `LeakageError` if `f` uses future data.
- **Raises** `ValueError` if the arguments are invalid, `period` is `"holdout"`, or there is too little history.

## Result

```python
class Result:
    name: str
    period: Period
    figures: dict[str, float]
    positions: pl.DataFrame
    prices: pl.DataFrame
    scoring: Scoring
```

The outcome of a run. Each key in `figures` is also an attribute, like `result.sharpe`: `adjusted_sharpe`, `sharpe`,
`sharpe_ex_slippage`, `sharpe_1d_lag`, `skewness`, `kurtosis`, `av_holding` and `position_bias`.

```python
@property
def returns(self) -> pl.DataFrame
```

`result.returns` is the daily net returns for the period.

```python
def by_year(self) -> pl.DataFrame
```

`result.by_year()` returns `days`, `total_return` and `sharpe` per calendar year.

```python
def by_instrument(self) -> pl.DataFrame
```

`result.by_instrument()` returns `sharpe_ex_slippage`, `av_holding` and `position_bias` per asset.

```python
def to_csv(self, path: str | Path) -> None
```

`result.to_csv(path)` saves the positions for all dates to `path`.

```python
def plot(self) -> Figure
```

`result.plot()` returns the four-panel dashboard as a matplotlib `Figure`.

## Comparison

```python
class Comparison:
    results: dict[str, Result]
    figures: pl.DataFrame
    correlations: pl.DataFrame
```

Each strategy's result, a table of key figures, and the correlations of daily net returns.

## WalkForward

```python
class WalkForward:
    result: Result
    choices: pl.DataFrame
    fixed: pl.DataFrame
```

The stitched result, the parameters chosen per block, and each parameter held fixed over the same days.

## Scoring

From `backtest.scoring`.

```python
class Scoring:
    kind: ScoringKind = "pdf"
    delta: float = 0.0002
    costs: float = -0.02
```

The scoring formula and its cost setting.

- **Raises** `ValueError` if `kind` is unknown.

```python
def figures(self, positions: pl.DataFrame, prices: pl.DataFrame, names: Sequence[str]) -> dict[str, object]
```

`scoring.figures(positions, prices, names)` calculates the named key figures.

```python
def plot(self, positions: pl.DataFrame, prices: pl.DataFrame) -> Figure
```

`scoring.plot(positions, prices)` returns the dashboard.

## Pipeline

From `backtest.pipeline`.

```python
def load_data(source: Data) -> pl.DataFrame
```

`load_data(source)` reads a data source into a standard frame.

- **Raises** `ValueError` if there is no single date column, or a column isn't numeric.
- **Raises** `TypeError` if `source` isn't a path or frame.

```python
def instruments(frame: pl.DataFrame) -> list[str]
```

`instruments(frame)` returns the column names of `frame`, without `AsOfDate`.

```python
def vol_scale(
    forecast: pl.DataFrame,
    prices: pl.DataFrame,
    vol_window: int = 100,
    cap: float | None = 20.0,
    normalize: bool = True,
) -> pl.DataFrame
```

`vol_scale(forecast, prices)` scales and caps the forecast, then divides it by asset volatility (equation 2).

- **Raises** `ValueError` if the forecast doesn't line up with the prices.

```python
def risk_scale(positions: pl.DataFrame, prices: pl.DataFrame, risk_window: int = 100) -> pl.DataFrame
```

`risk_scale(positions, prices)` divides positions by trailing strategy risk (equation 3).

```python
def to_positions(
    forecast: pl.DataFrame,
    prices: pl.DataFrame,
    vol_window: int = 100,
    risk_window: int = 100,
    cap: float | None = 20.0,
    normalize: bool = True,
) -> pl.DataFrame
```

`to_positions(forecast, prices)` applies `vol_scale`, then `risk_scale`.

- **Raises** `ValueError` if the forecast doesn't line up with the prices.

## Leakage

From `backtest.leakage`.

```python
def check_no_leakage(
    positions_fn: Callable[..., pl.DataFrame],
    prices: pl.DataFrame,
    extras: pl.DataFrame | None = None,
) -> None
```

`check_no_leakage(f, prices)` checks that past positions from `f` don't change when later data changes.

- **Raises** `LeakageError` if they do.

## Trend

From `strategies.trend`.

```python
def forecast(prices: pl.DataFrame, spans: tuple[int, ...] = (8, 16, 32)) -> pl.DataFrame
```

`forecast(prices)` returns the average vote of moving average crossovers, from -1 to +1.
