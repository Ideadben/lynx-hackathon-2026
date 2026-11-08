# Module backtest

```python
from backtest import Backtest, Comparison, Result, WalkForward, grid
```

Test one trading strategy at a time. The statistics in `stats.md` can also be imported from here. How it behaves is in
`explanation.md`.

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
    rules: Mapping[str, Strategy],
    /,
    *,
    choose: Choose | None = None,
    period: Period = "dev",
    first_train: float = 0.3,
    blocks: int = 12,
    check_leakage: bool = True,
) -> WalkForward
```

`bt.walk_forward(rules, choose=c)` asks `c` for weights from earlier days before each block, then trades the weighted
rules through that block. `choose` defaults to `best_single()`.

- **Raises** `LeakageError` if a rule uses future data.
- **Raises** `ValueError` if the arguments are invalid, `period` is `"holdout"`, `choose` returns unknown rules or
  non-finite weights, or there are too few training days.

```python
def grid(forecast: Strategy, /, **values: Iterable[object]) -> dict[str, Strategy]
```

`grid(f, span=[8, 16])` returns one rule per combination of values, named like `"span=8"`.

- **Raises** `ValueError` if no parameters are given.

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

The stitched result, the weights chosen per block, and each rule on its own over the same days.
