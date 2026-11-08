# Module backtest.stats

```python
from backtest import Bootstrap, adjusted_sharpe, bootstrap, paired_bootstrap, sharpe
```

Sharpe ratios of daily returns, and bootstrap intervals for them.

---

```python
def sharpe(returns: Returns, days_per_year: int = 252) -> float | np.ndarray
```

`sharpe(r)` returns the yearly Sharpe ratio (equation 4). A 2D array gives one value per column.

```python
def adjusted_sharpe(returns: Returns, days_per_year: int = 252) -> float | np.ndarray
```

`adjusted_sharpe(r)` returns the Adjusted Sharpe ratio (equation 5). A 2D array gives one value per column.

- **Raises** `ValueError` if a 2D array contains NaN.

```python
def bootstrap(
    returns: Returns,
    statistic: Statistic = adjusted_sharpe,
    *,
    block: float | None = 10.0,
    samples: int = 10_000,
    seed: int = 0,
) -> Bootstrap
```

`bootstrap(r)` resamples `r` in blocks, and returns the 95% interval of `statistic`.

```python
def paired_bootstrap(
    first: Returns,
    second: Returns,
    statistic: Statistic = adjusted_sharpe,
    *,
    block: float | None = 10.0,
    samples: int = 10_000,
    seed: int = 0,
) -> Bootstrap
```

`paired_bootstrap(a, b)` resamples the same days of `a` and `b`, and returns the interval of the difference.

- **Raises** `ValueError` if arrays differ in length.

```python
class Bootstrap:
    estimate: float
    low: float
    high: float
    p_at_most_zero: float
    block: float
    draws: np.ndarray
```

The statistic, its 95% interval, the share of draws at or below 0, the average block length, and every draw.
