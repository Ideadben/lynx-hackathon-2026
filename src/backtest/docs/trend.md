# Module strategies.trend

```python
from strategies import trend
```

Trend following at several speeds. Finished strategies live in `src/strategies/`, one file each, with a `forecast`
function.

---

```python
def forecast(prices: pl.DataFrame, spans: tuple[int, ...] = (8, 16, 32)) -> pl.DataFrame
```

`forecast(prices)` returns the average vote of moving average crossovers, from -1 to +1.
