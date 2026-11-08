# Module backtest.leakage

```python
from backtest.leakage import LeakageError, check_no_leakage
```

Checks that a strategy never uses future data.

---

```python
class LeakageError(AssertionError)
```

Positions depend on future data. From `backtest.leakage`.

```python
def check_no_leakage(
    positions_fn: Callable[..., pl.DataFrame],
    prices: pl.DataFrame,
    extras: pl.DataFrame | None = None,
) -> None
```

`check_no_leakage(f, prices)` checks that past positions from `f` don't change when later data changes.

- **Raises** `LeakageError` if they do.
