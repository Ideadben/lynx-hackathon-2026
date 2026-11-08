# Module backtest.pipeline

```python
from backtest.pipeline import instruments, load_data, risk_scale, to_positions, vol_scale
```

Reading data, and turning a forecast into risk-managed positions.

---

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
