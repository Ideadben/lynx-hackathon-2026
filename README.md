# Hackathon code example

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then in this folder:

```
uv sync                # create .venv and install dependencies
uv run jupyter lab     # open the notebooks in jupyter
uv add scikit-learn    # add a package
```

In VS Code, open a notebook and select the `.venv` kernel.

## Testing a strategy

Write a function that takes prices and returns a forecast: `AsOfDate` plus one column per instrument,
one row per price date. The sign is the direction, the size is the conviction. Row T may only use data
up to and including T.

```python
from backtest import Backtest

def my_forecast(prices):
    ...

bt = Backtest("prices/example_prices.csv")  # A csv path, or a polars or pandas frame.
result = bt.run(my_forecast)                # Scores the dev period by default.

print(result)                               # All key figures.
result.sharpe, result.adjusted_sharpe       # Or one at a time.
result.plot()                               # The dashboard.
```

Create one `Backtest` per data source, and run as many strategies on it as you like. Non-price series
go in as `Backtest(prices, extras)`, and the strategy is then called as `forecast(prices, extras)`.

A full tour is in `example.py`: run it with `uv run python example.py`. The reference is
`src/backtest/docs.md`, and `src/backtest/explanation.md` explains how it behaves.

`Backtest` handles volatility scaling, capping and portfolio risk scaling, and raises `LeakageError` if
the strategy uses future data. Use `period="holdout"` sparingly. Run the tests with `uv run pytest`.

## Positions format

`src/backtest/evaluation.py` takes Polars frames in wide format: an `AsOfDate` column plus one column per
instrument.

```
AsOfDate    diamonds  butter  cars  cucumber  bananas
2000-01-03  0.5       -1.0    0.0   0.2       1.3
```

You can build positions in NumPy or pandas and convert at the end.

From pandas, with dates in an index named `AsOfDate`:

```python
positions = pl.from_pandas(df.reset_index())
```

From NumPy, with a `(dates, instruments)` array:

```python
positions = pl.DataFrame(array, schema=instruments).with_columns(AsOfDate=dates)
```
