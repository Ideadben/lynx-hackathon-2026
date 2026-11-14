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

A full tour is in `src/examples/example.py`, and `src/examples/walk_forward_example.py` shows how to
test parameter choices. Run them with `uv run python src/examples/example.py`. The reference is in
`src/backtest/docs/`, one page per module, and `src/backtest/docs/explanation.md` explains how it behaves.

`Backtest` handles volatility scaling, capping and portfolio risk scaling, and raises `LeakageError` if
the strategy uses future data. Use `period="holdout"` sparingly. Run the tests with `uv run pytest`.

## Combining strategies

`Portfolio` blends the team's forecast functions into the positions we submit. Give it fixed weights, or a choosing
function that picks weights from past returns every quarter:

```python
from portfolio import Portfolio, risk_parity

portfolio = Portfolio({"trend": trend.forecast, "carry": carry}, choose=risk_parity())
result = bt.run_positions(portfolio, name="portfolio")   # Leakage-checked like any strategy.
portfolio.weights(bt.prices)                              # The weights over time.
```

See `src/examples/portfolio_example.py` and `src/backtest/docs/portfolio.md`.

## The final submission

`src/portfolio/final.py` builds the portfolio we submit and writes its positions:

```
uv run python -m portfolio.final                                   # in-sample data -> positions.csv
uv run python -m portfolio.final --prices p.csv --macro m.csv --out positions.csv
```

It blends four strategies by share of risk: 40% `tikhonov.forecast_garch` (a Tikhonov-regularised
linear filter of the last 250 daily returns, with GARCH vol timing as risk management) and 20% each
for the three parts of `strategies/triple.py` (slow trend, long stocks and bonds, the macro rule).
The whole blend passes the leakage test before the file is written, and the command prints each
strategy's Sharpe on the 75 / 15 / 10 dev / validation / test windows. Change the weights in
`final.WEIGHTS`. The GARCH code is `strategies/garch.py`, copied from Søren's research file.

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
