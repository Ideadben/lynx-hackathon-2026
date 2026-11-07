# Backtest docs

`backtest` tests one strategy at a time. You give it data and a forecast function, and it gives you back
key figures, positions and plots. To combine strategies, see `src/portfolio/TODO.md` (not built yet).

## Quick start

```python
import polars as pl
from backtest import Backtest


def trend(prices):
    assets = [c for c in prices.columns if c != "AsOfDate"]
    fast = pl.col(assets).ewm_mean(span=16, adjust=False, min_samples=16)
    slow = pl.col(assets).ewm_mean(span=64, adjust=False, min_samples=64)
    return prices.with_columns((fast - slow).sign())


bt = Backtest("prices/example_prices.csv")
result = bt.run(trend)

print(result)            # All key figures.
print(result.sharpe)     # One key figure.
result.plot()            # The dashboard.
```

## The forecast function

This is the only part a researcher writes. It answers one question for every day and every asset:
**how much do we want to be long or short, using only what we know at the end of that day?**

**Input:** a polars frame of prices, with an `AsOfDate` column plus one column per asset. With extras, the
function takes a second argument: `forecast(prices, extras)`.

**Output:** a polars frame with the same dates and the same asset columns.

| AsOfDate | diamonds | butter | cars |
| --- | --- | --- | --- |
| 2010-01-04 | 1.0 | -1.0 | 0.3 |
| 2010-01-05 | 1.0 | -0.5 | null |

- **Positive** means long, **negative** means short, and **zero** means no opinion.
- **Bigger** means more confident. `+2` is twice the size of `+1` on the same asset.
- **`null`** means no position, for example while a moving average warms up.

Rules:

- One row per price date, in the same order. Anything else raises `ValueError`.
- A column for every asset. Missing columns raise `ValueError`, and extra columns are ignored.
- Row T may only use data up to and including T. Otherwise `run` raises `LeakageError`.
- Don't size for risk yourself. Don't divide by volatility or try to control overall risk: the pipeline
  does that.
- By default the units don't matter, because each asset's forecast is rescaled. This also means a permanent
  size difference between assets is removed: always forecasting `2` on one asset and `1` on another gives
  both the same size. A forecast only changes size relative to that asset's own past. To keep your sizes
  as they are, use `normalize=False`.

To set parameters, wrap the function:

```python
from functools import partial

def trend_with_span(prices, span):
    ...

bt.run(partial(trend_with_span, span=32))
```

### What `span` means

`span` is a setting of the moving average in the quick start, not of the backtester. It is roughly how many
days of history the average looks at. Every past day counts, but older days count less. A span of 16 puts
most of the weight on roughly the last 16 days.

The trend example compares two averages, with spans of 16 and 64, so it goes long when the last ~16 days are
above the last ~64 days. A small span reacts quickly, trades more and pays more in costs. A big span smooths
out noise, but reacts late.

## `Backtest`

```python
Backtest(data, extras=None, *, scoring="pdf", delta=0.0002, costs=-0.02,
         vol_window=100, risk_window=100, cap=20.0, normalize=True)
```

Holds one data source and its settings. Make one instance per data source, then run as many strategies on
it as you like.

| Argument | Meaning |
| --- | --- |
| `data` | Prices: a csv path, or a polars or pandas frame. |
| `extras` | Optional non-price series, in the same forms. |
| `scoring` | `"pdf"` for the PDF's formulas, `"lynx"` for `evaluation.py`. See "Scoring" below. |
| `delta` | Flat cost per unit traded, for `"pdf"`. |
| `costs` | Cost per unit traded in daily standard deviations, for `"lynx"`. Negative. |
| `vol_window` | Days used to estimate each asset's volatility. |
| `risk_window` | Days used to estimate the strategy's risk. |
| `cap` | Largest absolute forecast, after scaling to an average size of 10. `None` for no cap. |
| `normalize` | Scale each asset to an average absolute forecast of 10. See below. |

### Forecast scaling options

| Setting | What happens | Use it when |
| --- | --- | --- |
| Default | Scaled to an average size of 10, capped at 20. | Most strategies. |
| `cap=None` | Scaled, not capped. | You want extreme signals to keep their full size. |
| `normalize=False` | Used as it is. `cap` has no effect. | Your sizes already compare across assets. |

The scaling only changes the relative size of positions. Overall risk is still controlled either way. With
`normalize=False`, a forecast in price units gives more weight to assets with bigger price moves, so your
forecast should already be in comparable units, such as a z-score. A `sign` forecast gives the same
positions either way, because every value is already the same size.

How data is read:

- There must be exactly one date column, called `AsOfDate`, `dates` or `date` (any case), or a pandas date
  index. It is renamed to `AsOfDate`. Text dates must be in `YYYY-MM-DD` format.
- Every other column must be numeric. In `data`, each one counts as an asset.
- Rows are sorted by date, and values become floats.
- A csv is scanned in full to work out column types, so an asset that starts late still reads as numbers.
- A missing value repeats the last known value, which uses no future data. Values missing at the very
  start stay `null`.
- `extras` are not lined up with the price dates. The forecast function has to do that if the dates differ.

## `Backtest.run`

```python
bt.run(forecast, *, period="dev", check_leakage=True)
```

| Argument | Meaning |
| --- | --- |
| `forecast` | The forecast function. |
| `period` | `"dev"`, `"holdout"` or `"full"`: which dates the figures cover. |
| `check_leakage` | Runs the leakage test. Only turn it off for slow strategies during a search. |

`dev` is the first 75% of dates and `holdout` is the rest. The function still runs on all dates, and
`period` only chooses which positions are scored. Check the holdout sparingly: each look makes it less
trustworthy.

With the leakage test on, your function runs 8 times per `run`. Keep that in mind for slow models.

## `Result`

| Attribute | Meaning |
| --- | --- |
| `adjusted_sharpe` | The competition score: Sharpe after costs, penalised for negative skew and fat tails. |
| `sharpe` | Yearly Sharpe after costs. |
| `sharpe_ex_slippage` | Yearly Sharpe before costs. |
| `sharpe_1d_lag` | Sharpe with positions one day later. A big drop suggests the signal is very short-lived. |
| `skewness` | Skew of daily returns. |
| `kurtosis` | Kurtosis of daily returns. Normal returns give 3. |
| `av_holding` | Average days a position is held. |
| `position_bias` | Long/short tilt, from -1 (short only) to 1 (long only). |
| `figures` | All of the above as a dict. |
| `positions` | Final positions for all dates, not just `period`. |
| `returns` | Daily net returns for `period`: `AsOfDate` and `returns`. Warm-up days are left out. |
| `plot()` | The four-panel dashboard for `period`. Returns a matplotlib `Figure`. |
| `by_year()` | Per calendar year in `period`: `days`, `total_return` and `sharpe` after costs. |
| `by_instrument()` | Per asset in `period`: `sharpe_ex_slippage`, `av_holding` and `position_bias`. |
| `to_csv(path)` | Saves `positions` to a csv. |
| `name`, `period`, `prices`, `scoring` | What was run. |

`print(result)` shows all the figures as a table.

- `by_instrument()` shows Sharpe **before** costs, the same as the Lynx dashboard. An asset with no data
  in `period` shows NaN. With `scoring="lynx"`, such an asset makes `by_instrument()` and `plot()` fail,
  because `evaluation.py` can't handle it.
- `by_year()` can show a partial first or last year, so check `days`.
- `to_csv` always saves all dates, whatever `period` was, because the submission needs the full run.
  Warm-up rows have empty cells, the same as the Lynx Quickstart notebook.

## `Backtest.run_positions`

```python
bt.run_positions(positions, *, period="dev", check_leakage=True, name=None)
```

Scores positions exactly as given, skipping the scaling, capping and risk steps. Use it for the final blend.
`positions` can be either of these:

- **A function** called like a forecast, returning final positions. This gets the leakage test, so use it
  for anything you submit.
- **A finished frame** of positions. A frame can't be rerun on changed data, so it **skips the leakage test**.

Either way, it needs one row per price date and a column for every asset. `run` calls this method after the
pipeline, so both score the same way.

## `Backtest.compare`

```python
comparison = bt.compare([trend, reversion], period="dev", check_leakage=True)
comparison = bt.compare({"fast": partial(f, span=8), "slow": partial(f, span=32)})

print(comparison)
```

Runs each forecast and returns a `Comparison`:

| Attribute | Meaning |
| --- | --- |
| `figures` | One row per strategy, with all key figures. |
| `correlations` | Correlations of daily net returns, using dates where every strategy has a return. |
| `results` | Each strategy's full `Result`, by name. |

Use a dict when functions share a name, such as two `partial` versions of one function. A list raises
`ValueError` in that case. Low correlation is what makes a blend useful.

## `Backtest.sweep`

```python
table = bt.sweep(trend_with_span, span=[8, 16, 32])
table = bt.sweep(f, fast=[8, 16], slow=[32, 64], check_leakage=False)
```

`sweep` runs one strategy with several parameter values, so you don't have to run each by hand. With more than
one parameter it tries every combination, so the second example makes 4 runs. It returns one row per
combination: the parameters, then the key figures. The function must take them as keyword arguments, and `period` and
`check_leakage` can't be used as parameter names.

Use it to see whether a result is stable across nearby settings, not to pick the single best row. If
neighbouring values all score well, the strategy probably has a real edge. If one value stands out, it may
just be luck on this data. The more
settings you try, the more the best one is flattered by luck. The number of runs multiplies quickly, and each
run calls your function 8 times with the leakage test on.

## What happens inside `run`

1. **Forecast:** your function runs on the data.
2. **Scale and cap:** each asset is scaled so its average absolute forecast so far is 10, then capped at
   `cap`. The average only uses past data, because a full-sample average would leak. `normalize=False`
   skips this step.
3. **Divide by asset volatility** (equation 2 in the PDF): volatility is
   `sqrt(mean(r^2))` over `vol_window` days, so wilder assets get smaller positions.
4. **Divide by strategy risk** (equation 3): strategy risk is the standard deviation of the strategy's daily
   return over `risk_window` days, using the two-day lag. This keeps total risk steady, at about 1 per day.
   It uses a standard deviation like the Lynx notebook, while the PDF's equation skips subtracting the
   mean. The difference is tiny, because the average daily return is close to zero.
5. **Score:** returns, costs and key figures are calculated as in "Scoring" below.

Early rows are `null` until the windows fill. With a 64-day average and both 100-day windows, the first
position is on row 201.

If volatility is zero, or a value comes out as NaN or infinite, the position becomes `null`.

## Timing

A position dated T uses data through the close of T, trades on T + 1 and earns the return of T + 2. The
scoring applies this two-day lag, so don't shift your forecast yourself.

## Scoring, costs and slippage

There are two ways to score, set with `Backtest(data, scoring=...)`. The default follows the PDF.

| | `scoring="pdf"` (default) | `scoring="lynx"` |
| --- | --- | --- |
| Cost per unit traded | Flat `delta`, default 0.0002 (equation 1) | `costs` (default -0.02) x asset's volatility |
| Days per year | 252 (equation 4) | 260 |
| Holding period | Equation 6, on raw positions | Volatility-weighted |
| Code | `scoring.py` | Lynx's `evaluation.py` |

Both use excess kurtosis in the Adjusted Sharpe, as the PDF's reference does. `sharpe` and `adjusted_sharpe`
include costs, and `sharpe_ex_slippage` doesn't. Neither models market impact or a separate bid-ask spread,
and trades fill at the close, as the PDF assumes.

**The two differ a lot on costs.** On the example data, the Lynx cost is about 100-500x the PDF cost, and PDF
costs barely affect any strategy. A fast strategy can look fine under `"pdf"` and lose money under `"lynx"`, so
check fast strategies with both until Lynx confirm which one they score with.

The first day of a position has no earlier position to trade from, so it has no return, gross or net.

## Leakage test

The test checks that past positions don't change when later data changes. It runs at three dates: 50% and
75% of the way through the data, and 20 days from the end. At each date it:

1. Removes everything after the date, and reruns the strategy.
2. Changes every value after the date, and reruns the strategy.

Positions up to that date must match, or it raises `LeakageError`. Both prices and extras are changed.
Very small floating-point differences are allowed.

It tests the full pipeline, not just your function. It can't catch future knowledge you added by hand, like
a parameter chosen after looking at the holdout.

## Errors

| Error | Cause |
| --- | --- |
| `LeakageError` | Positions depend on future data. |
| `ValueError: ...no column for...` | The forecast or positions frame is missing an asset. |
| `ValueError: ...one row per price date...` | The frame's dates don't match the price dates. |
| `ValueError: Expected one date column...` | No recognised date column, or more than one. |
| `ValueError: All columns except the date must be numeric...` | A column holds text. |
| `TypeError: Expected a csv path...` | `data` isn't a path or frame. |
| `ValueError: Strategies share a name...` | Two functions in a `compare` list have the same name. |
| `ValueError: There are no strategies...` | `compare` got an empty list. |
| `ValueError: Give at least one parameter...` | `sweep` got no parameters. |

## Results in a notebook

`figures`, `correlations`, `by_year()`, `by_instrument()` and `sweep(...)` all return polars frames, so you can
sort, filter or plot them as usual. Use `.to_pandas()` if you prefer pandas.

## Strategies

Finished strategies live in `src/strategies/`, one file each, with a function called `forecast`:

```python
from strategies import trend

result = bt.run(trend.forecast)
```

`trend.py` is the first one. For spans of 8, 16 and 32, it compares a recent average price with one four times
slower, and each speed votes +1 or -1. The forecast is the average vote.

## Other functions

These aren't part of the top-level import, but are useful for `portfolio`:

```python
from backtest.pipeline import load_data, to_positions, vol_scale, risk_scale
from backtest.leakage import check_no_leakage
from backtest.scoring import Scoring
```

## Tests

`uv run pytest` runs `tests/test_backtest.py`.
