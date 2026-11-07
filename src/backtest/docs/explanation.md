# How the backtester behaves

`docs.md` lists the types and functions. This file explains how they behave. Working scripts are in
`src/examples/`.

## Writing a forecast

A forecast is the only part a researcher writes. It answers one question for every day and asset: **how much do
we want to be long or short, using only what we know at the end of that day?**

| AsOfDate | diamonds | butter | cars |
| --- | --- | --- | --- |
| 2010-01-04 | 1.0 | -1.0 | 0.3 |
| 2010-01-05 | 1.0 | -0.5 | null |

- **Positive** is long, **negative** is short, and **zero** is no opinion.
- **Bigger** is more confident. `+2` is twice the size of `+1` on the same asset.
- **`null`** is no position, for example while a moving average warms up.

The rules:

- One row per price date, in the same order, and a column for every asset. Extra columns are ignored.
- Row T may only use data up to and including T.
- Don't shift for the two-day lag. Scoring applies it.
- Don't size for risk. The pipeline divides by volatility and controls overall risk.

With extras, the forecast is called as `forecast(prices, extras)`. Extras aren't lined up with the price dates, so
the forecast has to do that if the dates differ.

To set parameters, use `functools.partial`:

```python
bt.run(partial(trend.forecast, spans=(16,)))
```

## From forecast to positions

1. **Forecast:** your function runs on all dates.
2. **Scale and cap:** each asset is scaled so its average absolute forecast so far is 10, then capped. The average
   only uses past data, because a full-sample average would leak.
3. **Divide by asset volatility** (equation 2): `sqrt(mean(r^2))` over `vol_window` days.
4. **Divide by strategy risk** (equation 3): the standard deviation of daily strategy returns over `risk_window`
   days, using the two-day lag. This keeps risk steady at about 1 per day. The PDF's equation doesn't subtract the
   mean, but the difference is tiny because average daily returns are close to zero.
5. **Score:** see "Scoring".

Early rows are `null` until the windows fill. If volatility is zero, or a value is NaN or infinite, the position
becomes `null`.

### Scaling options

| Setting | What happens | Use it when |
| --- | --- | --- |
| Default | Scaled to an average size of 10, capped at 20. | Most strategies. |
| `cap=None` | Scaled, not capped. | Extreme signals should keep their full size. |
| `normalize=False` | Used as it is. `cap` has no effect. | Sizes already compare across assets, like a z-score. |

Each asset is scaled against its own past, so a permanent size difference between assets is removed: always
forecasting `2` on one and `1` on another gives both the same size. `normalize=False` keeps that difference.

Overall risk is controlled in every case. With `normalize=False`, a forecast in price units gives more weight to
assets with bigger price moves. A `sign` forecast gives the same positions either way.

## Periods

The function always runs on all dates, and `period` only chooses which positions are scored. Check the holdout
sparingly: each look makes it less trustworthy.

## Timing

A position dated T uses data through the close of T, trades on T + 1 and earns the return of T + 2.

## Scoring

| | `"pdf"` (default) | `"lynx"` |
| --- | --- | --- |
| Cost per unit traded | Flat `delta` (equation 1) | `costs` x asset's trailing volatility |
| Days per year | 252 (equation 4) | 260 |
| Holding period | Equation 6, on raw positions | Volatility-weighted |
| Code | `scoring.py` | Lynx's `evaluation.py` |

Both use excess kurtosis in the Adjusted Sharpe, as the PDF's reference does. Neither models market impact or a
separate bid-ask spread, and trades fill at the close, as the PDF assumes. The first day of a position has no
earlier position to trade from, so it has no return, gross or net.

**The two differ a lot on costs.** On the example data the Lynx cost is about 100-500x the PDF cost, and PDF costs
barely affect any strategy. A fast strategy can look fine under `"pdf"` and lose money under `"lynx"`, so check
fast strategies with both until Lynx confirm which one they score with.

## Key figures

| Figure | Meaning |
| --- | --- |
| `adjusted_sharpe` | The competition score: Sharpe penalised for negative skew and fat tails. |
| `sharpe` | Yearly Sharpe after costs. |
| `sharpe_ex_slippage` | Yearly Sharpe before costs. |
| `sharpe_1d_lag` | Sharpe with positions one day later. A big drop means a short-lived signal. |
| `skewness` | Skew of daily returns. |
| `kurtosis` | Kurtosis of daily returns. Normal returns give 3. |
| `av_holding` | Average days a position is held. |
| `position_bias` | Long/short tilt, from -1 (short only) to 1 (long only). |

## Results

- `positions` and `to_csv` cover all dates, whatever the period, because the submission needs the full run. Warm-up
  rows in the csv have empty cells, the same as the Lynx Quickstart notebook.
- `returns` leaves out warm-up days.
- `by_year` can show a partial first or last year, so check `days`.
- `by_instrument` shows Sharpe before costs, the same as the dashboard. An asset with no data in the period shows
  NaN. With `scoring="lynx"`, such an asset makes `by_instrument` and `plot` fail, because `evaluation.py` can't
  handle it.

## `run_positions`

This skips the scaling, capping and risk steps. Use it for the final blend.

Pass a function for anything you submit. A finished frame can't be rerun on changed data, so it skips the leakage
test. `run` calls this method after the pipeline, so both score the same way.

## `compare`

Each strategy's figures cover its own dates, so strategies with different warm-ups are scored over slightly
different days. Treat small gaps with care. The correlations use the dates every strategy shares, and low
correlation is what makes a blend useful.

Use a dict of names when functions share a name, such as two `partial` versions of one function.

## `sweep`

With more than one parameter it tries every combination, so two parameters with two values each make 4 runs. The
function must take the parameters as keyword arguments, and they can't be called `period` or `check_leakage`.

Use it to see whether a result is stable, not to pick the best row. If neighbouring values score well, the effect
is probably real. If one value stands out, it may be luck, and the more values you try, the more luck flatters the
top score.

## `walk_forward`

`sweep` shows how each parameter did with hindsight. `walk_forward` tests the process of choosing one:

```python
walk = bt.walk_forward(single_speed, span=[8, 16, 32])
print(walk)
```

1. The first `first_train` of the period is only used for training.
2. The rest is split into `blocks` equal blocks.
3. Before each block, it picks the parameters with the best Adjusted Sharpe on all earlier days. It only uses
   returns already known at that point.
4. It trades those parameters through the block, and the blocks are stitched together into `result`.

`result` is the honest number, because every traded day used a parameter chosen from earlier data. The best row
of `fixed` is what hindsight would have picked over the same days, so the gap between them shows how much the
choice was flattered. `choices` shows what was picked: if it jumps around, the choice is mostly noise.

- **Costs:** switching parameters at a block boundary is a real trade, so it is charged.
- **Slow to adapt:** training uses all earlier days, so old years keep counting and a change in the market takes a
  long time to show up in the choice.
- **Holdout:** the default `period="dev"` leaves the holdout untouched. `"full"` includes it, so it counts as a look
  at the holdout. `"holdout"` alone is rejected, because it has no history to train on.
- **Grid:** picking the grid after looking at results brings hindsight back. Use a wide grid.
- **Time:** it runs the forecast once per combination, plus the leakage test for each one.

## Different data

Nothing depends on calendar years or a particular start date:

- `dev` and `holdout` are fractions of the rows.
- `walk_forward` blocks are fractions of the rows. On 23 years of data the blocks are about a year, and on 6 years
  they are about 3 months.
- Windows such as `vol_window` count trading days, so gaps in the calendar don't matter.
- Any number of assets works, and assets can start late.

Two things do not adapt by themselves:

- **Too little history:** `walk_forward` needs 126 days after warm-up to judge a parameter, and raises `ValueError`
  otherwise. A larger `first_train` helps. Dates inside `by_year` can also show partial years.
- **Parameters:** values tuned on the example data, like the spans in `trend.py`, may not suit other data. Rerun
  `sweep` and `walk_forward` on the real data.

The 252 days per year assumes daily data, as the PDF describes.

## Leakage test

With `check_leakage` on, your function runs 8 times per `run`. Only turn it off for slow strategies during a search.

The test runs at three dates: 50% and 75% of the way through the data, and 20 days from the end. At each date it:

1. Removes everything after the date, and reruns the strategy.
2. Changes every value after the date, and reruns the strategy.

Positions up to that date must match. Both prices and extras are changed, and very small floating-point differences
are allowed. It can't catch future knowledge added by hand, like a parameter chosen after looking at the holdout.

## Loading data

- There must be exactly one date column, called `AsOfDate`, `dates` or `date` (any case), or a pandas date index.
  It is renamed to `AsOfDate`. Text dates must be in `YYYY-MM-DD` format.
- Every other column must be numeric. In prices, each one counts as an asset.
- Rows are sorted by date, and values become floats.
- A csv is scanned in full to work out column types, so an asset that starts late still reads as numbers.
- A missing value repeats the last known value, which uses no future data. Values missing at the very start stay
  `null`.

## Trend strategy

For each span, `trend.forecast` compares a recent average price with one four times slower. If the recent one is
higher, that speed votes +1, otherwise -1. The forecast is the average vote, and days stay `null` until every speed
has warmed up.

`span` is roughly how many days of history an average looks at. Every past day counts, but older days count less.
A small span reacts quickly, trades more and pays more in costs. A big span smooths out noise, but reacts late.

## Tests

`uv run pytest` runs `tests/test_backtest.py`.
