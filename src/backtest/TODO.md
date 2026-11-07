# Backtest todo

Tests one strategy at a time: forecast in, key figures and plots out. This package must never import
`strategies` or `portfolio`. How it behaves is in `docs.md`. The full plan is
in `dirs/strategy_and_mixer_todo.md`.

```python
from backtest import Backtest

bt = Backtest("prices/example_prices.csv")
result = bt.run(my_forecast)
print(result.sharpe, result.adjusted_sharpe)
result.plot()
```

## Done

Written and tested (`uv run pytest`, passing), but not committed yet.

- [x] `pipeline.py`: `load_data`, `vol_scale` (eq. 2), `risk_scale` (eq. 3), `to_positions`.
- [x] Forecasts scaled to an average size of 10 and capped at 20. Our own convention, not a PDF rule.
      The average only uses data so far, because a full-sample average would leak. Optional:
      `normalize=False` and `cap=None`.
- [x] `leakage.py`: `check_no_leakage` runs both PDF tests at three dates, and raises `LeakageError`.
- [x] `runner.py`: `Backtest(data, extras)` holds a data source, and `.run(forecast)` returns a `Result`.
- [x] `Result` has every key figure as an attribute (`result.sharpe`), plus `figures`, `positions` and
      `.plot()`.
- [x] Data can be a csv path, or a polars or pandas frame. No paths are hardcoded.
- [x] Shared dev/holdout split: 75%, same as Benjamin's. `run` scores dev by default.
- [x] Any number of instruments. Missing prices are forward-filled. Optional `extras` frame.
- [x] Csv files are scanned in full, so empty cells at the start don't turn a column into text.
- [x] `bt.run_positions`, `bt.compare` and `bt.sweep`.
- [x] `result.returns`, `result.by_year()`, `result.by_instrument()` and `result.to_csv(path)`.
- [x] `scoring.py`: PDF scoring by default (flat `delta`, 252 days), and `scoring="lynx"` for `evaluation.py`.
- [x] Tested on messy made-up data: assets that start late or in the holdout, gaps, and more assets.
- [x] Positions match `BENJAMIN/ewma.ipynb`. With `scoring="lynx"`, scores match
      `BENJAMIN/lstm/results/summary.csv`.

`evaluation.py` is the Lynx scorer. Do not change it.

## Still to do

- [ ] Robustness report: cost sensitivity, placebo and bootstrap. `sweep` and `by_year` cover the rest.
- [ ] Check `load_data` and `extras` against the real event data. The `extras` format is a guess, and
      forward-filling missing prices is a default we may want to change.
- [ ] Confirm with Lynx which scoring they use. We follow the PDF by default. If they use `evaluation.py`,
      switch the default to `scoring="lynx"`, because its costs are about 100-500x higher.

Not planned: purged CV and walk-forward, because we aren't using ML for now.
