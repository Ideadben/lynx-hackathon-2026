# Portfolio todo

Combines the finished strategies from `src/strategies/` into the one final set of positions we submit.
The full plan is in `dirs/strategy_and_mixer_todo.md`.

```python
from strategies.trend import forecast as trend
from strategies.reversion import forecast as reversion
from strategies.macro import forecast as macro

WEIGHTS = {trend: 1 / 3, reversion: 1 / 3, macro: 1 / 3}

# each strategy -> positions, take the weighted average, rescale risk once at the end
```

## Done

- [x] Folder exists. Nothing is written yet.
- [x] Decided: blend positions, not raw forecasts, so a weight of 1/3 means a third of the risk.

- [x] `bt.run_positions` and `result.returns` are ready in `backtest`. Pass `run_positions` a function, not
      a frame, so the blend gets the leakage test.

## Still to do

Needs first:

- [x] `src/strategies/trend.py` exists, with `forecast(prices)`.
- [ ] `reversion.py` and `macro.py`, with the same `forecast` function.

`mixer.py`:

- [ ] Run each strategy through `to_positions`, so they are all at equal risk.
- [ ] Take the weighted average of positions. Start with equal weights. Only change them for a strong
      reason, and fit any weights on dev only.
- [ ] Apply `risk_scale` once at the end, to restore the risk that diversification removes.
- [ ] Report the correlation of strategy returns (`bt.compare`), and each strategy's contribution to the
      blend.
- [ ] Optional: covariance-aware sizing with a shrunk rolling covariance.

`final.py`:

- [ ] Command line: `python -m portfolio.final --data ... --out positions.csv`, with frozen parameters.
- [ ] Run the leakage test on the final blended positions, not just on each strategy.
- [ ] Save with `result.to_csv`. Warm-up rows are empty, like the Lynx Quickstart, so confirm with Lynx
      that this is fine.
- [ ] Dry run: hide the last two years and run it end to end, as we will at the event.
