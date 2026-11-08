# Module portfolio

```python
from portfolio import Portfolio, inverse_vol, risk_parity, shrunk_max_sharpe
```

Combines the team's strategies into the positions we submit. Researchers write forecast functions; the portfolio
decides how much of each to hold.

Each strategy goes through the risk pipeline on its own, so each has about the same risk, and a weight of 0.5 means half
of a strategy's risk. The weighted sum is then risk-scaled again, because strategies that don't move together partly
cancel out. A strategy that hasn't started yet counts as 0 until it does.

---

```python
class Portfolio(
    strategies: Mapping[str, Strategy],
    weights: Weights | None = None,
    *,
    choose: Choose | None = None,
    refit_every: int = 63,
    min_train: int = 126,
    lookback: int | None = None,
    scoring: Scoring | None = None,
    vol_window: int = 100,
    risk_window: int = 100,
    cap: float | None = 20.0,
    normalize: bool = True,
)
```

A positions function: `portfolio(prices)` or `portfolio(prices, extras)` returns final positions. Pass it to
`Backtest.run_positions`, which runs the leakage check on the whole blend.

- With `weights`, the weights are fixed, like `{"trend": 0.6, "carry": 0.4}`.
- With `choose`, they are picked again every `refit_every` days from each strategy's net returns up to that day. The
  days are counted from the start of the data, so they don't move when data is added. Until `min_train` days of returns
  exist, the weights are equal. `lookback` limits training to the most recent days.
- With neither, the weights are equal.

Keep `vol_window`, `risk_window`, `cap` and `normalize` the same as the backtest's.

```python
portfolio.weights(prices) -> pl.DataFrame
```

The weights in use, one row per day they changed.

```python
portfolio.positions(prices) -> dict[str, pl.DataFrame]
```

Each strategy's positions before blending.

---

The choosing functions below have the same form as those in [`backtest.choose`](choose.md), so they also work with
`Backtest.walk_forward`. Each returns weights of at least 0 that add up to 1. `backtest.choose.equal_weights()` and
`max_sharpe()` work with `Portfolio` too.

```python
def inverse_vol() -> Choose
```

`inverse_vol()` weights each strategy by one over its volatility. Ignores correlations and average returns.

```python
def risk_parity(shrink: float = 0.0) -> Choose
```

`risk_parity()` gives every strategy the same share of the blend's risk. Strategies that move together share their risk,
so each gets less weight. `shrink` pulls correlations towards 0; at 1 it equals `inverse_vol()`.

```python
def shrunk_max_sharpe(shrink: float = 0.5, mean_shrink: float = 0.5) -> Choose
```

`shrunk_max_sharpe()` is `max_sharpe()` on steadier estimates. `shrink` pulls correlations towards 0, and `mean_shrink`
pulls each strategy's average return towards the average of all. With both at 0 it is `max_sharpe()`; with
`mean_shrink=1` only risk and correlation decide.

Average returns are the noisiest estimate, so `max_sharpe()` tends to put everything on whichever strategy happened to
do best. Check any choice with `walk_forward` before trusting it.
