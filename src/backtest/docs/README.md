# Library: backtest

Reference for the backtester, one page per module. How it all behaves is in [`explanation.md`](explanation.md), and
working scripts are in `src/examples/`.

| Module | Contents |
| --- | --- |
| [`backtest`](backtest.md) | `Backtest`, `Result`, `Comparison`, `WalkForward`, `grid` |
| [`backtest.choose`](choose.md) | Choosing functions for `walk_forward` |
| [`backtest.stats`](stats.md) | `sharpe`, `adjusted_sharpe`, `bootstrap`, `paired_bootstrap` |
| [`backtest.scoring`](scoring.md) | `Scoring`: PDF or Lynx formulas |
| [`backtest.pipeline`](pipeline.md) | Loading data, and forecast to positions |
| [`backtest.leakage`](leakage.md) | `check_no_leakage`, `LeakageError` |
| [`strategies.trend`](trend.md) | The trend strategy |
