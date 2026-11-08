# Examples

Run any of them from the repo root, for example `uv run python src/examples/quickstart.py`. Files they save go
to `src/examples/output/`, which Git ignores.

| Script | Shows |
| --- | --- |
| `quickstart.py` | Write a forecast, backtest it, read the result. Start here. |
| `parameters_example.py` | Setting parameters with `partial`, and trying many with `sweep`. |
| `compare_example.py` | Several strategies side by side, and a paired bootstrap between them. |
| `bootstrap_example.py` | How uncertain a Sharpe ratio is, with any statistic. |
| `costs_example.py` | PDF against Lynx costs, and where a fast strategy stops paying. |
| `data_example.py` | Loading csv, polars or pandas data, and using extra series. |
| `leakage_example.py` | Strategies that use future data by mistake, and the error they raise. |
| `blend_example.py` | Combining strategies into final positions, and saving them. |
| `portfolio_example.py` | Combining strategies with `Portfolio`, comparing weighting methods, and saving the submission. |
| `walk_forward_example.py` | Choosing between rules from past data only, with several choosing functions. |
| `example.py` | A longer tour of most features. |

The reference is in `src/backtest/docs/`.
