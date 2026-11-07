"""Backtesting tools shared by all strategies.

from backtest import Backtest

bt = Backtest("path/to/prices.csv")
result = bt.run(my_forecast)
print(result.sharpe, result.adjusted_sharpe)
result.plot()
"""

from backtest.runner import Backtest, Comparison, Result, WalkForward

__all__ = ["Backtest", "Comparison", "Result", "WalkForward"]
