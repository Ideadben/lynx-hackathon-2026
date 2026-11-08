"""Combines the team's strategies into the positions we submit.

from portfolio import Portfolio, risk_parity

portfolio = Portfolio({"trend": trend.forecast, "carry": carry}, choose=risk_parity())
result = bt.run_positions(portfolio, name="portfolio")
"""

from portfolio.blend import Portfolio
from portfolio.weights import inverse_vol, risk_parity, shrunk_max_sharpe

__all__ = ["Portfolio", "inverse_vol", "risk_parity", "shrunk_max_sharpe"]
