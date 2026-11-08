"""Combining the team's strategies with ``Portfolio``, and saving the submission.

Run it with: uv run python src/examples/portfolio_example.py
"""

from functools import partial
from pathlib import Path

from backtest import Backtest
from backtest.choose import equal_weights, max_sharpe
from portfolio import Portfolio, risk_parity, shrunk_max_sharpe
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "example_prices.csv"

# The researchers' forecast functions, by name. Here three trend speeds stand in for them.
STRATEGIES = {
    "fast": partial(trend.forecast, spans=(8,)),
    "mid": partial(trend.forecast, spans=(32,)),
    "slow": partial(trend.forecast, spans=(128,)),
}


def main() -> None:
    bt = Backtest(PRICES)
    for name, strategy in STRATEGIES.items():
        print(f"{name:<18} {bt.run(strategy).adjusted_sharpe:.2f}")

    # Each portfolio re-chooses its weights every quarter from past returns only.
    choices = {
        "equal": equal_weights(),
        "risk parity": risk_parity(),
        "max sharpe": max_sharpe(),
        "shrunk max sharpe": shrunk_max_sharpe(),
    }
    for name, choose in choices.items():
        result = bt.run_positions(Portfolio(STRATEGIES, choose=choose), name=name)
        print(f"{name:<18} {result.adjusted_sharpe:.2f}")

    # Fixed weights, if we'd rather decide ourselves.
    fixed = Portfolio(STRATEGIES, {"fast": 0.5, "mid": 0.5})
    print(f"{'fixed 50/50':<18} {bt.run_positions(fixed).adjusted_sharpe:.2f}")

    portfolio = Portfolio(STRATEGIES, choose=shrunk_max_sharpe())
    print("\nWeights over time\n", portfolio.weights(bt.prices))

    # The submission: positions for every date, from the portfolio we pick.
    (HERE / "output").mkdir(exist_ok=True)
    bt.run_positions(portfolio, period="full").to_csv(
        HERE / "output" / "portfolio_positions.csv"
    )


if __name__ == "__main__":
    main()
