"""How sure can we be about a Sharpe ratio from one history?

Run it with: uv run python src/examples/bootstrap_example.py
"""

from pathlib import Path

import numpy as np

from backtest import Backtest, adjusted_sharpe, bootstrap, sharpe
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "prices_old_example" / "example_prices.csv"


def main() -> None:
    returns = Backtest(PRICES).run(trend.forecast).returns

    # The same numbers the backtester reports, from the daily returns.
    print(f"Sharpe:          {sharpe(returns):.2f}")
    print(f"Adjusted Sharpe: {adjusted_sharpe(returns):.2f}")

    # Resample the history in blocks, 10,000 times.
    result = bootstrap(returns)
    print(f"\nAdjusted Sharpe: {result}")

    # Any statistic works: here, the worst year out of every resampled history.
    def worst_year(values: np.ndarray) -> float:
        years = values[: len(values) // 252 * 252].reshape(-1, 252)
        return float(sharpe(years.T).min())

    print(f"Worst year's Sharpe: {bootstrap(returns, worst_year, samples=2000)}")

    # Pure noise should give an interval around zero.
    noise = np.random.default_rng(0).normal(0, 1, len(returns))
    print(f"\nNoise: {bootstrap(noise)}")


if __name__ == "__main__":
    main()
