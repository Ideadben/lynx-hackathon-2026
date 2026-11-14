"""Tests choosing between rules using only past data.

Run it with: uv run python src/examples/walk_forward_example.py
"""

from pathlib import Path

import polars as pl

from backtest import Backtest, bootstrap, grid, paired_bootstrap
from backtest.choose import (
    best_single,
    drop_small,
    equal_weights,
    max_sharpe,
    random_search,
    stateless,
)
from strategies import trend

HERE = Path(__file__).resolve().parent
PRICES = HERE.parents[1] / "prices" / "prices_old_example" / "example_prices.csv"
OUTPUT = HERE / "output"


def single_speed(prices: pl.DataFrame, span: int) -> pl.DataFrame:
    """Trend at one speed."""
    return trend.forecast(prices, spans=(span,))


def favour_recent(train: pl.DataFrame) -> dict[str, float]:
    """A custom choosing rule: weight each rule by its Sharpe over the last year, if positive."""
    last_year = train.tail(252)
    scores = (last_year.mean() / last_year.std()).row(0, named=True)
    positive = {name: max(score, 0.0) for name, score in scores.items()}
    total = sum(positive.values()) or 1.0
    return {name: score / total for name, score in positive.items()}


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    bt = Backtest(PRICES)

    # Five rules: trend at five speeds. Use a wide range, because picking the range after
    # seeing results brings hindsight back.
    rules = grid(single_speed, span=[4, 8, 16, 32, 64])

    # Each choosing function is tested on the same blocks.
    choosers = {
        "equal weights": equal_weights(),
        "best single": best_single(),
        "max sharpe": max_sharpe(),
        "random search": drop_small(random_search(candidates=500, keep=50), below=0.05),
        "favour recent": stateless(favour_recent),
    }
    walks = {name: bt.walk_forward(rules, choose=c) for name, c in choosers.items()}

    for name, walk in walks.items():
        print(f"{name:<15}{walk.result.adjusted_sharpe:6.2f}")

    # Each rule on its own over the same days. The best row is what hindsight would pick.
    baseline = walks["equal weights"]
    print(baseline.fixed.select("rule", "adjusted_sharpe", "av_holding"))

    # The weights the random search chose before each block.
    print(walks["random search"].choices)

    # How uncertain is the score? And does learning weights beat equal weights?
    print(f"\nEqual weights: {bootstrap(baseline.result.returns)}")
    for name in ("best single", "max sharpe", "random search", "favour recent"):
        gap = paired_bootstrap(walks[name].result.returns, baseline.result.returns)
        print(f"{name} minus equal weights: {gap}")

    walks["random search"].result.plot().savefig(OUTPUT / "walk_forward_dashboard.png")
    print(f"\nSaved the dashboard to {OUTPUT}")


if __name__ == "__main__":
    main()
