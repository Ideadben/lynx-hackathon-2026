from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from backtest import Backtest
from backtest.choose import equal_weights, max_sharpe, stateless
from backtest.evaluation import DATE
from backtest.leakage import LeakageError
from backtest.pipeline import instruments, load_data, risk_scale, to_positions
from backtest.scoring import Scoring
from portfolio import Portfolio, inverse_vol, risk_parity, shrunk_max_sharpe
from strategies import trend

PRICES_CSV = Path(__file__).resolve().parents[1] / "prices" / "prices_old_example" / "example_prices.csv"
STRATEGIES = {
    "fast": partial(trend.forecast, spans=(8,)),
    "mid": partial(trend.forecast, spans=(32,)),
    "slow": partial(trend.forecast, spans=(128,)),
}


@pytest.fixture(scope="module")
def prices():
    return load_data(PRICES_CSV)


def values(frame):
    return frame.select(instruments(frame)).to_numpy().astype(float)


def test_fixed_weights_blend_then_rescale_risk(prices):
    weights = {"fast": 0.7, "mid": 0.3}
    parts = [
        values(to_positions(STRATEGIES[name](prices), prices)) * w
        for name, w in weights.items()
    ]
    # A strategy that hasn't started yet counts as 0, unless none has started.
    summed = np.where(np.isnan(parts).all(axis=0), np.nan, np.nansum(parts, axis=0))
    blend = pl.DataFrame(summed, schema=instruments(prices), orient="row")
    expected = risk_scale(blend.with_columns(prices[DATE]), prices)
    actual = Portfolio(STRATEGIES, weights)(prices)
    np.testing.assert_allclose(values(actual), values(expected), equal_nan=True)


def test_one_strategy_alone_is_that_strategy_rescaled(prices):
    alone = Portfolio(STRATEGIES, {"fast": 1.0})(prices)
    expected = risk_scale(to_positions(STRATEGIES["fast"](prices), prices), prices)
    np.testing.assert_allclose(values(alone), values(expected), equal_nan=True)


def test_without_weights_or_choose_the_weights_are_equal(prices):
    equal = Portfolio(STRATEGIES)(prices)
    third = Portfolio(STRATEGIES, {name: 1 / 3 for name in STRATEGIES})(prices)
    np.testing.assert_allclose(values(equal), values(third), equal_nan=True)


@pytest.mark.parametrize(
    "choose",
    [
        equal_weights(),
        max_sharpe(),
        inverse_vol(),
        risk_parity(),
        shrunk_max_sharpe(),
    ],
)
def test_portfolio_passes_the_leakage_check(prices, choose):
    result = Backtest(prices).run_positions(Portfolio(STRATEGIES, choose=choose))
    assert np.isfinite(result.adjusted_sharpe)
    assert result.name == "portfolio"


def test_a_leaky_strategy_makes_the_portfolio_fail(prices):
    def leaky(prices):
        return prices.with_columns(pl.col(instruments(prices)).shift(-1))

    with pytest.raises(LeakageError):
        Backtest(prices).run_positions(
            Portfolio({**STRATEGIES, "leaky": leaky}, choose=risk_parity())
        )


def test_weights_change_only_on_refit_days(prices):
    portfolio = Portfolio(STRATEGIES, choose=risk_parity(), refit_every=50)
    weights = portfolio.weights(prices)
    rows = [prices[DATE].to_list().index(d) for d in weights[DATE]]
    assert rows[0] == 0
    assert all(row % 50 == 0 for row in rows)
    np.testing.assert_allclose(weights.drop(DATE).sum_horizontal(), 1.0)
    np.testing.assert_allclose(weights.drop(DATE).row(0), [1 / 3] * 3)


def test_weights_come_from_the_returns_up_to_the_refit_day(prices):
    portfolio = Portfolio(STRATEGIES, choose=max_sharpe(), refit_every=500)
    weights = portfolio.weights(prices)
    day = weights[DATE][2]
    row = prices[DATE].to_list().index(day)

    positions = portfolio.positions(prices)
    returns = pl.DataFrame(
        {
            name: Scoring().figures(frame, prices, ["returns"])["returns"]["returns"]
            for name, frame in positions.items()
        }
    )
    train = returns.head(row + 1).drop_nulls()
    expected, _ = max_sharpe()(train, None)
    np.testing.assert_allclose(weights.drop(DATE).row(2), list(expected.values()))


def test_choose_gets_lookback_days_and_its_state_back(prices):
    seen = []

    def remembering(train, state):
        seen.append((train.height, state))
        return {"fast": 1.0}, len(seen)

    Portfolio(STRATEGIES, choose=remembering, refit_every=200, lookback=300)(prices)
    assert all(height <= 300 for height, _ in seen)
    assert [state for _, state in seen] == [None, *range(1, len(seen))]


def test_weights_stay_equal_until_there_is_enough_history(prices):
    calls = []
    choose = stateless(lambda train: calls.append(train.height) or {"fast": 1.0})
    portfolio = Portfolio(STRATEGIES, choose=choose, refit_every=20, min_train=400)
    weights = portfolio.weights(prices)
    assert min(calls) >= 400
    first_choice = weights[DATE][1]
    assert weights.drop(DATE).row(0) == pytest.approx((1 / 3,) * 3)
    assert weights.filter(pl.col(DATE) == first_choice)["fast"].item() == 1.0


def test_extras_are_passed_to_every_strategy(prices):
    extras = prices.select(DATE, pl.col(instruments(prices)[0]).alias("signal"))
    received = []

    def with_extras(prices, extras):
        received.append(extras.columns)
        return trend.forecast(prices)

    bt = Backtest(prices, extras)
    bt.run_positions(
        Portfolio({"a": with_extras, "b": with_extras}), check_leakage=False
    )
    assert received == [[DATE, "signal"]] * 2


@pytest.mark.parametrize(
    "arguments, message",
    [
        (({},), "at least one strategy"),
        ((STRATEGIES, {"nope": 1.0}), "unknown strategies"),
        ((STRATEGIES, {"fast": float("nan")}), "finite"),
        ((STRATEGIES, {"fast": 1.0}, {"choose": equal_weights()}), "not both"),
        ((STRATEGIES, None, {"refit_every": 0}), "refit_every"),
        ((STRATEGIES, None, {"lookback": 10}), "lookback"),
    ],
)
def test_bad_arguments_get_a_clear_error(arguments, message):
    positional = arguments[:2]
    keywords = arguments[2] if len(arguments) > 2 else {}
    with pytest.raises(ValueError, match=message):
        Portfolio(*positional, **keywords)


def test_choose_returning_unknown_names_is_rejected(prices):
    choose = stateless(lambda train: {"nope": 1.0})
    with pytest.raises(ValueError, match="unknown strategies"):
        Portfolio(STRATEGIES, choose=choose, refit_every=500)(prices)


def correlated_returns(seed=0, days=2000):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=days)
    columns = {
        "a": 0.01 * (common + 0.3 * rng.normal(size=days)),
        "b": 0.01 * (common + 0.3 * rng.normal(size=days)),
        "c": 0.01 * rng.normal(size=days),
    }
    return pl.DataFrame(columns)


def with_drift(train, drifts):
    return train.with_columns(
        pl.col(name) + drift for name, drift in zip(train.columns, drifts, strict=True)
    )


def risk_contributions(train, weights):
    covariance = np.cov(train.to_numpy(), rowvar=False)
    w = np.array(list(weights.values()))
    contributions = w * (covariance @ w)
    return contributions / contributions.sum()


def test_risk_parity_gives_equal_risk_contributions():
    train = correlated_returns()
    weights, _ = risk_parity()(train, None)
    np.testing.assert_allclose(risk_contributions(train, weights), 1 / 3, atol=1e-4)
    # All three are equally volatile, but a and b move together, so c gets more weight.
    assert weights["c"] > weights["a"] and weights["c"] > weights["b"]


def test_risk_parity_without_correlation_is_inverse_vol():
    train = correlated_returns()
    parity, _ = risk_parity(shrink=1.0)(train, None)
    inverse, _ = inverse_vol()(train, None)
    np.testing.assert_allclose(list(parity.values()), list(inverse.values()), atol=1e-5)


def test_shrunk_max_sharpe_without_shrinking_is_max_sharpe():
    train = with_drift(correlated_returns(), [0.001, 0.0005, 0.0002])
    shrunk, _ = shrunk_max_sharpe(shrink=0.0, mean_shrink=0.0)(train, None)
    plain, _ = max_sharpe()(train, None)
    # Two optimisers, so only close: they stop at slightly different points.
    np.testing.assert_allclose(list(shrunk.values()), list(plain.values()), atol=5e-3)


def test_shrinking_means_spreads_the_weights():
    train = with_drift(correlated_returns(), [0.001, 0.0, 0.0])
    sharp, _ = shrunk_max_sharpe(shrink=0.0, mean_shrink=0.0)(train, None)
    steady, _ = shrunk_max_sharpe(shrink=0.5, mean_shrink=0.9)(train, None)
    assert max(steady.values()) < max(sharp.values())


@pytest.mark.parametrize("choose", [inverse_vol(), risk_parity(), shrunk_max_sharpe()])
def test_weighting_functions_are_long_only_and_add_up_to_one(choose):
    weights, _ = choose(correlated_returns(seed=3), None)
    assert all(w >= 0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("make", [risk_parity, shrunk_max_sharpe])
def test_shrink_outside_zero_to_one_is_rejected(make):
    with pytest.raises(ValueError, match="between 0 and 1"):
        make(shrink=1.5)


def test_weighting_functions_work_with_walk_forward(prices):
    walk = Backtest(prices).walk_forward(STRATEGIES, choose=risk_parity(shrink=0.5))
    assert np.isfinite(walk.result.adjusted_sharpe)
    np.testing.assert_allclose(
        walk.choices.drop("first_day", "last_day").sum_horizontal(), 1.0
    )
