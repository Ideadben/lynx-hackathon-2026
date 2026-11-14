"""Tests for the final strategies (tikhonov, triple) and the final portfolio."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from backtest import Backtest
from backtest.pipeline import load_data
from portfolio import final
from strategies import tikhonov, triple

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "prices" / "prices_old_example" / "example_prices.csv"
PRICES = ROOT / "prices" / "close_prices_insample.csv"
MACRO = ROOT / "prices" / "macro_data_insample.csv"


@pytest.fixture(scope="module")
def example():
    return load_data(EXAMPLE)


def test_tikhonov_passes_the_leakage_check(example):
    result = Backtest(example).run(tikhonov.forecast)  # raises LeakageError if it peeks
    assert np.isfinite(result.adjusted_sharpe)


def test_tikhonov_weights_for_a_year_only_use_earlier_years(example):
    full = tikhonov.weights_by_year(example)
    year = sorted(full)[2]
    cut = example.filter(pl.col("AsOfDate").dt.year() <= year)  # drop every later year
    np.testing.assert_allclose(tikhonov.weights_by_year(cut)[year], full[year])


def test_tikhonov_starts_after_min_years(example):
    fc = tikhonov.forecast(example)
    first = fc.drop_nulls()["AsOfDate"][0]
    assert first.year == example["AsOfDate"][0].year + tikhonov.MIN_YEARS


def test_smooth_gamma_penalises_curvature_not_straight_lines():
    G = tikhonov.smooth_gamma(10)
    line = np.arange(10.0)
    bumpy = np.array([0, 1] * 5, dtype=float)
    assert np.sum((G @ line) ** 2) < np.sum((G @ bumpy) ** 2)


def test_long_stocks_bonds_is_one_on_stocks_and_bonds_only():
    prices = load_data(PRICES).head(5)
    fc = triple.long_stocks_bonds(prices)
    for c in fc.columns[1:]:
        expected = 1.0 if c.endswith(("_stocks", "_bonds")) else 0.0
        assert fc[c].to_list() == [expected] * 5


def test_triple_with_macro_passes_the_leakage_check():
    result = Backtest(PRICES, MACRO).run_positions(triple.portfolio())
    assert np.isfinite(result.adjusted_sharpe)


def test_final_weights_cover_every_strategy_and_add_up_to_one():
    assert set(final.WEIGHTS) == set(final.STRATEGIES)
    assert sum(final.WEIGHTS.values()) == pytest.approx(1.0)
