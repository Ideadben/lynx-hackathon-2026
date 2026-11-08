from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backtest import (
    Backtest,
    adjusted_sharpe,
    bootstrap,
    grid,
    paired_bootstrap,
    sharpe,
)
from backtest.choose import (
    best_single,
    drop_small,
    equal_weights,
    max_sharpe,
    random_search,
    stateless,
)
from backtest.evaluation import DATE, calc_key_figures
from backtest.leakage import LeakageError, check_no_leakage
from backtest.pipeline import (
    instruments,
    load_data,
    risk_scale,
    to_positions,
    vol_scale,
)
from backtest.runner import dev_end
from backtest.scoring import Scoring
from strategies import trend

PRICES_CSV = Path(__file__).resolve().parents[1] / "prices" / "example_prices.csv"


@pytest.fixture(scope="module")
def prices():
    return load_data(PRICES_CSV)


def ewma_gap(prices, span=16):
    columns = instruments(prices)
    fast = prices.select(
        pl.col(columns).ewm_mean(span=span, adjust=False, min_samples=span)
    )
    slow = prices.select(
        pl.col(columns).ewm_mean(span=4 * span, adjust=False, min_samples=4 * span)
    )
    return (fast - slow).with_columns(prices[DATE])


def ewma_sign(prices):
    columns = instruments(prices)
    return ewma_gap(prices).with_columns(pl.col(columns).sign())


def benjamin_positions(M, prices, N=100):
    """calc_positions from BENJAMIN/ewma.ipynb, copied unchanged."""
    columns = instruments(prices)
    returns = prices.select(pl.col(columns).diff())
    ewma_fast = prices.select(
        pl.col(columns).ewm_mean(span=M, adjust=False, min_samples=M)
    )
    ewma_slow = prices.select(
        pl.col(columns).ewm_mean(span=4 * M, adjust=False, min_samples=4 * M)
    )
    X = (ewma_fast - ewma_slow).select(pl.all().sign())
    sigma = returns.select(pl.col(columns).pow(2).rolling_mean(N).sqrt())
    X_tilde = X / sigma
    asset_returns = X_tilde.select(pl.all().shift(2)) * returns
    R = asset_returns.select(
        pl.when(pl.all_horizontal(pl.all().is_null()))
        .then(None)
        .otherwise(pl.sum_horizontal(pl.all()))
        .alias("R")
    )["R"]
    X_hat = X_tilde.select(pl.all() / R.rolling_std(N)).fill_nan(None)
    return X_hat.with_columns(prices[DATE]).select(DATE, *columns)


def test_pipeline_matches_benjamins_ewma(prices):
    columns = instruments(prices)
    ours = to_positions(ewma_sign(prices), prices).select(columns).to_numpy()
    his = benjamin_positions(16, prices).select(columns).to_numpy()
    assert np.isfinite(ours).sum() > 5000 * len(columns)
    np.testing.assert_allclose(ours, his, rtol=1e-9, equal_nan=True)


def test_scores_match_benjamins_recorded_results(prices):
    """Figures from BENJAMIN/lstm/results/summary.csv. His dev score starts at day 1300."""
    result = Backtest(prices, scoring="lynx").run(ewma_sign, period="holdout")
    assert result.figures["sharpe"] == pytest.approx(0.8681230181697263)

    dev = result.positions.filter(pl.col(DATE) <= dev_end(prices))[1300:]
    assert calc_key_figures(dev, prices)["sharpe"] == pytest.approx(0.9576362196831378)


def test_forecast_is_capped_and_scaled(prices):
    columns = instruments(prices)
    forecast = ewma_gap(prices)
    capped = vol_scale(forecast, prices, cap=20.0)
    sigma = prices.select(pl.col(columns).diff().pow(2).rolling_mean(100).sqrt())
    implied = (capped.select(columns) * sigma).to_numpy()
    assert np.nanmax(np.abs(implied)) <= 20.0 + 1e-9
    assert 5 < np.nanmean(np.abs(implied)) < 15


def test_risk_scale_steadies_strategy_risk(prices):
    columns = instruments(prices)
    positions = to_positions(ewma_gap(prices), prices)
    returns = prices.select(pl.col(columns).diff())
    daily = (
        (positions.select(pl.col(columns).shift(2)) * returns)
        .select(pl.sum_horizontal(pl.all()))
        .to_series()
    )
    assert daily.tail(4000).std() == pytest.approx(1.0, abs=0.15)


def test_backtest_reports_every_period(prices):
    results = {
        p: Backtest(prices).run(ewma_sign, period=p) for p in ("dev", "holdout", "full")
    }
    assert results["dev"].figures["sharpe"] != results["holdout"].figures["sharpe"]
    assert results["dev"].figures["sharpe"] > 0.5
    assert "adjusted_sharpe" in str(results["dev"])
    assert results["dev"].plot() is not None


def test_score_matches_scoring_the_filtered_positions_by_hand(prices):
    result = Backtest(prices, scoring="lynx").run(ewma_sign, period="dev")
    by_hand = calc_key_figures(
        result.positions.filter(pl.col(DATE) <= dev_end(prices)), prices
    )
    assert result.figures["adjusted_sharpe"] == pytest.approx(
        by_hand["adjusted_sharpe"]
    )


def test_unknown_period_is_rejected(prices):
    with pytest.raises(ValueError, match="Unknown period"):
        Backtest(prices).run(ewma_sign, period="test", check_leakage=False)


def test_clean_strategies_pass_the_leakage_check(prices):
    check_no_leakage(lambda p: to_positions(ewma_gap(p), p), prices)


def peeks_one_day(prices):
    return ewma_gap(prices).with_columns(pl.col(instruments(prices)).shift(-1))


def uses_full_sample_mean(prices):
    columns = instruments(prices)
    return prices.with_columns((pl.col(c) - pl.col(c).mean()) for c in columns)


def uses_last_price(prices):
    columns = instruments(prices)
    return prices.with_columns((pl.col(c) - pl.col(c).last()) for c in columns)


@pytest.mark.parametrize(
    "leaky", [peeks_one_day, uses_full_sample_mean, uses_last_price]
)
def test_leaky_strategies_are_caught(prices, leaky):
    with pytest.raises(LeakageError):
        Backtest(prices).run(leaky)


def test_future_extras_are_caught(prices):
    extras = prices.select(DATE, pl.col(instruments(prices)[0]).alias("macro"))

    def peeks_at_macro(prices, extras):
        macro = extras["macro"].shift(-1)
        return ewma_gap(prices).with_columns(
            pl.col(c) * macro for c in instruments(prices)
        )

    def uses_past_macro(prices, extras):
        macro = extras["macro"].diff().sign()
        return ewma_gap(prices).with_columns(
            pl.col(c) * macro for c in instruments(prices)
        )

    with pytest.raises(LeakageError):
        Backtest(prices, extras).run(peeks_at_macro)
    Backtest(prices, extras).run(uses_past_macro)


def test_bad_forecast_frames_get_a_clear_error(prices):
    with pytest.raises(ValueError, match="no column"):
        to_positions(ewma_gap(prices).drop(instruments(prices)[0]), prices)
    with pytest.raises(ValueError, match="one row per price date"):
        to_positions(ewma_gap(prices).tail(100), prices)


def test_missing_prices_are_forward_filled(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("dates,a\n2000-01-03,1.0\n2000-01-04,\n2000-01-05,3.0\n")
    assert load_data(path)["a"].to_list() == [1.0, 1.0, 3.0]


def test_zero_volatility_gives_nulls_not_infinity():
    dates = pl.date_range(pl.date(2000, 1, 1), pl.date(2000, 12, 31), eager=True).alias(
        DATE
    )
    prices = pl.DataFrame({DATE: dates, "flat": [100.0] * dates.len()})
    positions = vol_scale(prices.with_columns(flat=pl.lit(1.0)), prices)
    assert positions["flat"].null_count() == dates.len()
    assert risk_scale(positions, prices)["flat"].null_count() == dates.len()


def test_result_exposes_key_figures_as_attributes(prices):
    result = Backtest(prices).run(ewma_sign)
    assert result.sharpe == result.figures["sharpe"]
    assert result.adjusted_sharpe == result.figures["adjusted_sharpe"]
    with pytest.raises(AttributeError, match="Key figures"):
        _ = result.not_a_figure


def test_one_instance_runs_many_strategies_and_sources_stay_separate(prices):
    bt = Backtest(prices)
    assert bt.run(ewma_sign).sharpe != bt.run(ewma_gap).sharpe
    half = Backtest(prices.head(3000))
    assert half.run(ewma_sign).sharpe != bt.run(ewma_sign).sharpe


def test_every_data_source_loads_the_same(prices):
    pandas_frame = prices.to_pandas()
    sources = [
        str(PRICES_CSV),
        PRICES_CSV,
        prices,
        pandas_frame,
        pandas_frame.set_index(DATE),
        pandas_frame.set_index(DATE).rename_axis(None),
        prices.rename({DATE: "date"}),
        prices.with_columns(pl.col(DATE).cast(pl.Datetime)),
        prices.with_columns(pl.col(DATE).cast(pl.String)),
    ]
    for source in sources:
        assert load_data(source).equals(prices)


def test_integer_prices_become_floats():
    frame = pl.DataFrame({"date": ["2000-01-03", "2000-01-04"], "a": [1, 2]})
    assert load_data(frame).schema["a"] == pl.Float64


def test_unusable_data_gets_a_clear_error():
    with pytest.raises(ValueError, match="one date column"):
        load_data(pl.DataFrame({"timestamp": ["2000-01-03"], "a": [1.0]}))
    with pytest.raises(ValueError, match="one date column"):
        load_data(
            pl.DataFrame({"date": ["2000-01-03"], "dates": ["2000-01-03"], "a": [1.0]})
        )
    with pytest.raises(ValueError, match="must hold dates"):
        load_data(pd.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="numeric"):
        load_data(pl.DataFrame({"date": ["2000-01-03"], "a": ["high"]}))
    with pytest.raises(TypeError, match="csv path"):
        load_data(123)


def test_sign_forecast_gives_the_same_positions_with_or_without_normalizing(prices):
    columns = instruments(prices)
    scaled = to_positions(ewma_sign(prices), prices).select(columns).to_numpy()
    raw = to_positions(ewma_sign(prices), prices, normalize=False)
    np.testing.assert_allclose(
        raw.select(columns).to_numpy(), scaled, rtol=1e-9, equal_nan=True
    )


def test_without_normalizing_size_differences_between_assets_are_kept(prices):
    first, second = instruments(prices)[:2]
    forecast = prices.with_columns(pl.lit(1.0).alias(first), pl.lit(2.0).alias(second))
    volatility = prices.select(
        pl.col(first, second).diff().pow(2).rolling_mean(100).sqrt()
    ).row(500, named=True)

    def size_ratio(**options):
        positions = vol_scale(forecast, prices, **options).row(500, named=True)
        return (positions[second] * volatility[second]) / (
            positions[first] * volatility[first]
        )

    assert size_ratio() == pytest.approx(1.0)
    assert size_ratio(normalize=False) == pytest.approx(2.0)


def test_no_cap_keeps_extreme_forecasts(prices):
    columns = instruments(prices)
    volatility = prices.select(
        pl.col(columns).diff().pow(2).rolling_mean(100).sqrt()
    ).to_numpy()
    forecast = ewma_gap(prices)
    capped = vol_scale(forecast, prices).select(columns).to_numpy() * volatility
    uncapped = vol_scale(forecast, prices, cap=None).select(columns).to_numpy()
    assert np.nanmax(np.abs(capped)) <= 20.0 + 1e-9
    assert np.nanmax(np.abs(uncapped * volatility)) > 20.0


def test_options_pass_through_backtest_and_stay_leak_free(prices):
    default = Backtest(prices).run(ewma_gap)
    assert Backtest(prices, cap=None).run(ewma_gap).sharpe != default.sharpe
    assert Backtest(prices, normalize=False).run(ewma_gap).sharpe != default.sharpe


def test_returns_reproduce_the_reported_sharpe(prices):
    result = Backtest(prices).run(ewma_sign)
    returns = result.returns["returns"]
    assert returns.null_count() == 0
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    assert sharpe == pytest.approx(result.sharpe)


def test_returns_cover_only_the_period(prices):
    bt = Backtest(prices)
    assert bt.run(ewma_sign).returns[DATE].max() <= dev_end(prices)
    assert bt.run(ewma_sign, period="holdout").returns[DATE].min() > dev_end(prices)


def test_by_year_adds_up(prices):
    result = Backtest(prices).run(ewma_sign)
    years = result.by_year()
    assert years["days"].sum() == result.returns.height
    assert years["total_return"].sum() == pytest.approx(result.returns["returns"].sum())
    assert years["year"].to_list() == sorted(years["year"].to_list())

    one_year = result.returns.filter(pl.col(DATE).dt.year() == 2010)["returns"]
    expected = one_year.mean() / one_year.std() * np.sqrt(252)
    assert years.filter(year=2010)["sharpe"].item() == pytest.approx(expected)


def test_by_instrument_has_one_row_per_asset(prices):
    table = Backtest(prices).run(ewma_sign).by_instrument()
    assert table["instrument"].to_list() == instruments(prices)
    assert table.columns == [
        "instrument",
        "sharpe_ex_slippage",
        "av_holding",
        "position_bias",
    ]
    assert table.null_count().sum_horizontal().item() == 0


def test_to_csv_round_trips_and_rescores_the_same(prices, tmp_path):
    bt = Backtest(prices)
    result = bt.run(ewma_sign, period="holdout")
    path = tmp_path / "positions.csv"
    result.to_csv(path)

    saved = load_data(path)
    assert saved.height == prices.height
    rescored = bt.run_positions(saved, period="holdout")
    assert rescored.sharpe == pytest.approx(result.sharpe)


def test_run_positions_scores_positions_unchanged(prices):
    bt = Backtest(prices)
    result = bt.run(ewma_gap)

    from_frame = bt.run_positions(result.positions)
    assert from_frame.positions.equals(result.positions)
    assert from_frame.figures == result.figures
    assert from_frame.name == "positions"

    def blend(prices):
        return to_positions(ewma_gap(prices), prices)

    from_function = bt.run_positions(blend)
    assert from_function.figures == result.figures
    assert from_function.name == "blend"


def test_run_positions_checks_functions_for_leakage(prices):
    def leaky(prices):
        return to_positions(peeks_one_day(prices), prices)

    with pytest.raises(LeakageError):
        Backtest(prices).run_positions(leaky)


def test_run_positions_passes_extras_and_checks_the_frame(prices):
    extras = prices.select(DATE, macro=pl.col(instruments(prices)[0]))

    def with_extras(prices, extras):
        return to_positions(ewma_gap(prices), prices)

    Backtest(prices, extras).run_positions(with_extras)
    with pytest.raises(ValueError, match="one row per price date"):
        Backtest(prices).run_positions(ewma_gap(prices).tail(10))


def test_compare_lines_up_with_individual_runs(prices):
    bt = Backtest(prices)
    comparison = bt.compare([ewma_sign, ewma_gap])

    assert comparison.figures["strategy"].to_list() == ["ewma_sign", "ewma_gap"]
    assert comparison.figures["sharpe"][0] == pytest.approx(bt.run(ewma_sign).sharpe)
    assert comparison.results["ewma_gap"].sharpe == bt.run(ewma_gap).sharpe

    table = comparison.correlations
    assert table["strategy"].to_list() == ["ewma_sign", "ewma_gap"]
    assert table["ewma_sign"][0] == pytest.approx(1.0)
    assert table["ewma_sign"][1] == pytest.approx(table["ewma_gap"][0])
    assert 0.5 < table["ewma_sign"][1] < 1.0
    assert "Correlation" in str(comparison)


def test_compare_correlation_matches_numpy(prices):
    comparison = Backtest(prices).compare([ewma_sign, ewma_gap])
    a = comparison.results["ewma_sign"].returns
    b = comparison.results["ewma_gap"].returns
    joined = a.join(b, on=DATE, suffix="_b")
    expected = np.corrcoef(joined["returns"], joined["returns_b"])[0, 1]
    assert comparison.correlations["ewma_sign"][1] == pytest.approx(expected)


def test_compare_needs_unique_names(prices):
    bt = Backtest(prices)
    fast, slow = partial(ewma_gap, span=8), partial(ewma_gap, span=32)
    with pytest.raises(ValueError, match="share a name"):
        bt.compare([fast, slow])
    with pytest.raises(ValueError, match="no strategies"):
        bt.compare([])
    named = bt.compare({"fast": fast, "slow": slow})
    assert list(named.results) == ["fast", "slow"]
    assert named.results["fast"].name == "fast"


def test_compare_catches_leaks(prices):
    with pytest.raises(LeakageError):
        Backtest(prices).compare([ewma_sign, peeks_one_day])


def test_sweep_tries_every_combination(prices):
    bt = Backtest(prices)

    def strategy(prices, span, flip):
        return ewma_gap(prices, span).with_columns(pl.col(instruments(prices)) * flip)

    table = bt.sweep(strategy, span=[8, 16], flip=[1, -1])
    assert table.select("span", "flip").rows() == [(8, 1), (8, -1), (16, 1), (16, -1)]
    assert table.columns[2] == "adjusted_sharpe"

    expected = bt.run(partial(strategy, span=16, flip=-1)).sharpe
    assert table.filter(span=16, flip=-1)["sharpe"].item() == pytest.approx(expected)


def test_sweep_needs_parameters_and_catches_leaks(prices):
    bt = Backtest(prices)
    with pytest.raises(ValueError, match="at least one parameter"):
        bt.sweep(ewma_gap)

    def leaky(prices, span):
        return ewma_gap(prices, span).with_columns(
            pl.col(instruments(prices)).shift(-1)
        )

    with pytest.raises(LeakageError):
        bt.sweep(leaky, span=[16])


def test_csv_with_an_asset_that_starts_late_stays_numeric(tmp_path):
    path = tmp_path / "prices.csv"
    rows = [f"2000-01-{day:02d},1.0," for day in range(1, 29)] * 10
    path.write_text("dates,early,late\n" + "\n".join(rows) + "\n2001-01-01,1.0,5.0\n")
    loaded = load_data(path)
    assert loaded.schema["late"] == pl.Float64
    assert loaded["late"].null_count() == loaded.height - 1


def test_pdf_returns_follow_equation_one_by_hand():
    rng = np.random.default_rng(2)
    days = 20
    dates = pl.date_range(pl.date(2000, 1, 3), pl.date(2000, 1, 22), eager=True)
    p = 100 + rng.normal(0, 1, days).cumsum()
    x = rng.normal(0, 2, days)
    prices = pl.DataFrame({DATE: dates, "a": p})
    positions = pl.DataFrame({DATE: dates, "a": x})
    delta = 0.5

    result = Backtest(prices, delta=delta).run_positions(positions, period="full")

    expected = [
        x[t - 2] * (p[t] - p[t - 1]) - delta * abs(x[t - 2] - x[t - 3])
        for t in range(3, days)
    ]
    np.testing.assert_allclose(result.returns["returns"].to_numpy(), expected)


def test_pdf_scoring_without_costs_agrees_with_lynx_apart_from_days_per_year(prices):
    positions = Backtest(prices).run(ewma_sign).positions
    pdf = Backtest(prices, delta=0.0).run_positions(positions)
    lynx = Backtest(prices, scoring="lynx", costs=0.0).run_positions(positions)
    assert pdf.sharpe == pytest.approx(lynx.sharpe * np.sqrt(252 / 260))
    assert pdf.skewness == pytest.approx(lynx.skewness)
    assert pdf.kurtosis == pytest.approx(lynx.kurtosis)
    assert pdf.position_bias == pytest.approx(lynx.position_bias)


def test_adjusted_sharpe_follows_equation_five(prices):
    result = Backtest(prices).run(ewma_gap)
    s = result.sharpe
    excess = result.kurtosis - 3
    expected = s * (1 + s / 6 * result.skewness - s**2 / 24 * excess)
    assert result.adjusted_sharpe == pytest.approx(expected)


def test_pdf_holding_period_follows_equation_six(prices):
    result = Backtest(prices).run(ewma_gap, period="full")
    x = result.positions.select(instruments(prices)).shift(2).to_numpy()
    expected = 2 * np.nansum(np.abs(x)) / np.nansum(np.abs(np.diff(x, axis=0)))
    assert result.av_holding == pytest.approx(expected)


def test_small_pdf_costs_still_lower_the_sharpe(prices):
    bt = Backtest(prices)
    for strategy in (ewma_sign, ewma_gap, trend.forecast):
        result = bt.run(strategy)
        assert result.sharpe < result.sharpe_ex_slippage


def test_costs_lower_the_sharpe_in_both_scorings(prices):
    for bt in (Backtest(prices, delta=0.05), Backtest(prices, scoring="lynx")):
        result = bt.run(ewma_sign)
        assert result.sharpe < result.sharpe_ex_slippage


def test_every_report_works_in_both_scorings(prices):
    for kind in ("pdf", "lynx"):
        result = Backtest(prices, scoring=kind).run(ewma_sign)
        assert result.plot() is not None
        assert result.by_instrument().height == len(instruments(prices))
        assert result.by_year()["sharpe"].is_finite().all()


def test_unknown_scoring_is_rejected():
    with pytest.raises(ValueError, match="Unknown scoring"):
        Scoring("other")


def messy_prices(start_row=1500, rows=3000):
    """Eight assets of made-up prices: one starts late, one has gaps, one starts in the holdout."""
    rng = np.random.default_rng(1)
    dates = pl.date_range(pl.date(2000, 1, 3), pl.date(2060, 1, 1), "1d", eager=True)
    dates = dates.filter(dates.dt.weekday() <= 5).head(rows).alias(DATE)
    walks = 1000 + rng.normal(0, 5, (rows, 8)).cumsum(axis=0)
    data = {f"asset{i}": walks[:, i].copy() for i in range(8)}
    data["asset6"][:start_row] = np.nan
    data["asset7"][: int(rows * 0.9)] = np.nan
    data["asset5"][rng.integers(0, rows, 100)] = np.nan
    return pl.DataFrame({DATE: dates, **data})


def test_messy_data_runs_through_the_whole_backtest():
    frame = messy_prices()
    bt = Backtest(frame)
    prices = bt.prices
    for period in ("dev", "holdout", "full"):
        result = bt.run(trend.forecast, period=period)
        assert np.isfinite(list(result.figures.values())).all()
        assert result.by_instrument().height == 8
        assert result.plot() is not None
        assert result.returns["returns"].is_finite().all()

    positions = bt.run(trend.forecast).positions
    values = positions.select(instruments(prices)).to_numpy()
    assert not np.isinf(values).any()

    late = positions["asset6"]
    assert late.head(1500).null_count() == 1500
    assert late.tail(1000).null_count() < 1000 and late.tail(1000).null_count() == 0
    assert positions["asset0"].tail(1000).null_count() == 0


def test_an_asset_with_no_data_in_the_period_is_reported_as_nan():
    bt = Backtest(messy_prices())
    table = bt.run(trend.forecast, period="dev").by_instrument()
    assert table.filter(instrument="asset7")["sharpe_ex_slippage"].is_nan().all()
    assert table.filter(instrument="asset0")["sharpe_ex_slippage"].is_finite().all()


def test_trend_strategy_is_clean_and_within_range(prices):
    forecast = trend.forecast(prices).select(instruments(prices)).to_numpy()
    assert np.nanmax(np.abs(forecast)) <= 1.0
    assert set(np.unique(forecast[~np.isnan(forecast)])) <= {-1.0, -1 / 3, 1 / 3, 1.0}
    assert np.isnan(forecast[:126]).all()
    assert not np.isnan(forecast[127:]).any()
    Backtest(prices).run(trend.forecast)


def test_trend_with_one_span_matches_the_single_moving_average(prices):
    columns = instruments(prices)
    ours = trend.forecast(prices, spans=(16,)).select(columns).to_numpy()
    theirs = ewma_sign(prices).select(columns).to_numpy()
    np.testing.assert_array_equal(ours, theirs)


def single_speed(prices, span):
    return trend.forecast(prices, spans=(span,))


def regime_prices(rows=2000):
    """One asset that rises through the first 40% of its history, then falls."""
    turn = int(rows * 0.4)
    rng = np.random.default_rng(3)
    drift = np.where(np.arange(rows) < turn, 0.3, -0.3)
    dates = pl.date_range(pl.date(2000, 1, 3), pl.date(2040, 1, 1), eager=True)
    return pl.DataFrame(
        {DATE: dates.head(rows), "a": 2000 + (drift + rng.normal(0, 1, rows)).cumsum()}
    )


def constant_bet(prices, direction):
    return prices.with_columns(a=pl.lit(float(direction)))


def picked(walk):
    """The rule with the largest weight in each block."""
    weights = walk.choices.drop("first_day", "last_day")
    return [max(row, key=row.get) for row in weights.iter_rows(named=True)]


SPANS = grid(single_speed, span=[8, 16, 32])


def test_grid_names_and_builds_rules(prices):
    rules = grid(single_speed, span=[8, 16])
    assert list(rules) == ["span=8", "span=16"]
    two = grid(constant_bet, direction=[1, -1])
    assert two["direction=-1"](prices)["a"][0] == -1.0
    assert list(grid(ewma_gap, span=[8], other=[1])) == ["span=8, other=1"]
    with pytest.raises(ValueError, match="at least one parameter"):
        grid(single_speed)


def test_walk_forward_cannot_see_a_regime_change_coming():
    bt = Backtest(regime_prices())
    walk = bt.walk_forward(grid(constant_bet, direction=[1, -1]), period="full")
    picks = picked(walk)

    assert picks[0] == "direction=1"
    assert picks[-1] == "direction=-1"
    assert picks.index("direction=-1") > 0
    assert "direction=1" not in picks[picks.index("direction=-1") :]
    assert walk.result.adjusted_sharpe < walk.fixed["adjusted_sharpe"].max()


def test_walk_forward_stitches_the_weighted_positions(prices):
    bt = Backtest(prices)
    walk = bt.walk_forward(SPANS, choose=equal_weights(), blocks=4)
    columns = instruments(prices)
    each = [
        bt.run(rule).positions.select(columns).to_numpy() for rule in SPANS.values()
    ]
    average = np.mean(each, axis=0)

    first = walk.choices["first_day"][0]
    last = walk.choices["last_day"][-1]
    window = pl.col(DATE).is_between(first, last)
    rows = prices.with_row_index().filter(window)["index"].to_numpy()
    stitched = walk.result.positions.filter(window).select(columns).to_numpy()
    np.testing.assert_allclose(stitched, average[rows])

    before = walk.result.positions.filter(pl.col(DATE) < first)
    assert before.drop(DATE).null_count().sum_horizontal().item() == before.height * 5


def test_walk_forward_best_single_uses_one_rule_per_block(prices):
    bt = Backtest(prices)
    walk = bt.walk_forward(SPANS, blocks=4)
    for row in walk.choices.iter_rows(named=True):
        weights = [row[name] for name in SPANS]
        assert sorted(weights) == [0.0, 0.0, 1.0]
        chosen = bt.run(SPANS[max(SPANS, key=row.get)]).positions
        window = pl.col(DATE).is_between(row["first_day"], row["last_day"])
        np.testing.assert_allclose(
            walk.result.positions.filter(window).drop(DATE).to_numpy(),
            chosen.filter(window).drop(DATE).to_numpy(),
        )


def test_walk_forward_only_uses_the_past(prices):
    walk = Backtest(prices).walk_forward(SPANS, choose=max_sharpe(), blocks=6)
    cut = walk.choices["first_day"][3]
    future = pl.col(DATE) >= cut
    columns = instruments(prices)
    rng = np.random.default_rng(4)
    noise = (rng.normal(0, 50, (prices.height, len(columns)))).cumsum(axis=0)
    changed = prices.with_columns(
        pl.when(future)
        .then(pl.col(c) + pl.Series(noise[:, i]))
        .otherwise(pl.col(c))
        .alias(c)
        for i, c in enumerate(columns)
    )
    other = Backtest(changed).walk_forward(SPANS, choose=max_sharpe(), blocks=6)

    assert walk.choices.head(4).equals(other.choices.head(4))
    earlier = pl.col(DATE) < cut
    assert walk.result.positions.filter(earlier).equals(
        other.result.positions.filter(earlier)
    )


def test_walk_forward_passes_training_returns_and_threads_state(prices):
    seen = []

    def recording(train, state):
        seen.append((train, state))
        return {name: 1 / train.width for name in train.columns}, (state or 0) + 1

    bt = Backtest(prices)
    walk = bt.walk_forward(SPANS, choose=recording, blocks=3)
    assert [state for _, state in seen] == [None, 1, 2]

    for (train, _), first_day in zip(seen, walk.choices["first_day"], strict=True):
        assert train.columns == list(SPANS)
        assert train.null_count().sum_horizontal().item() == 0
        alone = bt.run(SPANS["span=16"], period="full").returns
        known = alone.filter(pl.col(DATE) < first_day)
        assert train.height <= known.height
        assert train["span=16"].tail(5).to_list() == known["returns"].tail(5).to_list()
    assert seen[1][0].height > seen[0][0].height


def test_walk_forward_rejects_bad_weights(prices):
    bt = Backtest(prices)
    with pytest.raises(ValueError, match="unknown rules"):
        bt.walk_forward(SPANS, choose=stateless(lambda train: {"nope": 1.0}))
    with pytest.raises(ValueError, match="finite"):
        bt.walk_forward(SPANS, choose=stateless(lambda train: {"span=8": np.nan}))


def test_walk_forward_stays_inside_the_period(prices):
    bt = Backtest(prices)
    walk = bt.walk_forward(SPANS)
    assert walk.choices["last_day"].max() == dev_end(prices)
    assert walk.result.period == "dev"
    assert walk.result.returns[DATE].max() <= dev_end(prices)
    full = bt.walk_forward(SPANS, period="full")
    assert full.choices["last_day"].max() == prices[DATE].max()


def test_walk_forward_fixed_table_covers_the_same_days(prices):
    walk = Backtest(prices).walk_forward(grid(single_speed, span=[16]), blocks=4)
    assert walk.fixed["rule"].to_list() == ["span=16"]
    assert walk.fixed["sharpe"][0] == pytest.approx(walk.result.sharpe)


def test_walk_forward_catches_leaks(prices):
    def leaky(prices):
        return ewma_gap(prices).with_columns(pl.col(instruments(prices)).shift(-1))

    with pytest.raises(LeakageError):
        Backtest(prices).walk_forward({"leaky": leaky})


@pytest.mark.parametrize(
    "options, message",
    [
        ({"period": "holdout"}, "period"),
        ({"first_train": 1.5}, "first_train"),
        ({"blocks": 0}, "blocks"),
        ({"blocks": 100000}, "blocks"),
    ],
)
def test_walk_forward_rejects_bad_arguments(prices, options, message):
    with pytest.raises(ValueError, match=message):
        Backtest(prices).walk_forward(SPANS, **options)


def test_walk_forward_needs_rules(prices):
    with pytest.raises(ValueError, match="at least one rule"):
        Backtest(prices).walk_forward({})


@pytest.mark.parametrize("rows", [1500, 3000, 6000, 9000])
@pytest.mark.parametrize("years_later", [0, 17])
def test_walk_forward_adapts_to_any_length_and_start_date(rows, years_later):
    frame = messy_prices(start_row=rows // 2, rows=rows).with_columns(
        pl.col(DATE).dt.offset_by(f"{years_later}y")
    )
    rules = grid(single_speed, span=[8, 16])
    walk = Backtest(frame).walk_forward(
        rules, choose=random_search(candidates=50, keep=5)
    )
    assert walk.choices.height == 12
    assert walk.choices["first_day"].is_sorted()
    assert np.isfinite(walk.result.sharpe)


def test_walk_forward_explains_when_history_is_too_short():
    frame = messy_prices(start_row=100, rows=500)
    with pytest.raises(ValueError, match="training days"):
        Backtest(frame).walk_forward(grid(single_speed, span=[8, 16]))


def training_frame(rows=600, seed=5):
    rng = np.random.default_rng(seed)
    good = rng.normal(0.1, 1, rows)
    return pl.DataFrame(
        {
            "good": good,
            "bad": rng.normal(-0.1, 1, rows),
            "noise": rng.normal(0, 1, rows),
        }
    )


def test_best_single_and_equal_weights():
    train = training_frame()
    weights, state = best_single()(train, None)
    assert weights == {"good": 1.0, "bad": 0.0, "noise": 0.0}
    assert state is None
    weights, _ = equal_weights()(train, None)
    assert weights == pytest.approx({"good": 1 / 3, "bad": 1 / 3, "noise": 1 / 3})


def test_max_sharpe_beats_equal_weights_in_training():
    train = training_frame()
    returns = train.to_numpy()
    for long_only in (True, False):
        weights, _ = max_sharpe(long_only)(train, None)
        w = np.array([weights[c] for c in train.columns])
        assert np.abs(w).sum() == pytest.approx(1.0)
        if long_only:
            assert (w >= 0).all()
        assert sharpe(returns @ w) >= sharpe(returns.mean(axis=1)) - 1e-9
        assert weights["good"] > weights["bad"]


def test_random_search_keeps_a_shortlist_and_repeats_with_the_seed():
    train = training_frame()
    choose = random_search(candidates=200, keep=20, seed=7)
    first, state = choose(train, None)
    shortlist, _ = state
    assert shortlist.shape == (20, 3)
    assert sum(first.values()) == pytest.approx(1.0)
    assert min(first.values()) >= 0
    assert first["good"] > first["bad"]

    _, state = choose(train, state)
    assert state[0].shape == (20, 3)
    again, _ = random_search(candidates=200, keep=20, seed=7)(train, None)
    assert again == first


def test_drop_small_removes_tiny_weights_and_keeps_the_total():
    fixed = stateless(lambda train: {"good": 0.6, "bad": 0.37, "noise": 0.03})
    weights, _ = drop_small(fixed, below=0.05)(training_frame(), None)
    assert weights["noise"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["good"] / weights["bad"] == pytest.approx(0.6 / 0.37)
    everything = stateless(lambda train: {"good": 0.01, "bad": 0.01, "noise": 0.01})
    unchanged, _ = drop_small(everything, below=0.05)(training_frame(), None)
    assert unchanged == {"good": 0.01, "bad": 0.01, "noise": 0.01}


def test_sharpe_functions_match_the_backtest(prices):
    result = Backtest(prices).run(ewma_gap)
    returns = result.returns
    assert sharpe(returns) == pytest.approx(result.sharpe)
    assert adjusted_sharpe(returns) == pytest.approx(result.adjusted_sharpe)
    assert adjusted_sharpe(returns, 260) != pytest.approx(result.adjusted_sharpe)


def test_sharpe_functions_score_columns_at_once():
    matrix = training_frame().to_numpy()
    together = adjusted_sharpe(matrix)
    apart = [adjusted_sharpe(column) for column in matrix.T]
    np.testing.assert_allclose(together, apart)
    assert np.isnan(sharpe(np.array([1.0])))
    assert np.isnan(adjusted_sharpe(np.zeros(50)))
    with pytest.raises(ValueError, match="NaN"):
        sharpe(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_bootstrap_interval_contains_the_estimate(prices):
    returns = Backtest(prices).run(ewma_gap).returns
    result = bootstrap(returns, samples=500)
    assert result.estimate == pytest.approx(adjusted_sharpe(returns))
    assert result.low < result.estimate < result.high
    assert 0 <= result.p_at_most_zero < 0.5
    assert result.block == 10
    assert bootstrap(returns, samples=200, block=None).block >= 1
    assert result.draws.shape == (500,)
    assert "95% interval" in str(result)
    assert bootstrap(returns, samples=500, seed=1).low != result.low
    assert bootstrap(returns, sharpe, samples=200, block=20).block == 20


def test_bootstrap_finds_no_edge_in_noise():
    noise = np.random.default_rng(8).normal(0, 1, 3000)
    result = bootstrap(noise, samples=1000)
    assert result.low < 0 < result.high


def test_paired_bootstrap_matches_on_dates_and_compares(prices):
    bt = Backtest(prices)
    a, b = bt.run(ewma_sign).returns, bt.run(trend.forecast).returns
    same = paired_bootstrap(a, a, samples=200)
    assert same.estimate == 0 and same.low == 0 and same.high == 0

    result = paired_bootstrap(a, b, samples=500)
    joined = a.join(b, on=DATE, suffix="_b")
    expected = adjusted_sharpe(joined["returns"]) - adjusted_sharpe(joined["returns_b"])
    assert result.estimate == pytest.approx(expected)
    assert result.low < result.estimate < result.high
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap(np.zeros(10), np.zeros(11))
