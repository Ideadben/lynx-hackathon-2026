from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from backtest import Backtest
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
    dates = pl.date_range(pl.date(2000, 1, 3), pl.date(2030, 1, 1), "1d", eager=True)
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
