"""Turns a forecast into risk-managed positions, following equations (2) and (3) of the instructions.

A forecast is a frame with ``AsOfDate`` plus one column per instrument. Its sign is the direction and
its size is the conviction. Row T may only use data up to and including T.
"""

from pathlib import Path
from typing import Any

import polars as pl

from backtest.evaluation import DATE, POSITION_LAG

Data = (
    str | Path | pl.DataFrame | Any
)  # Any covers pandas frames, which we don't import here.
"""A data source: a csv path, or a polars or pandas frame."""
DATE_NAMES = (
    "asofdate",
    "dates",
    "date",
    "index",
)  # "index" is an unnamed pandas date index.
RISK_WINDOW = 100
FORECAST_TARGET = 10.0  # Average absolute forecast after scaling. Our own convention.
FORECAST_CAP = 20.0


def load_data(source: Data) -> pl.DataFrame:
    """Reads a data source into the standard frame.

    The result has an ``AsOfDate`` column of dates, plus float columns, sorted by date. A missing
    value repeats the last known value, which uses no future data. Values missing at the very
    start stay null.

    Args:
        source: A csv path, or a polars or pandas frame. Needs exactly one date column, called
            ``AsOfDate``, ``dates`` or ``date`` (any case), or a pandas date index.

    Returns:
        The frame, with every column except ``AsOfDate`` as floats.

    Raises:
        ValueError: If there is no single date column, or a column isn't numeric.
        TypeError: If ``source`` isn't a path or frame.
    """
    if isinstance(source, (str, Path)):
        # Scan every row: an asset that starts late would otherwise be read as text.
        frame = pl.read_csv(source, try_parse_dates=True, infer_schema_length=None)
    elif isinstance(source, pl.DataFrame):
        frame = source
    elif hasattr(source, "reset_index"):
        has_date_column = any(
            str(column).lower() in DATE_NAMES for column in source.columns
        )
        frame = pl.from_pandas(source if has_date_column else source.reset_index())
    else:
        raise TypeError(
            f"Expected a csv path, or a polars or pandas frame, not {type(source).__name__}."
        )

    frame = _standard_date(frame)
    text = [
        name
        for name, dtype in frame.schema.items()
        if name != DATE and not dtype.is_numeric()
    ]
    if text:
        raise ValueError(
            f"All columns except the date must be numeric, but these are not: {text}"
        )
    values = (
        pl.exclude(DATE).cast(pl.Float64).fill_nan(None).fill_null(strategy="forward")
    )
    return frame.sort(DATE).with_columns(values)


def instruments(frame: pl.DataFrame) -> list[str]:
    """The column names of a frame, without ``AsOfDate``.

    Args:
        frame: Any frame in the standard format.

    Returns:
        The asset (or series) names, in column order.
    """
    return [column for column in frame.columns if column != DATE]


def vol_scale(
    forecast: pl.DataFrame,
    prices: pl.DataFrame,
    vol_window: int = RISK_WINDOW,
    cap: float | None = FORECAST_CAP,
    normalize: bool = True,
) -> pl.DataFrame:
    """Equation (2): scales and caps a forecast, then divides by asset volatility.

    With ``normalize``, each asset is first scaled to an average absolute forecast of 10 over the
    data so far, then capped at ``cap``. Dividing by volatility gives each asset about equal risk.

    Args:
        forecast: ``AsOfDate`` plus one column per asset, one row per price date.
        prices: ``AsOfDate`` plus one price column per asset.
        vol_window: Days of squared price changes used for volatility.
        cap: Largest absolute forecast after scaling. ``None`` for no cap.
        normalize: Scale to an average size of 10. If False, the forecast is used as it is and
            ``cap`` has no effect.

    Returns:
        Positions before strategy risk scaling. Null where there is no forecast or volatility.

    Raises:
        ValueError: If the forecast doesn't line up with the prices.
    """
    columns = instruments(prices)
    check_frame(forecast, prices)
    values = _finite(forecast.select(columns))

    if normalize:
        average_size = values.select(
            pl.all().abs().fill_null(0).cum_sum() / pl.all().cum_count()
        )
        values = _finite(values * FORECAST_TARGET / average_size)
        if cap is not None:
            values = values.select(pl.all().clip(-cap, cap))

    volatility = prices.select(
        pl.col(columns).diff().pow(2).rolling_mean(vol_window).sqrt()
    )
    return (
        _finite(values / volatility).with_columns(prices[DATE]).select(DATE, *columns)
    )


def risk_scale(
    positions: pl.DataFrame, prices: pl.DataFrame, risk_window: int = RISK_WINDOW
) -> pl.DataFrame:
    """Equation (3): divides positions by the strategy's trailing risk.

    Risk is the standard deviation of the strategy's daily returns over ``risk_window`` days, with
    the two-day lag. This keeps total risk at about 1 per day.

    Args:
        positions: ``AsOfDate`` plus one column per asset, one row per price date.
        prices: ``AsOfDate`` plus one price column per asset.
        risk_window: Days of strategy returns used for risk.

    Returns:
        The scaled positions. Null until the window has filled.
    """
    columns = instruments(positions)
    returns = prices.select(pl.col(columns).diff())
    asset_returns = positions.select(pl.col(columns).shift(POSITION_LAG)) * returns

    any_position = pl.any_horizontal(pl.all().is_not_null())
    strategy_returns = asset_returns.select(
        pl.when(any_position).then(pl.sum_horizontal(pl.all())).alias("strategy_return")
    ).to_series()
    risk = strategy_returns.rolling_std(risk_window)

    scaled = positions.select(pl.col(columns) / risk)
    return _finite(scaled).with_columns(positions[DATE]).select(DATE, *columns)


def to_positions(
    forecast: pl.DataFrame,
    prices: pl.DataFrame,
    vol_window: int = RISK_WINDOW,
    risk_window: int = RISK_WINDOW,
    cap: float | None = FORECAST_CAP,
    normalize: bool = True,
) -> pl.DataFrame:
    """The full pipeline: forecast to final positions.

    Runs ``vol_scale``, then ``risk_scale``. ``Backtest.run`` calls this for you.

    Args:
        forecast: ``AsOfDate`` plus one column per asset, one row per price date.
        prices: ``AsOfDate`` plus one price column per asset.
        vol_window: Days used for asset volatility.
        risk_window: Days used for strategy risk.
        cap: Largest absolute forecast after scaling. ``None`` for no cap.
        normalize: Scale each asset to an average absolute forecast of 10.

    Returns:
        Final positions, ready to score or blend.

    Raises:
        ValueError: If the forecast doesn't line up with the prices.
    """
    scaled = vol_scale(forecast, prices, vol_window, cap, normalize)
    return risk_scale(scaled, prices, risk_window)


def _standard_date(frame: pl.DataFrame) -> pl.DataFrame:
    """Renames the date column to ``AsOfDate`` and gives it the date type."""
    names = [column for column in frame.columns if column.lower() in DATE_NAMES]
    if len(names) != 1:
        raise ValueError(
            f"Expected one date column named like {DATE_NAMES}, found {names or 'none'}."
        )
    frame = frame.rename({names[0]: DATE})
    dtype = frame.schema[DATE]
    if dtype == pl.String:
        return frame.with_columns(pl.col(DATE).str.to_date())
    if dtype == pl.Date or isinstance(dtype, pl.Datetime):
        return frame.with_columns(pl.col(DATE).cast(pl.Date))
    raise ValueError(f"The date column must hold dates, not {dtype}.")


def _finite(frame: pl.DataFrame) -> pl.DataFrame:
    """Replaces NaN and infinity, for example from zero volatility, with null."""
    return frame.select(
        pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in frame.columns
    )


def check_frame(frame: pl.DataFrame, prices: pl.DataFrame) -> None:
    """Checks that a forecast or positions frame lines up with the prices.

    Args:
        frame: The frame to check.
        prices: The prices it should line up with.

    Raises:
        ValueError: If ``frame`` has no ``AsOfDate`` column, misses an asset, or its dates don't
            match the price dates one to one.
    """
    if DATE not in frame.columns:
        raise ValueError(f"The frame needs an {DATE} column.")
    missing = set(instruments(prices)) - set(frame.columns)
    if missing:
        raise ValueError(f"The frame has no column for: {sorted(missing)}")
    if not frame[DATE].equals(prices[DATE]):
        raise ValueError(
            "The frame must have exactly one row per price date, in the same order."
        )
