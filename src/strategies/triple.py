"""The triple strategy: slow trend + long stocks and bonds + macro, a third of the risk each.

Three independent sources of return, each a forecast function for ``Backtest`` or ``Portfolio``:

- ``slow_trend``: trend following (``strategies.trend``) at spans 16/32/64, on all instruments.
  Prices drift for weeks to months, so following the recent direction pays.
- ``long_stocks_bonds``: always long the stock and bond series, sized by the pipeline's vol scaling.
  Investors are paid a risk premium for holding them.
- ``macro``: growth and inflation acceleration per planet, with signs from textbook macro and no
  fitted weights:

      stocks of P        + growth acceleration of P
      bonds of P         - inflation acceleration of P - 0.5 x growth acceleration of P
      commodities        + world growth acceleration + world inflation acceleration
      FX of P vs titan   + growth acceleration of P - growth acceleration of titan

  Acceleration is the change over the last 91 days minus the change over the 91 days before,
  of log GDP or log CPI. Each feature is standardised with its expanding mean and std (past
  data only) and capped at ±3.

``PARTS`` and ``WEIGHTS`` give the parts and their equal weights; ``portfolio()`` builds the
three-part blend. Inside a larger portfolio, list the three parts directly (as
``portfolio.final`` does) rather than nesting this blend, so each part is vol-scaled once.

Research: YANNICK/macro_strategy.py, YANNICK/oos_test.py, YANNICK/block_validation.py.
"""

import numpy as np
import pandas as pd
import polars as pl

from strategies import trend

DATE = "AsOfDate"
PLANETS = ["ganymede", "io", "titan", "callisto", "europa"]
SLOW_SPANS = (16, 32, 64)
ACCEL_DAYS = 91
STANDARDISE_MIN_DAYS = 500


def _assets(prices: pl.DataFrame) -> list[str]:
    return [c for c in prices.columns if c != DATE]


def asset_class(asset: str) -> tuple[str, str | None]:
    """("stocks" | "bonds" | "titan_fx", planet) for planet series, ("commodity", None) otherwise."""
    for p in PLANETS:
        if asset.startswith(p + "_"):
            return asset[len(p) + 1 :], p
    return "commodity", None


def slow_trend(prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
    """Trend vote at spans 16/32/64 (slow EWMA four times slower), from -1 to +1."""
    return trend.forecast(prices, spans=SLOW_SPANS)


def long_stocks_bonds(prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
    """+1 on every stock and bond series, 0 elsewhere."""
    return prices.with_columns(
        pl.lit(1.0 if asset_class(c)[0] in ("stocks", "bonds") else 0.0).alias(c) for c in _assets(prices)
    )


def macro_features(extras: pl.DataFrame, dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Standardised growth and inflation acceleration per planet, plus 'world' (the planet average)."""
    raw = extras.to_pandas().set_index(DATE)
    raw.index = pd.to_datetime(raw.index)
    lg = np.log(raw.asfreq("D").ffill())  # calendar days, so a shift of 91 rows is 91 days
    out = {}
    for kind in ("gdp", "cpi"):
        x = lg[[f"{kind}|{p}" for p in PLANETS]].set_axis(PLANETS, axis=1)
        g = x - x.shift(ACCEL_DAYS)
        f = (g - g.shift(ACCEL_DAYS)).reindex(dates, method="ffill")
        f["world"] = f.mean(axis=1)
        mu = f.expanding(min_periods=STANDARDISE_MIN_DAYS).mean()
        sd = f.expanding(min_periods=STANDARDISE_MIN_DAYS).std()
        out[f"{kind}_accel"] = ((f - mu) / sd).clip(-3, 3)
    return out


def macro(prices: pl.DataFrame, extras: pl.DataFrame) -> pl.DataFrame:
    """The macro rule. Needs the GDP/CPI extras (``Backtest(prices, macro_csv)``)."""
    dates = pd.to_datetime(prices[DATE].to_pandas())
    f = macro_features(extras, pd.DatetimeIndex(dates))
    growth, inflation = f["gdp_accel"], f["cpi_accel"]
    out = pd.DataFrame(0.0, index=growth.index, columns=_assets(prices))
    for a in out.columns:
        cls, p = asset_class(a)
        if cls == "stocks":
            out[a] = growth[p]
        elif cls == "bonds":
            out[a] = -inflation[p] - 0.5 * growth[p]
        elif cls == "commodity":
            out[a] = growth["world"] + inflation["world"]
        else:  # titan_fx
            out[a] = growth[p] - growth["titan"]
    started = out.notna().any(axis=1).cummax()
    out[~started.to_numpy()] = np.nan  # empty until the features exist
    return prices.select(DATE).with_columns(
        pl.Series(a, out[a].to_numpy(), nan_to_null=True) for a in out.columns
    )


PARTS = {"slow_trend": slow_trend, "long_stocks_bonds": long_stocks_bonds, "macro": macro}
WEIGHTS = {name: 1 / 3 for name in PARTS}


def portfolio():
    """The three parts at a third each, as a positions function for ``Backtest.run_positions``."""
    from portfolio import Portfolio

    return Portfolio(PARTS, WEIGHTS)
