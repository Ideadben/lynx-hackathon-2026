"""Macro strategies, and two commodity strategies. All are forecast(prices, extras) functions.

Macro features per planet (from daily GDP and CPI, used as of each price date):
    gdp_accel, cpi_accel: growth over the last 3 months minus growth over the 3 months before
    gdp_12m,   cpi_12m:   growth over the last 12 months
Each is standardised with an expanding mean and std, so only past data shapes the scale.

1. macro_prior: signs from textbook macro, equal weights, nothing fitted.
     stocks  +gdp_accel                 (improving growth lifts earnings)
     bonds   -cpi_accel - gdp_accel/2   (rising inflation and growth push yields up, prices down)
     commodities  +world gdp_accel +world cpi_accel   (demand, and inflation hedging)
     fx (planet vs titan)  +(gdp_accel_planet - gdp_accel_titan)   (growth attracts capital)
2. macro_walk_forward: per asset class, a ridge regression of the next 20 days' move on all
   features, pooled over the planets, refit each January on earlier years only.
3. commodity_pairs: spread trading on the three pairs of commodities that move together.

Run: uv run python YANNICK/macro_strategy.py
"""

from functools import partial

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import paired_bootstrap
from data import MACRO, PRICES
from portfolio import Portfolio
from strategies import trend

PLANETS = ["ganymede", "io", "titan", "callisto", "europa"]
FEATURES = ["gdp_accel", "cpi_accel", "gdp_12m", "cpi_12m"]
PAIRS = [("americium_ore", "refined_americium"), ("beryllium", "beryllium_monofilaments"), ("antimatter", "aerogel")]


def _pd(frame: pl.DataFrame) -> pd.DataFrame:
    return frame.to_pandas().set_index("AsOfDate")


def _pl(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())


def asset_class(asset: str) -> tuple[str, str | None]:
    for p in PLANETS:
        if asset.startswith(p + "_"):
            return asset[len(p) + 1 :], p  # "stocks" | "bonds" | "titan_fx"
    return "commodity", None


def macro_features(extras: pl.DataFrame, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Standardised features per planet (columns = planets, plus 'world' = planet average)."""
    lg = np.log(_pd(extras))
    lg.index = pd.to_datetime(lg.index)
    out = {}
    for kind in ("gdp", "cpi"):
        x = lg[[f"{kind}|{p}" for p in PLANETS]].set_axis(PLANETS, axis=1)
        g3 = x - x.shift(91)
        raw = {f"{kind}_accel": g3 - g3.shift(91), f"{kind}_12m": x - x.shift(365)}
        for name, f in raw.items():
            f = f.reindex(index, method="ffill")
            f["world"] = f.mean(axis=1)
            mu = f.expanding(min_periods=500).mean()
            sd = f.expanding(min_periods=500).std()
            out[name] = ((f - mu) / sd).clip(-3, 3)
    return out


def macro_prior(prices: pl.DataFrame, extras: pl.DataFrame, classes: tuple[str, ...] = ("stocks", "bonds", "commodity", "titan_fx")) -> pl.DataFrame:
    px = _pd(prices)
    f = macro_features(extras, pd.to_datetime(px.index))
    out = pd.DataFrame(0.0, index=pd.to_datetime(px.index), columns=px.columns)
    for a in px.columns:
        cls, p = asset_class(a)
        if cls not in classes:
            continue
        if cls == "stocks":
            out[a] = f["gdp_accel"][p]
        elif cls == "bonds":
            out[a] = -f["cpi_accel"][p] - 0.5 * f["gdp_accel"][p]
        elif cls == "commodity":
            out[a] = f["gdp_accel"]["world"] + f["cpi_accel"]["world"]
        elif cls == "titan_fx":
            out[a] = f["gdp_accel"][p] - f["gdp_accel"]["titan"]
    out[out.isna().all(axis=1) | (out.index < out.dropna(how="all").index.min())] = np.nan
    out.index = px.index
    return _pl(out)


def macro_walk_forward(prices: pl.DataFrame, extras: pl.DataFrame, horizon: int = 20, l2: float = 50.0,
                       min_years: int = 5) -> pl.DataFrame:
    """Per asset class, pooled ridge of the next `horizon` days' vol-normalised move on the features."""
    px = _pd(prices)
    idx = pd.to_datetime(px.index)
    f = macro_features(extras, idx)
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    z = (ret / vol.shift(1)).set_axis(idx)
    y = sum(z.shift(-k) for k in range(2, 2 + horizon))
    years = idx.year
    out = pd.DataFrame(np.nan, index=idx, columns=px.columns)
    groups: dict[str, list[tuple[str, str]]] = {}
    for a in px.columns:
        cls, p = asset_class(a)
        groups.setdefault(cls, []).append((a, p or "world"))

    def design(a: str, p: str, rows) -> np.ndarray:
        cols = [f[n][p].loc[rows] for n in FEATURES]
        if asset_class(a)[0] == "titan_fx":
            cols = [f[n][p].loc[rows] - f[n]["titan"].loc[rows] for n in FEATURES]
        return np.column_stack(cols)

    for year in sorted(set(years)):
        past = idx[years < year]
        if len(set(past.year)) < min_years:
            continue
        train = past[: -(horizon + 2)]  # labels fully known before the refit
        now = idx[years == year]
        for cls, members in groups.items():
            X = np.vstack([design(a, p, train) for a, p in members])
            Y = np.concatenate([y[a].loc[train].to_numpy() for a, _ in members])
            ok = ~np.isnan(X).any(1) & ~np.isnan(Y)
            if ok.sum() < 500:
                continue
            beta = np.linalg.solve(X[ok].T @ X[ok] + l2 * np.eye(len(FEATURES)), X[ok].T @ Y[ok])
            for a, p in members:
                out.loc[now, a] = design(a, p, now) @ beta
    out.index = px.index
    return _pl(out)


def commodity_pairs(prices: pl.DataFrame, extras: pl.DataFrame | None = None, n: int = 60, beta_window: int = 500) -> pl.DataFrame:
    """For each pair, bet that the log-price spread returns to its n-day mean.

    The hedge ratio is a rolling regression of log A on log B over `beta_window` days. Long the
    cheap leg, short the rich one, sized by the spread's z-score.
    """
    px = _pd(prices)
    lp = np.log(px)
    out = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for a, b in PAIRS:
        cov = lp[a].rolling(beta_window).cov(lp[b])
        beta = cov / lp[b].rolling(beta_window).var()
        spread = lp[a] - beta * lp[b]
        zs = ((spread - spread.rolling(n).mean()) / spread.rolling(n).std()).clip(-2, 2)
        out[a] = -zs
        out[b] = zs * beta
    out[out.index < out.index[beta_window + n]] = np.nan
    return _pl(out)


# baseline pieces from earlier ---------------------------------------------------------------------
def slow_trend(prices: pl.DataFrame, extras: pl.DataFrame | None = None) -> pl.DataFrame:
    return trend.forecast(prices, spans=(16, 32, 64))


def long_stocks_bonds(prices: pl.DataFrame, extras: pl.DataFrame | None = None) -> pl.DataFrame:
    cols = [c for c in prices.columns if c != "AsOfDate"]
    return prices.with_columns(pl.lit(1.0 if c.endswith(("stocks", "bonds")) else 0.0).alias(c) for c in cols)


def _ret(r) -> pd.Series:
    return r.returns.to_pandas().set_index("AsOfDate")["returns"].dropna()


def line(name: str, r, base) -> str:
    x = _ret(r)
    h = len(x) // 2
    sr = lambda s: s.mean() / s.std() * np.sqrt(252)  # noqa: E731
    both = pd.concat([x, _ret(base)], axis=1, join="inner")
    return (f"  {name:<42} adj {r.adjusted_sharpe:+.2f}  gross {r.sharpe_ex_slippage:+.2f}  halves "
            f"{sr(x[:h]):+.2f}/{sr(x[h:]):+.2f}  hold {r.av_holding:3.0f}d  corr(base) {both.corr().iloc[0, 1]:+.2f}")


def main() -> None:
    pd.set_option("display.width", 200)
    # Cointegration check for the pairs (dev)
    from statsmodels.tsa.stattools import coint

    px = pd.read_csv(PRICES, index_col=0, parse_dates=True)
    dev = px.iloc[: int(len(px) * 0.75)]
    print("Commodity pairs on dev: return correlation, Engle-Granger cointegration p-value on log prices")
    for a, b in PAIRS:
        print(f"  {a:>14} / {b:<24} corr {np.log(dev[a]).diff().corr(np.log(dev[b]).diff()):.2f}   "
              f"coint p={coint(np.log(dev[a]), np.log(dev[b]))[1]:.3f}")

    for scoring in ("pdf", "lynx"):
        bt = Backtest(PRICES, MACRO, scoring=scoring)
        base = bt.run_positions(Portfolio({"t": slow_trend, "l": long_stocks_bonds}, {"t": 0.5, "l": 0.5}), name="base")
        print(f"\n===== dev period, scoring={scoring}.  base = slow trend 50 / long stocks+bonds 50: adj {base.adjusted_sharpe:.2f}")
        tests = {
            "macro prior, all classes": macro_prior,
            "macro prior, stocks only": partial(macro_prior, classes=("stocks",)),
            "macro prior, bonds only": partial(macro_prior, classes=("bonds",)),
            "macro prior, commodities only": partial(macro_prior, classes=("commodity",)),
            "macro prior, fx only": partial(macro_prior, classes=("titan_fx",)),
            "macro walk-forward ridge, all": macro_walk_forward,
            "commodity pairs, spread n=60": commodity_pairs,
            "commodity pairs, spread n=20": partial(commodity_pairs, n=20),
            "commodity pairs, spread n=120": partial(commodity_pairs, n=120),
        }
        results = {}
        for name, f in tests.items():
            results[name] = bt.run(f, name=name)
            print(line(name, results[name], base))
        if scoring == "pdf":
            print("\n  macro prior by asset (gross Sharpe):")
            per = results["macro prior, all classes"].by_instrument().to_pandas().set_index("instrument")["sharpe_ex_slippage"]
            print("  " + per.round(2).to_string().replace("\n", "\n  "))

        print("\n  Blends (paired bootstrap vs base on the same days):")
        blends = {
            "base 70 / macro prior 30": ({"b1": slow_trend, "b2": long_stocks_bonds, "m": macro_prior}, {"b1": 0.35, "b2": 0.35, "m": 0.3}),
            "base 60 / macro prior 40": ({"b1": slow_trend, "b2": long_stocks_bonds, "m": macro_prior}, {"b1": 0.3, "b2": 0.3, "m": 0.4}),
            "base 70 / macro walk-forward 30": ({"b1": slow_trend, "b2": long_stocks_bonds, "m": macro_walk_forward}, {"b1": 0.35, "b2": 0.35, "m": 0.3}),
            "base 60 / macro prior 30 / pairs 10": ({"b1": slow_trend, "b2": long_stocks_bonds, "m": macro_prior, "p": commodity_pairs},
                                                    {"b1": 0.3, "b2": 0.3, "m": 0.3, "p": 0.1}),
        }
        for name, (S, w) in blends.items():
            r = bt.run_positions(Portfolio(S, w), name=name)
            both = pd.concat([_ret(r), _ret(base)], axis=1, join="inner")
            d = paired_bootstrap(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())
            print(line(name, r, base) + f"\n{'':46}vs base {str(d)[:50]}")


if __name__ == "__main__":
    main()
