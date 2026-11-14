"""VAR lead-lag strategy: leaders (Titan, Ganymede) predict followers (Callisto, Io, Europa).

Built on Seb's VARmodel (src/strategies/var_model.py): OLS fit, lag chosen by BIC.

The timing matters. A position decided after the close of day T earns the move from T+1 to T+2
(the instructions' two-day lag). A lead-lag that lasts one day (leaders on T -> followers on T+1)
is therefore not earnable. This strategy forecasts the move it can earn: it steps the VAR forward
twice to predict r[T+2] from returns up to T.

Run: uv run python YANNICK/var_strategy.py
"""

from functools import partial

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from data import PRICES
from strategies import var_model
from strategies.var_model import VARmodel

PLANETS = ["ganymede", "io", "titan", "callisto", "europa"]
SYSTEM = [f"{p}_{k}" for p in PLANETS for k in ("stocks", "bonds")]
FOLLOWERS = ["callisto_stocks", "callisto_bonds", "io_stocks", "io_bonds", "europa_stocks", "europa_bonds"]


def var_daily(
    prices: pl.DataFrame,
    system: tuple[str, ...] = tuple(SYSTEM),
    trade: tuple[str, ...] = tuple(FOLLOWERS),
    steps: int = 2,
    max_lag: int = 5,
    refit_every: int = 250,
    min_train: int = 750,
) -> pl.DataFrame:
    """Forecast of the daily log return `steps` days ahead, from a VAR on daily log returns.

    Refit on an expanding window every `refit_every` days, using only returns up to the fit date.
    steps=2 is the return a position decided today earns. steps=1 is the next day's return, which
    the two-day lag makes unearnable; it is here to show that.
    The forecast is divided by each asset's volatility, so it is in units of expected Sharpe.
    """
    assets = [c for c in prices.columns if c != "AsOfDate"]
    px = prices.to_pandas().set_index("AsOfDate")
    lr = np.log(px[list(system)]).diff().to_numpy()
    vol = pd.DataFrame(lr).rolling(100, min_periods=50).std().to_numpy()
    n, N = lr.shape
    out = np.full((n, N), np.nan)
    for start in range(min_train, n, refit_every):
        train = lr[1 : start + 1]  # returns known at the end of day `start`
        model = VARmodel(num_vars=N, max_lag=max_lag)
        model.load_training_data(train)
        model.fit()
        p = model.optimal_lag(verbose=False)
        c, A = model.intercepts[p - 1], model.coefs[p - 1]
        for t in range(start, min(start + refit_every, n)):
            hist = list(lr[t - p + 1 : t + 1])  # x_{t-p+1} .. x_t
            for _ in range(steps):
                recent = np.array(hist[-p:][::-1])  # newest first
                hist.append(c + np.einsum("mij,mj->i", A, recent))
            out[t] = hist[-1]
    fc = pd.DataFrame(out / vol, index=px.index, columns=list(system))
    frame = pd.DataFrame(0.0, index=px.index, columns=assets)
    for a in trade:
        frame[a] = fc[a]
    frame.iloc[:min_train] = np.nan
    return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())


def seb_weekly(prices: pl.DataFrame, trade: tuple[str, ...] | None = None) -> pl.DataFrame:
    """Seb's weekly VAR forecast on the stock and bond system, optionally trading a subset."""
    sub = prices.select(["AsOfDate", *SYSTEM])
    fc = var_model.forecast(sub).to_pandas().set_index("AsOfDate")
    assets = [c for c in prices.columns if c != "AsOfDate"]
    frame = pd.DataFrame(0.0, index=fc.index, columns=assets)
    for a in (trade or SYSTEM):
        frame[a] = fc[a]
    frame[fc.isna().all(axis=1)] = np.nan
    return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())


def lagged_correlations(prices: pd.DataFrame) -> pd.DataFrame:
    """corr(leader return on day T, follower return on day T+k), dev period."""
    lr = np.log(prices).diff()
    rows = {}
    for lead in ("titan_stocks", "ganymede_stocks", "titan_bonds", "ganymede_bonds"):
        for follow in FOLLOWERS:
            if lead.split("_")[1] != follow.split("_")[1]:
                continue
            rows[(lead, follow)] = {f"lag {k}": lr[lead].corr(lr[follow].shift(-k)) for k in (0, 1, 2, 3, 5)}
    return pd.DataFrame(rows).T.round(3)


def report(bt: Backtest, name: str, f) -> None:
    r = bt.run(f, name=name)
    x = r.returns.to_pandas().set_index("AsOfDate")["returns"].dropna()
    h = len(x) // 2
    sr = lambda s: s.mean() / s.std() * np.sqrt(252)  # noqa: E731
    print(f"  {name:<44} adj {r.adjusted_sharpe:+.2f}  gross {r.sharpe_ex_slippage:+.2f}  "
          f"halves {sr(x[:h]):+.2f}/{sr(x[h:]):+.2f}  hold {r.av_holding:.0f}d  1d-lag-sharpe {r.sharpe_1d_lag:+.2f}")


def main() -> None:
    px = pd.read_csv(PRICES, index_col=0, parse_dates=True)
    dev = px.iloc[: int(len(px) * 0.75)]
    print("Lead-lag on dev: corr(leader on T, follower on T+k). A position decided on T earns T+2.")
    print(lagged_correlations(dev).to_string())

    for scoring in ("pdf", "lynx"):
        bt = Backtest(PRICES, scoring=scoring)
        print(f"\n===== dev period, scoring={scoring}")
        report(bt, "VAR daily, forecast T+2, followers", var_daily)
        report(bt, "VAR daily, forecast T+2, Callisto only", partial(var_daily, trade=("callisto_stocks", "callisto_bonds")))
        report(bt, "VAR daily, forecast T+1 (unearnable), followers", partial(var_daily, steps=1))
        report(bt, "Seb weekly VAR, all 10", seb_weekly)
        report(bt, "Seb weekly VAR, followers", partial(seb_weekly, trade=tuple(FOLLOWERS)))


if __name__ == "__main__":
    main()
