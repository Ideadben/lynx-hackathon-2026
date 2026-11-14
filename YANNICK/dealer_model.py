"""Model the dealer: infer each asset's hidden drift with a Kalman filter, and bet on it.

The forensics on the example data suggest: price change = noise + a hidden, slowly changing drift.
That is a hidden-state problem, like reading an opponent's range in poker. The Kalman filter gives
the best estimate of the drift *and* how sure we are, so the bet can be sized by confidence.

Model, in vol-normalised returns z:
    z[t]  = mu[t] + noise,            noise ~ N(0, 1)
    mu[t] = phi * mu[t-1] + shock,    shock ~ N(0, q)
phi < 1 lets the drift die out over time (half-life ln(0.5)/ln(phi) days). (phi, q) are fitted by
maximum likelihood per asset, each January, on earlier years only.

Strategies tested (all dev period), against the acceptance gate used by the Fullhouse poker bot:
beat the baseline, hold up in both halves, and do not lose on any asset.

Run: uv run python YANNICK/dealer_model.py
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import paired_bootstrap
from strategies import trend

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "prices" / "example_prices.csv"
HALF_LIVES = np.array([20, 40, 80, 160, 320, 640])
PHIS = 0.5 ** (1 / HALF_LIVES)
QS = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])


def _z(px: pd.DataFrame) -> pd.DataFrame:
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    return ret / vol.shift(1)


def kalman(z: np.ndarray, phi: float, q: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Filtered drift mean and variance after each day, and the log-likelihood."""
    n = len(z)
    mean = np.zeros(n)
    var = np.zeros(n)
    m, p = 0.0, q / (1 - phi**2)  # start at the stationary distribution
    ll = 0.0
    for t in range(n):
        m, p = phi * m, phi * phi * p + q  # predict
        if not np.isnan(z[t]):
            s = p + 1.0
            v = z[t] - m
            ll += -0.5 * (np.log(s) + v * v / s)
            k = p / s
            m, p = m + k * v, (1 - k) * p
        mean[t], var[t] = m, p
    return mean, var, ll


def fit(z: np.ndarray) -> tuple[float, float]:
    best, arg = -np.inf, (PHIS[2], QS[2])
    for phi in PHIS:
        for q in QS:
            ll = kalman(z, phi, q)[2]
            if ll > best:
                best, arg = ll, (phi, q)
    return arg


def drift(prices: pl.DataFrame, size: str = "tstat", min_years: int = 3, fixed: tuple | None = None) -> pl.DataFrame:
    """Forecast = what the filter believes about the drift the position will earn.

    size: "mean" bets the expected drift (Kelly); "tstat" bets drift / its uncertainty;
    "sign" bets only the direction.
    """
    px = prices.to_pandas().set_index("AsOfDate")
    z = _z(px)
    years = z.index.year
    out = pd.DataFrame(np.nan, index=z.index, columns=z.columns)
    for a in z.columns:
        x = z[a].to_numpy()
        for year in sorted(set(years)):
            past = years < year
            if fixed is None and len(set(years[past])) < min_years:
                continue
            phi, q = fixed if fixed is not None else fit(x[past])
            rows = years <= year
            mean, var, _ = kalman(x[rows], phi, q)
            ahead = phi**2  # the position earns the move two days ahead
            mu, sd = mean * ahead, np.sqrt(var) * ahead
            sel = years[rows] == year
            if size == "mean":
                value = mu
            elif size == "sign":
                value = np.sign(mu)
            else:
                value = np.clip(mu / sd, -3, 3)
            out.loc[years == year, a] = value[sel]
    return pl.from_pandas(out.rename_axis("AsOfDate").reset_index())


def fitted_params(prices: pl.DataFrame) -> pd.DataFrame:
    z = _z(prices.to_pandas().set_index("AsOfDate"))
    rows = {}
    for a in z.columns:
        phi, q = fit(z[a].to_numpy())
        rows[a] = {"half_life_days": round(np.log(0.5) / np.log(phi)), "q": q,
                   "drift_sd": round(np.sqrt(q / (1 - phi**2)), 3)}
    return pd.DataFrame(rows).T


def gate(bt: Backtest, base, f, name: str) -> None:
    r = bt.run(f, name=name)
    diff = paired_bootstrap(r.returns, base.returns)
    ret = r.returns.to_pandas().iloc[:, -1].dropna()
    h = len(ret) // 2
    sr = lambda x: x.mean() / x.std() * np.sqrt(252)  # noqa: E731
    per = r.by_instrument().to_pandas().set_index("instrument")["sharpe_ex_slippage"]
    print(f"{name:<28} adj {r.adjusted_sharpe:.2f}  halves {sr(ret[:h]):.2f}/{sr(ret[h:]):.2f}  "
          f"worst asset {per.min():+.2f}  hold {r.av_holding:.0f}d  vs trend {diff}")


def main() -> None:
    bt = Backtest(PRICES)
    dev = bt.prices.head(int(bt.prices.height * 0.75))
    print("What the dealer model learns from all of dev (for reading only, not used in trading):")
    print(fitted_params(dev))

    base = bt.run(trend.forecast, name="trend")
    ret = base.returns.to_pandas().iloc[:, -1].dropna()
    h = len(ret) // 2
    sr = lambda x: x.mean() / x.std() * np.sqrt(252)  # noqa: E731
    print(f"\ntrend (baseline)             adj {base.adjusted_sharpe:.2f}  halves {sr(ret[:h]):.2f}/{sr(ret[h:]):.2f}")
    for size in ("tstat", "mean", "sign"):
        gate(bt, base, partial(drift, size=size), f"kalman, size={size}")
    print("\nRobustness: one fixed (half-life, q) for all assets, no fitting")
    for hl in (40, 80, 160):
        for q in (1e-4, 3e-4):
            gate(bt, base, partial(drift, fixed=(0.5 ** (1 / hl), q)), f"fixed hl={hl} q={q:g}")


if __name__ == "__main__":
    main()
