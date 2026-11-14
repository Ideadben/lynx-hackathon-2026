"""Strategies from the books in book/quant-resources, tested on the dev period.

Each is a forecast function for Backtest. Sources:
- Carver, Systematic Trading (2015): EWMAC at his six speeds, continuous forecasts; position inertia.
- Kaufman, Trading Systems and Methods (2013), ch. 8 and 17: N-day breakout, N-day momentum,
  regression slope, efficiency ratio filter, KAMA.
- Lopez de Prado, Advances in Financial ML (2018), ch. 3: meta-labelling.
- Kakushadze et al., Finding Alphas (WorldQuant, 2015): cross-sectional ranking.

Run: uv run python YANNICK/book_strategies.py
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.pipeline import to_positions
from backtest.stats import paired_bootstrap
from strategies import trend

ROOT = Path(__file__).resolve().parents[1]
from data import PRICES  # noqa: E402


def _pd(prices: pl.DataFrame) -> pd.DataFrame:
    return prices.to_pandas().set_index("AsOfDate")


def _pl(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())


def _vol(px: pd.DataFrame) -> pd.DataFrame:
    """Daily price-change volatility (Carver uses an EWMA with span 36)."""
    return px.diff().ewm(span=36, min_periods=36).std()


# Carver ---------------------------------------------------------------------------------------------
def ewmac(prices: pl.DataFrame, fasts: tuple[int, ...] = (2, 4, 8, 16, 32, 64), cap: float = 20.0) -> pl.DataFrame:
    """Carver's EWMAC: (fast EWMA - slow EWMA) / price vol, slow = 4 x fast, per speed scaled to an
    average absolute value of 10 (expanding), capped, then averaged across speeds."""
    px = _pd(prices)
    vol = _vol(px)
    total = 0
    for f in fasts:
        raw = (px.ewm(span=f, adjust=False).mean() - px.ewm(span=4 * f, adjust=False).mean()) / vol
        scale = 10 / raw.abs().expanding(min_periods=250).mean()  # past-only forecast scalar
        total = total + (raw * scale).clip(-cap, cap)
    return _pl(total / len(fasts))


# Kaufman --------------------------------------------------------------------------------------------
def breakout(prices: pl.DataFrame, windows: tuple[int, ...] = (20, 40, 80, 160)) -> pl.DataFrame:
    """Where is the price inside its N-day range? +1 at the high, -1 at the low, smoothed.

    Kaufman's N-day breakout in the continuous form Carver later used, averaged over windows.
    """
    px = _pd(prices)
    total = 0
    for n in windows:
        hi, lo = px.rolling(n).max(), px.rolling(n).min()
        pos = (px - (hi + lo) / 2) / (hi - lo)  # -0.5 .. 0.5
        total = total + (2 * pos).ewm(span=max(n // 4, 2), adjust=False).mean()
    return _pl(total / len(windows))


def momentum(prices: pl.DataFrame, windows: tuple[int, ...] = (20, 60, 120, 250)) -> pl.DataFrame:
    """N-day momentum: sign of the change over the last N days, averaged over N (time-series momentum)."""
    px = _pd(prices)
    return _pl(sum(np.sign(px - px.shift(n)) for n in windows) / len(windows))


def regression_slope(prices: pl.DataFrame, windows: tuple[int, ...] = (20, 40, 80, 160)) -> pl.DataFrame:
    """t-statistic of the linear-regression slope of price over the last N days, averaged over N."""
    px = _pd(prices)
    vol = _vol(px)
    total = 0
    for n in windows:
        t = np.arange(n, dtype=float)
        t -= t.mean()
        w = t / (t**2).sum()
        slope = px.rolling(n).apply(lambda y, w=w: float(np.dot(w, y)), raw=True)
        # slope per day relative to daily vol, scaled by sqrt(n) to make speeds comparable
        total = total + (slope / vol * np.sqrt(n)).clip(-4, 4)
    return _pl(total / len(windows))


def efficiency_ratio(px: pd.DataFrame, n: int) -> pd.DataFrame:
    """Kaufman's ER: net move over n days divided by the sum of absolute daily moves. 1 = clean trend."""
    return (px - px.shift(n)).abs() / px.diff().abs().rolling(n).sum()


def trend_er_filter(prices: pl.DataFrame, n: int = 40, power: float = 1.0) -> pl.DataFrame:
    """Trend, sized up when the market is trending cleanly and down when it is noisy."""
    px = _pd(prices)
    er = efficiency_ratio(px, n)
    rel = er / er.expanding(min_periods=250).mean()  # relative to the asset's own past ER
    return _pl(_pd(trend.forecast(prices)) * rel**power)


def kama(prices: pl.DataFrame, n: int = 20, fast: int = 2, slow: int = 60, smooth: int = 10) -> pl.DataFrame:
    """Kaufman's adaptive moving average: fast in clean trends, slow in noise. Forecast = its slope."""
    px = _pd(prices)
    er = efficiency_ratio(px, n).fillna(0.0).to_numpy()
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    x = px.to_numpy()
    out = np.full_like(x, np.nan)
    out[0] = x[0]
    for t in range(1, len(x)):
        prev = np.where(np.isnan(out[t - 1]), x[t], out[t - 1])
        out[t] = prev + sc[t] * (x[t] - prev)
    k = pd.DataFrame(out, index=px.index, columns=px.columns)
    slope = (k - k.shift(smooth)) / _vol(px)
    return _pl(slope)


# WorldQuant -----------------------------------------------------------------------------------------
def cross_sectional(prices: pl.DataFrame, n: int = 60) -> pl.DataFrame:
    """Rank the assets by their vol-adjusted n-day move; long the leaders, short the laggards.

    Market-neutral: it only bets on relative performance, not on direction.
    """
    px = _pd(prices)
    move = (px - px.shift(n)) / (_vol(px) * np.sqrt(n))
    ranks = move.rank(axis=1)
    return _pl(ranks.sub(ranks.mean(axis=1), axis=0))


# Lopez de Prado -------------------------------------------------------------------------------------
def _logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 25) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method. Returns [intercept, coefs]."""
    A = np.c_[np.ones(len(X)), X]
    beta = np.zeros(A.shape[1])
    reg = l2 * np.eye(A.shape[1])
    reg[0, 0] = 0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-A @ beta))
        grad = A.T @ (p - y) + reg @ beta
        hess = (A * (p * (1 - p))[:, None]).T @ A + reg
        beta -= np.linalg.solve(hess, grad)
    return beta


def meta_labelled(prices: pl.DataFrame, horizon: int = 20, min_years: int = 4) -> pl.DataFrame:
    """Trend decides the side; a logistic model decides the size, from the odds that the bet wins.

    Labels: did the trend's side earn a positive return over the next `horizon` days (only labels
    complete before each refit are used). Features: efficiency ratio, vol relative to its past, trend
    agreement across speeds. Refit each January on earlier years.
    """
    px = _pd(prices)
    side = np.sign(_pd(trend.forecast(prices)))
    vol = _vol(px)
    feats = {
        "er": efficiency_ratio(px, 40),
        "vol_ratio": vol / vol.rolling(250, min_periods=100).mean(),
        "agree": _pd(trend.forecast(prices)).abs(),
    }
    fwd = px.shift(-horizon) - px
    label = (np.sign(fwd) == side).astype(float).where(fwd.notna() & side.ne(0))
    years = px.index.year
    out = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    for year in sorted(set(years)):
        cutoff = px.index[years < year]
        if len(set(cutoff.year)) < min_years:
            continue
        # labels need `horizon` future days, so drop the last `horizon` rows before the cutoff
        train_idx = cutoff[: -horizon - 2]
        X = np.column_stack([feats[f].loc[train_idx].stack().to_numpy() for f in feats])
        y = label.loc[train_idx].stack().reindex(feats["er"].loc[train_idx].stack().index).to_numpy()
        ok = ~np.isnan(X).any(1) & ~np.isnan(y)
        mu, sd = X[ok].mean(0), X[ok].std(0)
        beta = _logistic((X[ok] - mu) / sd, y[ok])
        rows = years == year
        Xn = np.column_stack([feats[f].loc[rows].stack().to_numpy() for f in feats])
        p = np.full(len(Xn), np.nan)
        good = ~np.isnan(Xn).any(1)
        p[good] = 1 / (1 + np.exp(-np.c_[np.ones(good.sum()), (Xn[good] - mu) / sd] @ beta))
        prob = pd.Series(p, index=feats["er"].loc[rows].stack().index).unstack()
        # Bet more when the odds are good, fold (size 0) when they are no better than a coin flip.
        size = (2 * prob - 1).clip(lower=0) * 2
        out.loc[rows] = (side.loc[rows] * size).reindex(columns=px.columns).to_numpy()
    return _pl(out)


# Carver: position inertia ---------------------------------------------------------------------------
def with_inertia(forecast, threshold: float = 0.1):
    """Positions that only trade when the target moves more than `threshold` of the current size."""

    def positions(prices: pl.DataFrame) -> pl.DataFrame:
        target = _pd(to_positions(forecast(prices), prices))
        held = target.copy()
        cur = np.full(target.shape[1], np.nan)
        arr = target.to_numpy()
        res = np.full_like(arr, np.nan)
        for t in range(len(arr)):
            tgt = arr[t]
            new = np.where(np.isnan(cur) | (np.abs(tgt - cur) > threshold * np.abs(tgt)), tgt, cur)
            new = np.where(np.isnan(tgt), np.nan, new)
            res[t] = new
            cur = new
        held.loc[:, :] = res
        return _pl(held)

    return positions


def _ret(result) -> pd.Series:
    return result.returns.to_pandas().set_index("AsOfDate")["returns"].dropna()


def report(base, name: str, result) -> dict:
    ret = _ret(result)
    h = len(ret) // 2
    sr = lambda x: x.mean() / x.std() * np.sqrt(252)  # noqa: E731
    per = result.by_instrument().to_pandas().set_index("instrument")["sharpe_ex_slippage"]
    both = pd.concat([ret, _ret(base)], axis=1).dropna()
    return {
        "strategy": name,
        "adj_sharpe": round(result.adjusted_sharpe, 2),
        "half1": round(sr(ret[:h]), 2),
        "half2": round(sr(ret[h:]), 2),
        "worst_asset": round(per.min(), 2),
        "hold_days": round(result.av_holding),
        "corr_trend": round(both.corr().iloc[0, 1], 2),
    }


STRATEGIES = {
    "trend.py (baseline)": trend.forecast,
    "Carver EWMAC 6 speeds": ewmac,
    "Carver EWMAC 8-64": partial(ewmac, fasts=(8, 16, 32, 64)),
    "Kaufman breakout": breakout,
    "Kaufman N-day momentum": momentum,
    "Kaufman regression slope": regression_slope,
    "Kaufman trend x ER": trend_er_filter,
    "Kaufman KAMA": kama,
    "WorldQuant cross-sectional": cross_sectional,
    "LdP meta-labelled trend": meta_labelled,
}


def main() -> None:
    pd.set_option("display.width", 200)
    for scoring in ("pdf", "lynx"):
        bt = Backtest(PRICES, scoring=scoring)
        base = bt.run(trend.forecast, name="trend")
        rows = []
        results = {name: bt.run(f, name=name) for name, f in STRATEGIES.items()}
        rows = [report(base, name, r) for name, r in results.items()]
        for name, f in {"trend.py + inertia 10%": trend.forecast, "EWMAC 6 + inertia 10%": ewmac}.items():
            rows.append(report(base, name, bt.run_positions(with_inertia(f), name=name)))
        print(f"\n===== dev period, scoring={scoring} =====")
        print(pd.DataFrame(rows).to_string(index=False))
        if scoring != "pdf":
            continue
        print("\nPaired bootstrap vs trend.py on the same days, PDF scoring:")
        for name, r in results.items():
            if name == "trend.py (baseline)":
                continue
            both = pd.concat([_ret(r), _ret(base)], axis=1, join="inner")
            print(f"  {name:<28} {paired_bootstrap(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())}")

    bt = Backtest(PRICES)
    print("\nMeta-labelling robustness: label horizon")
    for h in (10, 20, 40):
        print(f"  horizon {h:>2}: adj {bt.run(partial(meta_labelled, horizon=h)).adjusted_sharpe:.2f}")


if __name__ == "__main__":
    main()
