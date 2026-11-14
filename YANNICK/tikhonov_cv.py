"""Tikhonov lag filter: choose L (lags in the design matrix), H (prediction horizon), Γ and λ
by block validation inside dev.

    prediction[t] = sum_{j<L} w_j z[t-j]  of  y[t] = z[t+2] + ... + z[t+1+H]
    minimise ||Xw - y||^2 + λ ||Γw||^2

Block validation (expanding window, yearly blocks): for each year 2003..2010, fit on all earlier
years and predict that year. The last H+2 rows before the block are dropped from the fit, since
their labels reach into it. The score is the pooled squared error over the 8 blocks, relative to
predicting zero (below 1 = the filter helps). Only dev years are used to choose.

The chosen setting is then traded walk-forward (refit each January on all earlier years), and its
Sharpe is reported per window: dev blocks, the 15% validation block, and the 10% test block (once).

Run: uv run python YANNICK/tikhonov_cv.py
"""

import itertools
import time

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from lag_tikhonov import z_returns

LS = (20, 60, 120, 250)
HS = (1, 5, 20, 60)
GAMMAS = ("ridge", "smooth")
LAMS = np.logspace(0, 9, 28)
CV_YEARS = range(2003, 2011)


def gamma(kind: str, L: int) -> np.ndarray:
    if kind == "ridge":
        return np.eye(L)
    D = np.zeros((L - 2, L))
    for i in range(L - 2):
        D[i, i:i + 3] = (1, -2, 1)
    return np.vstack([10 * D, 0.1 * np.eye(L)])


def year_blocks(z: pd.DataFrame, L: int, H: int) -> dict:
    """Per calendar year: normal equations of all its rows, and of its last H+2 rows (the purge)."""
    years = z.index.year
    y = sum(z.shift(-k) for k in range(2, 2 + H))
    blocks = {}
    for a in z.columns:
        za = np.nan_to_num(z[a].to_numpy())
        ya = y[a].to_numpy()
        ok = np.isfinite(ya) & np.isfinite(z[a].shift(L - 1).to_numpy())
        ok[: L - 1] = False
        idx = np.where(ok)[0]
        X = np.stack([za[idx - j] for j in range(L)], axis=1)
        b = ya[idx]
        yr = years[idx]
        for year in np.unique(yr):
            sel = yr == year
            Xs, bs = X[sel], b[sel]
            tail = slice(max(0, len(bs) - (H + 2)), None)
            full = blocks.setdefault(year, {"XtX": np.zeros((L, L)), "Xty": np.zeros(L), "yty": 0.0,
                                             "tXtX": np.zeros((L, L)), "tXty": np.zeros(L), "tyty": 0.0})
            full["XtX"] += Xs.T @ Xs
            full["Xty"] += Xs.T @ bs
            full["yty"] += bs @ bs
            full["tXtX"] += Xs[tail].T @ Xs[tail]
            full["tXty"] += Xs[tail].T @ bs[tail]
            full["tyty"] += bs[tail] @ bs[tail]
    return blocks


def train_eq(blocks: dict, year: int, L: int):
    """Normal equations of all years before `year`, without the last H+2 rows of the year before."""
    XtX, Xty = np.zeros((L, L)), np.zeros(L)
    for y_, b in blocks.items():
        if y_ < year:
            XtX += b["XtX"]
            Xty += b["Xty"]
    if year - 1 in blocks:
        XtX -= blocks[year - 1]["tXtX"]
        Xty -= blocks[year - 1]["tXty"]
    return XtX, Xty


def cv_errors(blocks: dict, L: int, kind: str) -> np.ndarray:
    """Pooled relative validation error over CV_YEARS, for every λ."""
    GtG = gamma(kind, L).T @ gamma(kind, L)
    num, den = np.zeros(len(LAMS)), 0.0
    for year in CV_YEARS:
        XtX, Xty = train_eq(blocks, year, L)
        v = blocks[year]
        den += v["yty"]
        for i, lam in enumerate(LAMS):
            w = np.linalg.solve(XtX + lam * GtG, Xty)
            num[i] += w @ v["XtX"] @ w - 2 * w @ v["Xty"] + v["yty"]
    return num / den


def walk_forward_forecast(z: pd.DataFrame, blocks: dict, L: int, kind: str, lam: float, first: int = 2000) -> pd.DataFrame:
    GtG = gamma(kind, L).T @ gamma(kind, L)
    years = z.index.year
    out = pd.DataFrame(np.nan, index=z.index, columns=z.columns)
    zz = np.nan_to_num(z.to_numpy())
    for year in sorted(set(years)):
        if year < first:
            continue
        XtX, Xty = train_eq(blocks, year, L)
        w = np.linalg.solve(XtX + lam * GtG, Xty)
        rows = np.where(years == year)[0]
        for k, a in enumerate(z.columns):
            f = np.convolve(zz[:, k], w)[: len(zz)]
            out.iloc[rows, k] = f[rows]
    return out


def trade(bt: Backtest, fc: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    frame = pl.from_pandas(fc.set_axis(bt.prices["AsOfDate"].to_pandas()).rename_axis("AsOfDate").reset_index())
    res = bt.run(lambda prices, *e: frame, period="full", check_leakage=False)  # honest by construction: each year uses earlier years only
    r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
    r.index = pd.to_datetime(r.index)
    return r


def stats(r: pd.Series, a, b) -> str:
    x = r[(r.index >= a) & (r.index < b)].dropna().to_numpy()
    return f"{float(adjusted_sharpe(x)):+.2f}"


def main() -> None:
    pd.set_option("display.width", 220)
    bt = Backtest(PRICES, MACRO)
    px = bt.prices.to_pandas().set_index("AsOfDate")
    px.index = pd.to_datetime(px.index)
    dates = px.index
    n = len(dates)
    z = z_returns(px)
    win = {"dev blocks 2003-2010": (pd.Timestamp("2003-01-01"), pd.Timestamp("2011-01-01")),
           "validation 15%": (dates[int(n * .75)], dates[int(n * .9)]),
           "test 10%": (dates[int(n * .9)], dates[-1] + pd.Timedelta(days=1))}

    results, cache = [], {}
    for L, H in itertools.product(LS, HS):
        t0 = time.time()
        blocks = year_blocks(z, L, H)
        cache[(L, H)] = blocks
        for kind in GAMMAS:
            err = cv_errors(blocks, L, kind)
            i = int(np.argmin(err))
            results.append({"L": L, "H": H, "Γ": kind, "best λ": LAMS[i], "cv error": err[i], "cv error OLS": err[0]})
        print(f"  L={L:>3} H={H:>2}: best cv error {min(r['cv error'] for r in results[-2:]):.5f}  ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(results).sort_values("cv error")
    print("\nBlock validation inside dev (8 yearly blocks 2003-2010), error relative to predicting 0:")
    print(res.head(12).to_string(index=False, float_format=lambda v: f"{v:.5f}" if v < 10 else f"{v:.3g}"))
    grid = res.loc[res.groupby(["L", "H"])["cv error"].idxmin()].pivot(index="L", columns="H", values="cv error")
    print("\nBest cv error per (L, H), any Γ and λ (rows L = lags, columns H = horizon):")
    print(grid.round(5).to_string())

    # Trade: the chosen setting, plus the best setting per horizon for comparison (no test for those)
    best = res.iloc[0]
    print(f"\nChosen: L = {best['L']}, H = {best['H']}, Γ = {best['Γ']}, λ = {best['best λ']:.3g}")
    compare = res.loc[res.groupby("H")["cv error"].idxmin()]
    rows = []
    for _, r_ in compare.iterrows():
        L, H, kind, lam = int(r_["L"]), int(r_["H"]), r_["Γ"], float(r_["best λ"])
        ret = trade(bt, walk_forward_forecast(z, cache[(L, H)], L, kind, lam), dates)
        chosen = (L == best["L"]) and (H == best["H"]) and kind == best["Γ"]
        row = {"setting": f"L={L}, H={H}, {kind}, λ={lam:.2g}" + ("  (chosen)" if chosen else ""), "cv error": r_["cv error"]}
        for k, (a, b) in win.items():
            row[k] = stats(ret, a, b) if (k != "test 10%" or chosen) else "not scored"
        rows.append(row)
    print("\nWalk-forward trading, adjusted Sharpe (test scored only for the chosen setting):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
