"""Tikhonov lag filter: how much do the input features matter?

    prediction[t] = sum_{j<L} w_j g[t - j*b]  (+ GARCH regime blocks)   g = feature summed over b days (b = 1, 5, 21)
    target y[t]   = z[t+2] + ... + z[t+1+H]        (vol-normalised move a position earns; same for every variant)
    minimise ||Xw - y||^2 + lam ||Γw||^2,   lam = alpha * trace(X'X) / L   (so alpha means the same for every feature)

For each feature variant, Γ (ridge / smooth) and alpha are chosen by block validation on dev:
refit each January on all earlier years (labels purged at the boundary), trade the next year, and
score the adjusted Sharpe of the blocks 2003-2010. The validation block (75-90% of dates) is then
reported for the chosen setting. The test block is not used.

Writes YANNICK/output/tikhonov_features/*.json for the Tikhonov Lab page.
Run: uv run python YANNICK/tikhonov_features.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES

H = 5
ALPHAS = np.logspace(-4, 3, 15)
DEV_BLOCKS = (2003, 2010)
OUT = Path(__file__).resolve().parent / "output" / "tikhonov_features"


def vol_norm(x: pd.DataFrame) -> pd.DataFrame:
    vol = np.sqrt((x**2).rolling(100, min_periods=50).mean())
    return (x / vol.shift(1)).clip(-6, 6)


def garch(prices: pl.DataFrame, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Søren's GARCH(1,1), refit quarterly on past data: next-day vol, and the standardised vol regime
    v = log(GARCH vol / its long-run level), scaled by its expanding mean and std (past only)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Søren"))
    import garch_lib

    paths = garch_lib.garch_paths(prices)
    sigma = pd.DataFrame({a: p.sigma for a, p in paths.items()}, index=index)
    ratio = np.log(pd.DataFrame({a: p.sigma / p.longrun for a, p in paths.items()}, index=index))
    v = ((ratio - ratio.expanding(250).mean()) / ratio.expanding(250).std()).clip(-3, 3)
    return sigma, v


def variants(px: pd.DataFrame, sigma: pd.DataFrame, v: pd.DataFrame) -> dict:
    """Each input is a list of column blocks (name, feature, bucket days b, lags L, multiplier).
    A block's columns are the feature summed over b days at lags 0..L-1, times the multiplier
    at the decision day (None = 1)."""
    d = px.diff()
    lr = np.log(px).diff()
    z = vol_norm(d)
    zg = (d / sigma.shift(1)).clip(-6, 6)  # sigma[t-1] is GARCH's forecast for day t
    one = lambda f, b, L: [("base", f, b, L, None)]  # noqa: E731
    return {
        "Daily price change ÷ vol (current)": one(z, 1, 250),
        "Daily raw price change": one(d, 1, 250),
        "Daily log return": one(lr, 1, 250),
        "Daily log return ÷ vol": one(vol_norm(lr), 1, 250),
        "Sign of daily return": one(np.sign(z), 1, 250),
        "Weekly sums of price change ÷ vol": one(z, 5, 52),
        "Monthly sums of price change ÷ vol": one(z, 21, 12),
        "Weekly log returns": one(lr, 5, 52),
        "Daily price change ÷ GARCH vol": one(zg, 1, 250),
        "Daily ÷ vol, × GARCH vol regime": [("base", z, 1, 250, None), ("× vol regime", z, 1, 250, v)],
        "Monthly ÷ vol, × GARCH vol regime": [("base", z, 21, 12, None), ("× vol regime", z, 21, 12, v),
                                             ("vol regime level", v, 1, 1, None)],
    }


def gamma(kind: str, L: int) -> np.ndarray:
    if kind == "ridge" or L < 3:
        return np.eye(L)
    D = np.zeros((L - 2, L))
    for i in range(L - 2):
        D[i, i:i + 3] = (1, -2, 1)
    return np.vstack([10 * D, 0.1 * np.eye(L)])


def block_gamma(kind: str, parts: list) -> np.ndarray:
    """Block-diagonal Γ: each block of columns gets its own penalty."""
    mats = [gamma(kind, p[3]) for p in parts]
    out = np.zeros((sum(m.shape[0] for m in mats), sum(m.shape[1] for m in mats)))
    r = c = 0
    for m in mats:
        out[r:r + m.shape[0], c:c + m.shape[1]] = m
        r, c = r + m.shape[0], c + m.shape[1]
    return out


def design(parts: list) -> dict:
    """Per asset: X (rows x total columns), one block per part."""
    assets = parts[0][1].columns
    out = {}
    for a in assets:
        cols = []
        for _, f, b, L, mult in parts:
            g = f[a].rolling(b, min_periods=b).sum() if b > 1 else f[a]
            m = 1.0 if mult is None else mult[a].to_numpy()
            cols += [g.shift(j * b).to_numpy() * m for j in range(L)]
        out[a] = np.column_stack(cols)
    return out


def year_blocks(X: dict, y: pd.DataFrame, years: np.ndarray, L: int) -> dict:
    blocks = {}
    for a, Xa in X.items():
        ya = y[a].to_numpy()
        ok = np.isfinite(ya) & np.isfinite(Xa).all(1)
        for year in np.unique(years[ok]):
            sel = np.where(ok & (years == year))[0]
            Xs, bs = Xa[sel], ya[sel]
            tail = slice(max(0, len(bs) - (H + 2)), None)
            blk = blocks.setdefault(int(year), {k: (np.zeros((L, L)) if "XtX" in k else np.zeros(L) if "Xty" in k else 0.0)
                                                for k in ("XtX", "Xty", "yty", "tXtX", "tXty", "tyty")})
            blk["XtX"] += Xs.T @ Xs
            blk["Xty"] += Xs.T @ bs
            blk["yty"] += bs @ bs
            blk["tXtX"] += Xs[tail].T @ Xs[tail]
            blk["tXty"] += Xs[tail].T @ bs[tail]
            blk["tyty"] += bs[tail] @ bs[tail]
    return blocks


def train_eq(blocks: dict, year: int, L: int):
    XtX, Xty = np.zeros((L, L)), np.zeros(L)
    for y_, b in blocks.items():
        if y_ < year:
            XtX += b["XtX"]
            Xty += b["Xty"]
    if year - 1 in blocks:
        XtX -= blocks[year - 1]["tXtX"]
        Xty -= blocks[year - 1]["tXty"]
    return XtX, Xty


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bt = Backtest(PRICES, MACRO)
    px = bt.prices.to_pandas().set_index("AsOfDate")
    px.index = pd.to_datetime(px.index)
    dates = px.index
    n = len(dates)
    years = dates.year.to_numpy()
    z = vol_norm(px.diff())
    y = sum(z.shift(-k) for k in range(2, 2 + H))
    val = (dates[int(n * .75)], dates[int(n * .9)])

    def trade(fc: np.ndarray) -> pd.Series:
        frame = pl.from_pandas(pd.DataFrame(fc, index=bt.prices["AsOfDate"].to_pandas(), columns=list(px.columns)).rename_axis("AsOfDate").reset_index())
        res = bt.run(lambda prices, *e: frame, period="full", check_leakage=False)  # each year uses earlier years only
        r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        return r

    def adj(r, a, b):
        x = r[(r.index >= a) & (r.index < b)].dropna().to_numpy()
        return float(adjusted_sharpe(x)) if len(x) > 100 else None

    t0 = time.time()
    sigma, vreg = garch(bt.prices, dates)
    print(f"  GARCH paths for {sigma.shape[1]} assets ({time.time() - t0:.0f}s)", flush=True)
    summary, curves, weights, yearly = [], [], [], []
    for name, parts in variants(px, sigma, vreg).items():
        t0 = time.time()
        _, _, b, Lb, _ = parts[0]
        X = design(parts)
        L = sum(p[3] for p in parts)  # total columns
        blocks = year_blocks(X, y, years, L)
        scale = np.trace(sum(bl["XtX"] for bl in blocks.values())) / L / sum(1 for _ in blocks)
        best = None
        for kind in ("ridge", "smooth"):
            GtG = block_gamma(kind, parts).T @ block_gamma(kind, parts)
            for alpha in ALPHAS:
                lam = alpha * scale
                fc = np.full((n, len(px.columns)), np.nan)
                num = den = 0.0
                for year in sorted(blocks):
                    if year < 2000:
                        continue
                    XtX, Xty = train_eq(blocks, year, L)
                    w = np.linalg.solve(XtX + lam * GtG, Xty)
                    rows = np.where(years == year)[0]
                    for k, a in enumerate(px.columns):
                        fc[rows, k] = X[a][rows] @ w
                    if DEV_BLOCKS[0] <= year <= DEV_BLOCKS[1]:
                        v = blocks[year]
                        num += w @ v["XtX"] @ w - 2 * w @ v["Xty"] + v["yty"]
                        den += v["yty"]
                r = trade(fc)
                dev_adj = adj(r, pd.Timestamp(f"{DEV_BLOCKS[0]}-01-01"), pd.Timestamp(f"{DEV_BLOCKS[1] + 1}-01-01"))
                row = {"variant": name, "gamma": kind, "alpha": float(alpha), "cv_err": num / den, "dev_blocks_adj": dev_adj,
                       "val_adj": adj(r, *val)}
                curves.append(row)
                if best is None or (dev_adj or -9) > (best[0]["dev_blocks_adj"] or -9):
                    best = (row, r, kind, lam)
        row, r, kind, lam = best
        XtX, Xty = train_eq(blocks, 2011, L)  # all dev years: the weights the page shows
        G = block_gamma(kind, parts)
        w = np.linalg.solve(XtX + lam * G.T @ G, Xty)
        split, c = {}, 0
        for pname, _, _, pL, _ in parts:
            split[pname] = w[c:c + pL]
            c += pL
        if "× vol regime" in split:  # effective filter w_base + v * w_vol at v = -1 and +1
            shown = {"calm (vol regime −1σ)": split["base"] - split["× vol regime"],
                     "turbulent (vol regime +1σ)": split["base"] + split["× vol regime"]}
        else:
            shown = {"base": split["base"]}
        for part, wp in shown.items():
            for j in range(Lb):
                weights.append({"variant": name, "part": part, "lag_days": j * b + (b - 1) / 2, "weight": float(wp[j]) / b})
        vol_coef = float(split["vol regime level"][0]) if "vol regime level" in split else None
        yr = r[(r.index.year >= 2000) & (r.index.year <= 2016)]
        for year, x in yr.groupby(yr.index.year):
            yearly.append({"variant": name, "year": int(year), "sharpe": float(sharpe(x.dropna().to_numpy()))})
        dev_years = [e["sharpe"] for e in yearly if e["variant"] == name and DEV_BLOCKS[0] <= e["year"] <= DEV_BLOCKS[1]]
        summary.append({"variant": name, "bucket_days": b, "lags": Lb, "columns": L, "window_days": b * Lb, "uses_garch": "GARCH" in name,
                        "vol_level_coef": vol_coef, "gamma": kind, "alpha": row["alpha"],
                        "cv_err": row["cv_err"], "dev_blocks_adj": row["dev_blocks_adj"], "val_adj": row["val_adj"],
                        "mean_yearly": float(np.mean(dev_years)), "years_positive": int(np.sum(np.array(dev_years) > 0)),
                        "years": len(dev_years)})
        print(f"  {time.time() - t0:4.0f}s  {name:<38} Γ={kind:<6} α={row['alpha']:.2g}  dev blocks {row['dev_blocks_adj']:+.2f}  "
              f"validation {row['val_adj']:+.2f}  cv err {row['cv_err']:.5f}", flush=True)

    code = Path(__file__).read_text()
    meta = [{"label": "tikhonov_features.py", "value": code, "type": "code", "lang": "python"},
            {"label": "Setup", "value": f"Target: next {H} days' vol-normalised move. Dev blocks {DEV_BLOCKS[0]}-{DEV_BLOCKS[1]} (walk-forward, yearly refits). "
                                       f"Validation {val[0].date()} to {val[1].date()}. Test block not used."}]
    r6 = lambda v: round(v, 6) if isinstance(v, float) else v  # noqa: E731
    for fname, rows in {"summary": summary, "curves": curves, "weights": weights, "yearly": yearly}.items():
        (OUT / f"{fname}.json").write_text(json.dumps({"rows": [{k: r6(v) for k, v in x.items()} for x in rows], "meta": meta}))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
