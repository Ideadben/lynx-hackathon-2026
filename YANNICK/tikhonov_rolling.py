"""Would a rolling window help the Tikhonov lag filter?

Same model as src/strategies/tikhonov.py (L = 250, H = 5, smooth Γ, α = 1000), refit each January,
but trained on:
    expanding     all earlier years (the current strategy)
    rolling N     only the last N years (N = 3, 5, 8, 10)
    half-life h   all earlier years, each row weighted 0.5 ** (age in years / h)  (h = 2, 4, 8)
Every variant starts trading in 2000 (needs 5 years for the first fit), so all are compared on the
same days. Scored: mean Sharpe per year 2000-2010 and 2011-2016, adjusted Sharpe on the dev blocks
(2003-2010) and on validation (Jun 2011 - Oct 2014). The test block is not used.

Run: uv run python YANNICK/tikhonov_rolling.py
"""

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from strategies import tikhonov as tk
from tikhonov_overfit import prepare


def walk_forward(X, y, usable, years, window=None, half_life=None, alpha=tk.ALPHA, start_year=2000):
    """Forecast for every row, refitting each January on the chosen training rows."""
    G = tk.smooth_gamma(tk.L)
    GtG = G.T @ G
    gap = tk.H + 2
    out = np.full(y.shape, np.nan)
    betas = {}
    for year in sorted(set(years)):
        if year < start_year:
            continue
        start = int(np.argmax(years == year))
        rows = np.arange(0, start - gap)
        if window is not None:
            rows = rows[years[rows] >= year - window]
        mask = usable[rows]
        Xr, yr = X[rows][mask], y[rows][mask]
        if half_life is not None:
            age = np.broadcast_to((year - years[rows])[:, None], mask.shape)[mask]  # age of each training row, in years
            sw = 0.5 ** (age / half_life)
        else:
            sw = np.ones(len(yr))
        XtX = (Xr * sw[:, None]).T @ Xr
        n_years = max(1.0, sw.sum() / max(1, mask.sum() / max(1, len(set(years[rows])))))
        lam = alpha * np.trace(XtX) / tk.L / n_years
        w = np.linalg.solve(XtX + lam * GtG, (Xr * sw[:, None]).T @ yr)
        betas[int(year)] = w
        rows_now = years == year
        out[rows_now] = X[rows_now] @ w
    return out, betas


def main() -> None:
    pd.set_option("display.width", 200)
    bt = Backtest(PRICES, MACRO)
    X, y, usable, years = prepare(bt.prices)
    assets = [c for c in bt.prices.columns if c != "AsOfDate"]
    dates = pd.DatetimeIndex(pd.to_datetime(bt.prices["AsOfDate"].to_pandas()))
    n = len(dates)
    val = (dates[int(n * .75)], dates[int(n * .9)])

    def score(fc):
        frame = bt.prices.select("AsOfDate").with_columns(pl.Series(a, fc[:, i], nan_to_null=True) for i, a in enumerate(assets))
        r = bt.run(lambda prices, *e: frame, period="full", check_leakage=False).returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        yearly = r[r.index.year >= 2000].groupby(r[r.index.year >= 2000].index.year).apply(lambda x: float(sharpe(x.dropna().to_numpy())))
        cut = lambda a, b: float(adjusted_sharpe(r[(r.index >= a) & (r.index < b)].dropna().to_numpy()))  # noqa: E731
        return {"mean/yr 2000-10": yearly[yearly.index <= 2010].mean(), "mean/yr 2011-16": yearly[yearly.index >= 2011].mean(),
                "dev blocks adj": cut(pd.Timestamp("2003-01-01"), pd.Timestamp("2011-01-01")), "validation adj": cut(*val),
                "worst yr": yearly.min()}, yearly

    rows, yearly_all, betas_all = {}, {}, {}
    variants = {"expanding (current)": {}}
    variants.update({f"rolling {w} years": {"window": w} for w in (3, 5, 8, 10)})
    variants.update({f"half-life {h} years": {"half_life": h} for h in (2, 4, 8)})
    for name, kw in variants.items():
        fc, betas = walk_forward(X, y, usable, years, **kw)
        rows[name], yearly_all[name] = score(fc)
        betas_all[name] = betas
        b = np.array([betas[k] for k in sorted(betas)])
        rows[name]["beta corr yr-to-yr"] = float(np.mean([np.corrcoef(b[i], b[i - 1])[0, 1] for i in range(1, len(b))]))
    print("Training window for the yearly refit (same α, Γ, lags; all trade from 2000):")
    print(pd.DataFrame(rows).T.round(2).to_string())
    print("\nSharpe per year:")
    print(pd.DataFrame(yearly_all).T.round(1).to_string())

    import json
    from pathlib import Path

    out = Path(__file__).resolve().parent / "output" / "tikhonov_rolling"
    out.mkdir(parents=True, exist_ok=True)
    meta = [{"label": "tikhonov_rolling.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"}]
    summary = [{"variant": k, **{c.replace(" ", "_").replace("/", "_per_").replace("-", "_"): float(v) for c, v in r.items()}} for k, r in rows.items()]
    yearly = [{"variant": k, "year": int(y_), "sharpe": float(v)} for k, s in yearly_all.items() for y_, v in s.items()]
    for name, data in {"summary": summary, "yearly": yearly}.items():
        (out / f"{name}.json").write_text(json.dumps({"rows": data, "meta": meta}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
