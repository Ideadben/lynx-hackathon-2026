"""Data for the Tikhonov strategy dashboard (src/strategies/tikhonov.py), plus the VAR's hit rate.

Settings: α = 46 (chosen on validation), α = 46 + GARCH vol timing, α = 1000 (the old default).
All walk-forward (refit each January on earlier years). Windows: dev = first 75% of dates,
validation = next 15%, test = last 10% (scored here once for α = 46).

Hit rate = share of (asset, day) pairs where sign(forecast) = sign(outcome), zero forecasts skipped:
    next earned day   outcome = z[t+2], the one-day return the position decided on day t earns first
    next 5 days       outcome = z[t+2] + ... + z[t+6], what the filter predicts
VAR (YANNICK/var_strategy.py, daily VAR on stocks and bonds, forecast two days ahead, followers only):
    outcome = log return on day t+2.

Writes YANNICK/output/tikhonov_dashboard/*.json.
Run: uv run python YANNICK/tikhonov_dashboard.py
"""

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from strategies import tikhonov as tk

OUT = Path(__file__).resolve().parent / "output" / "tikhonov_dashboard"
CONFIGS = {
    "α = 46": partial(tk.forecast, alpha=46.0),
    "α = 46 + GARCH": partial(tk.forecast_garch, alpha=46.0),
    "α = 1000 (old default)": partial(tk.forecast, alpha=1000.0),
}


def windows(dates: pd.DatetimeIndex) -> dict:
    n = len(dates)
    end = dates[-1] + pd.Timedelta(days=1)
    return {"dev": (dates[0], dates[int(n * .75)]), "validation": (dates[int(n * .75)], dates[int(n * .9)]),
            "test": (dates[int(n * .9)], end), "all trading (2000+)": (pd.Timestamp("2000-01-01"), end)}


def hit(fc: np.ndarray, out: np.ndarray, rows: np.ndarray) -> tuple[float, int]:
    f, o = fc[rows].ravel(), out[rows].ravel()
    ok = np.isfinite(f) & np.isfinite(o) & (f != 0) & (o != 0)
    return (float(np.mean(np.sign(f[ok]) == np.sign(o[ok]))) if ok.any() else float("nan")), int(ok.sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bt = Backtest(PRICES, MACRO)
    dates = pd.DatetimeIndex(pd.to_datetime(bt.prices["AsOfDate"].to_pandas()))
    W = windows(dates)
    assets = [c for c in bt.prices.columns if c != "AsOfDate"]
    z = tk.z_returns(bt.prices)
    m = z.shape[1]
    pad = lambda k: np.vstack([z[k:], np.full((k, m), np.nan)])  # noqa: E731  row t holds z[t+k]
    next_day, next5 = pad(2), sum(pad(k) for k in range(2, 2 + tk.H))

    summary, curve, yearly, per_asset, calib = [], [], [], [], []
    for name, f in CONFIGS.items():
        t0 = time.time()
        res = bt.run(f, period="full", check_leakage=(name == "α = 46"), name=name)
        fc = f(bt.prices, bt.extras).select(assets).to_numpy().astype(float)
        r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        for w, (a, b) in W.items():
            x = r[(r.index >= a) & (r.index < b)].dropna()
            rows = np.where((dates >= a) & (dates < b))[0]
            h1, n1 = hit(fc, next_day, rows)
            h5, _ = hit(fc, next5, rows)
            summary.append({"config": name, "window": w, "adj_sharpe": float(adjusted_sharpe(x.to_numpy())),
                            "sharpe": float(sharpe(x.to_numpy())), "skew": float(x.skew()), "excess_kurtosis": float(x.kurt()),
                            "share_positive_days": float((x > 0).mean()), "hit_next_day": h1, "hit_next_5": h5, "forecasts": n1})
        # Weekly-sampled curve: cumulative net P&L, drawdown, rolling one-year Sharpe
        rr = r[r.index >= pd.Timestamp("2000-01-01")].fillna(0.0)
        cum = rr.cumsum()
        dd = cum - cum.cummax()
        roll = rr.rolling(252).mean() / rr.rolling(252).std() * np.sqrt(252)
        for i in range(0, len(cum), 5):
            curve.append({"config": name, "date": cum.index[i].strftime("%Y-%m-%d"), "cum_pnl": float(cum.iloc[i]),
                          "drawdown": float(dd.iloc[i]), "rolling_1y_sharpe": None if np.isnan(roll.iloc[i]) else float(roll.iloc[i])})
        for y_, x in rr.groupby(rr.index.year):
            yearly.append({"config": name, "year": int(y_), "sharpe": float(sharpe(x.to_numpy()))})
        # Per instrument (2000+) and calibration by forecast strength, for the plain α = 46 filter
        if name == "α = 46":
            gross = res.by_instrument().to_pandas().set_index("instrument")["sharpe_ex_slippage"]
            oos = np.where(dates >= pd.Timestamp("2000-01-01"))[0]
            for k, a in enumerate(assets):
                h1, n1 = hit(fc[:, [k]], next_day[:, [k]], oos)
                h5, _ = hit(fc[:, [k]], next5[:, [k]], oos)
                per_asset.append({"instrument": a, "gross_sharpe": float(gross[a]), "hit_next_day": h1, "hit_next_5": h5})
            # strength = |forecast| / its trailing one-year std per asset (past data only), deciles pooled
            fdf = pd.DataFrame(fc, index=dates)
            strength = (fdf.abs() / fdf.rolling(252, min_periods=126).std().shift(1)).to_numpy()
            ok = np.isfinite(strength) & np.isfinite(fc) & np.isfinite(next5) & (fc != 0)
            ok[: np.argmax(dates >= pd.Timestamp("2000-01-01"))] = False
            s, sg, o5, o1 = strength[ok], np.sign(fc[ok]), next5[ok], next_day[ok]
            qs = np.quantile(s, np.linspace(0, 1, 11))
            for qi in range(10):
                sel = (s >= qs[qi]) & (s <= qs[qi + 1])
                n5, n1 = sel & (o5 != 0), sel & np.isfinite(o1) & (o1 != 0)  # flat outcomes skipped, as in hit()
                calib.append({"decile": qi + 1, "strength_mid": float(np.median(s[sel])),
                              "hit_next_5": float(np.mean(sg[n5] == np.sign(o5[n5]))),
                              "hit_next_day": float(np.mean(sg[n1] == np.sign(o1[n1]))),
                              "mean_signed_next_5": float(np.mean(sg[sel] * o5[sel]))})
        print(f"  {name}: {time.time() - t0:.0f}s", flush=True)

    # Betas per yearly refit, α = 46, and the last α = 1000 refit for comparison
    betas = []
    for year, w in tk.weights_by_year(bt.prices, alpha=46.0).items():
        betas += [{"fit": str(year), "lag": j, "beta": float(w[j])} for j in range(len(w))]
    w1000 = tk.weights_by_year(bt.prices, alpha=1000.0)
    last = max(w1000)
    betas += [{"fit": f"α = 1000, {last} refit", "lag": j, "beta": float(w1000[last][j])} for j in range(len(w1000[last]))]

    # VAR hit rate (followers, forecast two days ahead)
    from var_strategy import FOLLOWERS, var_daily

    vfc = var_daily(bt.prices).select(FOLLOWERS).to_numpy().astype(float)
    lr = np.log(bt.prices.select(FOLLOWERS).to_numpy().astype(float))
    lr = np.vstack([np.full((1, len(FOLLOWERS)), np.nan), np.diff(lr, axis=0)])
    v_out = np.vstack([lr[2:], np.full((2, len(FOLLOWERS)), np.nan)])
    var_rows = []
    for w, (a, b) in W.items():
        rows = np.where((dates >= a) & (dates < b))[0]
        h, nn = hit(vfc, v_out, rows)
        var_rows.append({"window": w, "hit_rate": h, "forecasts": nn})
    for k, a in enumerate(FOLLOWERS):
        h, nn = hit(vfc[:, [k]], v_out[:, [k]], np.arange(len(dates)))
        var_rows.append({"window": f"all, {a}", "hit_rate": h, "forecasts": nn})

    meta = [{"label": "tikhonov_dashboard.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"},
            {"label": "Strategy", "value": "src/strategies/tikhonov.py: L = 250 lags, H = 5, second-order Γ, expanding window, yearly refit"}]
    clean = lambda rows: [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]  # noqa: E731
    for name, rows in {"summary": summary, "curve": curve, "yearly": yearly, "per_asset": per_asset, "calibration": calib,
                       "betas": betas, "var_hit": var_rows}.items():
        (OUT / f"{name}.json").write_text(json.dumps({"rows": clean(rows), "meta": meta}))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary).pivot(index="config", columns="window", values=["adj_sharpe", "hit_next_day", "hit_next_5"]).round(3).to_string())
    print("\nVAR hit rate:\n" + pd.DataFrame(var_rows).round(3).to_string(index=False))
    print("\ncalibration:\n" + pd.DataFrame(calib).round(3).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
