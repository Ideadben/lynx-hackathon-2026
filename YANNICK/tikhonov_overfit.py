"""Does the Tikhonov lag filter overfit? And its betas (the filter weights) for every yearly refit.

Uses src/strategies/tikhonov.py exactly (L = 250, H = 5, smooth Γ, α = 1000), without GARCH.

1. Effective number of parameters per refit: df = trace((X'X + λΓ'Γ)^-1 X'X), out of 250.
2. Prediction error relative to predicting zero, in sample (the refit's training rows) and out of
   sample (the year it then trades). A big gap = overfitting.
3. Beta stability: correlation of each refit's betas with the previous refit's.
4. In-sample control: one fit on ALL years, traded on all years. If it also fades after 2011,
   the signal weakened (non-stationarity), which no amount of regularisation fixes.
Also: the same with α = 0.001 (almost plain least squares), to show what overfitting looks like.

Writes YANNICK/output/tikhonov_overfit.png and YANNICK/output/tikhonov_overfit/*.json.
Run: uv run python YANNICK/tikhonov_overfit.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import sharpe
from data import MACRO, PRICES
from strategies import tikhonov as tk

OUT = Path(__file__).resolve().parent / "output"


def prepare(prices: pl.DataFrame, lags=tk.L, horizon=tk.H):
    z = tk.z_returns(prices)
    n, m = z.shape
    years = np.array([d.year for d in prices["AsOfDate"].to_list()])
    pad = lambda k: np.vstack([z[k:], np.full((k, m), np.nan)])  # noqa: E731
    y = sum(pad(k) for k in range(2, 2 + horizon))
    bad = np.cumsum(~np.isfinite(z), axis=0)
    bad_before = np.vstack([np.zeros((lags, m)), bad[:-lags]])
    have = (bad - bad_before == 0) & (np.arange(n) >= lags - 1)[:, None]
    X = tk._lagged(np.nan_to_num(z), lags)
    return X, y, have & np.isfinite(y), years


def fit(X, y, usable, years, rows, alpha, lags=tk.L):
    mask = usable[rows]
    Xr, yr = X[rows][mask], y[rows][mask]
    XtX = Xr.T @ Xr
    G = tk.smooth_gamma(lags)
    lam = alpha * np.trace(XtX) / lags / max(1, len(set(years[rows])))
    A = XtX + lam * G.T @ G
    w = np.linalg.solve(A, Xr.T @ yr)
    df = float(np.trace(np.linalg.solve(A, XtX)))
    return w, df


def rel_err(X, y, usable, rows, w):
    mask = usable[rows]
    Xr, yr = X[rows][mask], y[rows][mask]
    return float(((Xr @ w - yr) ** 2).sum() / (yr**2).sum())


def main() -> None:
    import matplotlib.pyplot as plt

    bt = Backtest(PRICES, MACRO)
    X, y, usable, years = prepare(bt.prices)
    gap = tk.H + 2
    first = years[0] + tk.MIN_YEARS
    rows_out, betas = [], []
    prev = {}
    W_full, df_full = fit(X, y, usable, years, np.arange(len(years) - gap), tk.ALPHA)  # in-sample control
    for alpha, label in ((tk.ALPHA, "tikhonov"), (1e-3, "near least squares")):
        prev_w = None
        for year in sorted(set(years)):
            if year < first:
                continue
            start = int(np.argmax(years == year))
            train = np.arange(0, start - gap)
            test = np.where(years == year)[0]
            test = test[test < len(years) - gap]
            w, df = fit(X, y, usable, years, train, alpha)
            rows_out.append({"model": label, "year": int(year), "df": df, "train_err": rel_err(X, y, usable, train, w),
                             "test_err": rel_err(X, y, usable, test, w),
                             "beta_corr_prev": float(np.corrcoef(w, prev_w)[0, 1]) if prev_w is not None else None})
            prev_w = w
            if label == "tikhonov":
                betas += [{"refit": int(year), "lag": j, "beta": float(w[j])} for j in range(len(w))]
    for j in range(len(W_full)):
        betas.append({"refit": "all years (in-sample)", "lag": j, "beta": float(W_full[j])})
    diag = pd.DataFrame(rows_out)

    # Trading: walk-forward (the strategy) vs one in-sample fit on all years
    def yearly(fc_fn):
        r = bt.run(fc_fn, period="full", check_leakage=False).returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        r = r[r.index.year >= first]
        return r.groupby(r.index.year).apply(lambda x: float(sharpe(x.dropna().to_numpy())))

    assets = [c for c in bt.prices.columns if c != "AsOfDate"]
    in_sample = X @ W_full
    in_sample[years < first] = np.nan

    def in_sample_fc(prices, *extras):
        return prices.select("AsOfDate").with_columns(pl.Series(a, in_sample[:, i], nan_to_null=True) for i, a in enumerate(assets))

    sh = pd.DataFrame({"walk-forward (the strategy)": yearly(tk.forecast), "one fit on all years (in-sample)": yearly(in_sample_fc)})

    pd.set_option("display.width", 200)
    print("Per refit: effective parameters (of 250), error vs predicting 0 in and out of sample, beta stability")
    t = diag.pivot(index="year", columns="model", values=["df", "train_err", "test_err", "beta_corr_prev"])
    print(t.round(4).to_string())
    for label in ("tikhonov", "near least squares"):
        d = diag[diag["model"] == label]
        print(f"\n{label}: mean df {d['df'].mean():.1f} | mean train error {d['train_err'].mean():.4f} | "
              f"mean out-of-sample error {d['test_err'].mean():.4f} | gap {d['test_err'].mean() - d['train_err'].mean():+.4f} | "
              f"mean beta correlation with previous refit {d['beta_corr_prev'].mean():.2f}")
    print(f"\nIn-sample control: one fit on all years has df {df_full:.1f}")
    print("\nSharpe per year: walk-forward vs one fit on all years (in-sample)")
    print(sh.round(2).T.to_string())
    before, after = sh[sh.index <= 2010].mean(), sh[sh.index >= 2011].mean()
    print("\nmean 2000-2010:", before.round(2).to_dict(), "\nmean 2011-2016:", after.round(2).to_dict())

    # Figure: betas per refit | error in vs out of sample | Sharpe walk-forward vs in-sample
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    S1, S2 = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURF)
    for a in ax:
        a.set_facecolor(SURF)
        a.grid(True, color=GRID, linewidth=0.6)
        for s in a.spines.values():
            s.set_visible(False)
        a.tick_params(colors=INK2, labelsize=8)
    b = pd.DataFrame([x for x in betas if x["refit"] != "all years (in-sample)"])
    yrs = sorted(b["refit"].unique())
    cmap = plt.get_cmap("Blues")
    for i, yr in enumerate(yrs):
        bb = b[b["refit"] == yr]
        ax[0].plot(bb["lag"], bb["beta"], color=cmap(0.3 + 0.7 * i / max(1, len(yrs) - 1)), linewidth=1.2)
    ax[0].plot(range(len(W_full)), W_full, color=S2, linewidth=2, linestyle="--", label="one fit on all years")
    ax[0].axhline(0, color=INK2, linewidth=0.8)
    ax[0].set_title(f"Betas per yearly refit ({yrs[0]} light → {yrs[-1]} dark)", color=INK, fontsize=11, loc="left")
    ax[0].set_xlabel("lag j (days ago)", color=INK2, fontsize=9)
    ax[0].set_ylabel("beta (weight on z[t − j])", color=INK2, fontsize=9)
    ax[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    for label, col, ls in (("tikhonov", S1, "-"), ("near least squares", S2, "--")):
        d = diag[diag["model"] == label]
        ax[1].plot(d["year"], d["train_err"], color=col, linewidth=1, linestyle=":", marker="o", markersize=3, label=f"{label}, in sample")
        ax[1].plot(d["year"], d["test_err"], color=col, linewidth=2, linestyle=ls, marker="o", markersize=4, label=f"{label}, out of sample")
    ax[1].axhline(1, color=INK2, linewidth=0.8)
    ax[1].set_title("Error vs predicting 0 (below 1 = useful)", color=INK, fontsize=11, loc="left")
    ax[1].set_xlabel("refit year", color=INK2, fontsize=9)
    ax[1].legend(frameon=False, fontsize=7, labelcolor=INK)
    for colname, col, ls in ((sh.columns[0], S1, "-"), (sh.columns[1], S2, "--")):
        ax[2].plot(sh.index, sh[colname], color=col, linewidth=2, linestyle=ls, marker="o", markersize=4, label=colname)
    ax[2].axhline(0, color=INK2, linewidth=0.8)
    ax[2].axvspan(2010.5, sh.index.max() + 0.5, color=GRID, alpha=0.5)
    ax[2].set_title("Sharpe per year (shaded: after dev)", color=INK, fontsize=11, loc="left")
    ax[2].set_xlabel("year", color=INK2, fontsize=9)
    for a in ax[1:]:
        a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax[2].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "tikhonov_overfit.png", dpi=120)

    out = OUT / "tikhonov_overfit"
    out.mkdir(exist_ok=True)
    meta = [{"label": "tikhonov_overfit.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"}]
    sh_rows = [{"year": int(y_), "model": m, "sharpe": float(v)} for m in sh.columns for y_, v in sh[m].items()]
    for name, rows in {"betas": betas, "diagnostics": diag.to_dict("records"), "yearly": sh_rows}.items():
        clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]
        (out / f"{name}.json").write_text(json.dumps({"rows": clean, "meta": meta}))
    print(f"\nwrote {OUT / 'tikhonov_overfit.png'} and {out}")


if __name__ == "__main__":
    main()
