"""Non-overlapping split: dev (train) 75% | validation 15% | test 10%.

    dev         1995-01-02 .. 2011-06-27   fit weights / coefficients
    validation  2011-06-28 .. 2014-10-14   choose the hyperparameter (Tikhonov λ and Γ, VAR lag p)
    test        2014-10-15 .. 2016-12-30   refit on dev + validation with the chosen setting, score once

Between blocks, rows whose labels reach into the next block are dropped when fitting.

Part A: Tikhonov lag filter (prediction = sum_j w_j z[t-j], 250 lags, pooled over 27 assets).
        The L-curve (least-squares error vs ||Γw||²) and the train/validation error per λ.
Part B: daily VAR on the 10 stock and bond series, forecasting the return a position can earn
        (two steps ahead), traded on the followers. The lag p = 1..10 is tuned on validation.

Writes YANNICK/output/split_v2.json (for the scorecard) and YANNICK/output/split_v2_tikhonov.png.
Run: uv run python YANNICK/split_v2.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import lag_tikhonov as lt
from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from strategies.var_model import VARmodel
from var_strategy import FOLLOWERS, SYSTEM

OUT = Path(__file__).resolve().parent / "output"
SPLIT = (0.75, 0.90)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"


def window_sharpe(bt: Backtest, f, a, b, dates) -> tuple[float, float]:
    res = bt.run(f, period="full", check_leakage=False)
    r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
    r.index = pd.to_datetime(r.index)
    end = dates[b] if b < len(dates) else dates[-1] + pd.Timedelta(days=1)
    x = r[(r.index >= dates[a]) & (r.index < end)].dropna().to_numpy()
    return float(adjusted_sharpe(x)), float(sharpe(x))


# ------------------------------------------------------------------------------------- Part A
def tikhonov_part(bt: Backtest, dates, n) -> dict:
    import matplotlib.pyplot as plt

    px = bt.prices.to_pandas().set_index("AsOfDate")
    z = lt.z_returns(px)
    y = sum(z.shift(-k) for k in range(2, 2 + lt.H))
    gap = lt.H + 2
    d_end, v_end = int(n * SPLIT[0]), int(n * SPLIT[1])
    nm = {"train": lt.build(z, y, np.arange(0, d_end - gap)),
          "val": lt.build(z, y, np.arange(d_end, v_end - gap)),
          "train+val": lt.build(z, y, np.arange(0, v_end - gap)),
          "test": lt.build(z, y, np.arange(v_end, n - gap))}
    base = {k: v[2] / v[3] for k, v in nm.items()}
    curves, best = {}, {}
    for kind in ("ridge", "smooth"):
        G = lt.gamma(kind)
        XtX, Xty, yty, cnt = nm["train"]
        rows = []
        for lam in lt.LAMS:
            w = np.linalg.solve(XtX + lam * G.T @ G, Xty)
            rows.append({"lam": float(lam), "lse": lt.mse(w, XtX, Xty, yty, cnt) / base["train"],
                         "wnorm": float((G @ w) @ (G @ w)), "val": lt.mse(w, *nm["val"]) / base["val"], "w": w})
        curves[kind] = rows
        best[kind] = min(range(len(rows)), key=lambda i: rows[i]["val"])
    kind = min(best, key=lambda k: curves[k][best[k]]["val"])
    lam = curves[kind][best[kind]]["lam"]
    G = lt.gamma(kind)
    XtX, Xty, _, _ = nm["train+val"]
    w = np.linalg.solve(XtX + lam * G.T @ G, Xty)
    test_err = lt.mse(w, *nm["test"]) / base["test"]

    def learned(prices, *extras):
        return lt.filter_forecast(prices, w)

    test_adj, test_sh = window_sharpe(bt, learned, v_end, n, dates)
    print(f"Tikhonov: chosen on validation Γ = {kind}, λ = {lam:.3g}; validation error {curves[kind][best[kind]]['val']:.5f} "
          f"(least squares {curves[kind][0]['val']:.5f}); test error {test_err:.5f}; test adj Sharpe {test_adj:+.2f} (Sharpe {test_sh:+.2f})")

    # figure: L-curve | error vs λ | weights
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURF)
    for a_ in ax:
        a_.set_facecolor(SURF)
        a_.grid(True, color=GRID, linewidth=0.6)
        for s_ in a_.spines.values():
            s_.set_visible(False)
        a_.tick_params(colors=INK2, labelsize=8)
    for k, col in (("ridge", S1), ("smooth", S2)):
        c, b = curves[k], best[k]
        ax[0].plot([r["lse"] for r in c], [r["wnorm"] for r in c], color=col, linewidth=2, label=f"Γ = {k}")
        ax[0].scatter([c[b]["lse"]], [c[b]["wnorm"]], s=55, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        ax[0].annotate(f"λ = {c[b]['lam']:.2g}", (c[b]["lse"], c[b]["wnorm"]), textcoords="offset points", xytext=(8, 6), color=INK, fontsize=8)
        ax[1].semilogx(lt.LAMS, [r["val"] for r in c], color=col, linewidth=2, label=f"Γ = {k}, validation")
        ax[1].semilogx(lt.LAMS, [r["lse"] for r in c], color=col, linewidth=1, linestyle="--", label=f"Γ = {k}, train (dev)")
        ax[1].scatter([c[b]["lam"]], [c[b]["val"]], s=55, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        ax[2].plot(np.arange(lt.L), c[b]["w"], color=col, linewidth=2, label=f"Γ = {k}, chosen λ")
    ax[2].plot(np.arange(lt.L), curves["ridge"][0]["w"], color=GRID, linewidth=1, label="least squares (λ ≈ 0)")
    ax[0].set_yscale("log")
    ax[0].set_title("L-curve: least-squares error vs weight size", color=INK, fontsize=11, loc="left")
    ax[0].set_xlabel("least-squares error on dev / error of predicting 0", color=INK2, fontsize=9)
    ax[0].set_ylabel("weight size ‖Γw‖²", color=INK2, fontsize=9)
    ax[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    ax[1].axhline(1, color=INK2, linewidth=0.8)
    ax[1].set_title("Error by λ (dot = chosen on validation)", color=INK, fontsize=11, loc="left")
    ax[1].set_xlabel("λ", color=INK2, fontsize=9)
    ax[1].set_ylabel("error / error of predicting 0", color=INK2, fontsize=9)
    ax[1].legend(frameon=False, fontsize=7, labelcolor=INK)
    ax[2].axhline(0, color=INK2, linewidth=0.8)
    lim = 1.6 * max(abs(np.array(curves[kind][best[kind]]["w"])).max(), 1e-9)
    ax[2].set_ylim(-lim, lim)
    ax[2].set_title("Weight on each past day's return", color=INK, fontsize=11, loc="left")
    ax[2].set_xlabel("lag j (days ago)", color=INK2, fontsize=9)
    ax[2].legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.suptitle("Tikhonov lag filter, split 75 / 15 / 10 (dev / validation / test)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "split_v2_tikhonov.png", dpi=120)

    return {
        "curve": [{"gamma": k, "lam": r["lam"], "lse": r["lse"], "wnorm": r["wnorm"], "val_err": r["val"],
                   "chosen": i == best[k]} for k in curves for i, r in enumerate(curves[k])],
        "weights": [{"lag": j, "ridge": float(curves["ridge"][best["ridge"]]["w"][j]),
                     "smooth": float(curves["smooth"][best["smooth"]]["w"][j]),
                     "ols": float(curves["ridge"][0]["w"][j])} for j in range(lt.L)],
        "summary": {"gamma": kind, "lam": lam, "val_err": curves[kind][best[kind]]["val"],
                    "val_err_ols": curves[kind][0]["val"], "test_err": test_err, "test_adj": test_adj, "test_sharpe": test_sh},
    }


# ------------------------------------------------------------------------------------- Part B
def var_fit(lr: np.ndarray, end: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    m = VARmodel(num_vars=lr.shape[1], max_lag=p)
    m.load_training_data(lr[1:end])
    m.fit()
    return m.intercepts[p - 1], m.coefs[p - 1]


def var_forecast(lr: np.ndarray, c: np.ndarray, A: np.ndarray, steps: int = 2) -> np.ndarray:
    """Vectorised `steps`-ahead forecast for every day t from returns up to t (NaN before p rows)."""
    n, N = lr.shape
    p = A.shape[0]
    x = np.nan_to_num(lr)
    hist = [np.vstack([np.full((m, N), np.nan), x[: n - m]]) for m in range(p)]  # hist[m][t] = x[t-m]
    preds = []
    for _ in range(steps):
        nxt = c + sum(hist[m] @ A[m].T for m in range(p))
        preds.append(nxt)
        hist = [nxt] + hist[:-1]
    return preds[-1]


def var_part(bt: Backtest, dates, n) -> dict:
    px = bt.prices.to_pandas().set_index("AsOfDate")
    lr_df = np.log(px[SYSTEM]).diff()
    lr = lr_df.to_numpy()
    vol = lr_df.rolling(100, min_periods=50).std().to_numpy()
    d_end, v_end = int(n * SPLIT[0]), int(n * SPLIT[1])
    fi = [SYSTEM.index(a) for a in FOLLOWERS]
    target = np.vstack([lr[2:], np.full((2, lr.shape[1]), np.nan)])  # return earned by a position on t: r[t+2]

    def positions_from(fc: np.ndarray):
        def f(prices, *extras):
            frame = pd.DataFrame(0.0, index=px.index, columns=[c for c in prices.columns if c != "AsOfDate"])
            for a in FOLLOWERS:
                frame[a] = fc[:, SYSTEM.index(a)] / vol[:, SYSTEM.index(a)]
            frame.iloc[:200] = np.nan
            return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())
        return f

    def rel_err(fc, a, b):
        e = (fc[a:b, fi] - target[a:b, fi]) ** 2
        z0 = target[a:b, fi] ** 2
        ok = np.isfinite(e) & np.isfinite(z0)
        return float(e[ok].sum() / z0[ok].sum())

    rows = []
    bic = None
    for p in range(1, 11):
        c, A = var_fit(lr, d_end, p)
        fc = var_forecast(lr, c, A)
        val_adj, val_sh = window_sharpe(bt, positions_from(fc), d_end, v_end, dates)
        rows.append({"p": p, "val_err": rel_err(fc, d_end, v_end - 2), "val_adj": val_adj, "val_sharpe": val_sh,
                     "dev_err": rel_err(fc, 200, d_end - 2)})
        print(f"  VAR p={p:>2}: validation error {rows[-1]['val_err']:.5f}  validation adj Sharpe {val_adj:+.2f}", flush=True)
    m = VARmodel(num_vars=lr.shape[1], max_lag=10)
    m.load_training_data(lr[1:d_end])
    m.fit()
    bic = m.optimal_lag(verbose=False)
    p_err = min(rows, key=lambda r: r["val_err"])["p"]
    p_sh = max(rows, key=lambda r: r["val_adj"])["p"]
    out = {"lags": rows, "bic_p": bic, "chosen_p_by_error": p_err, "chosen_p_by_sharpe": p_sh, "test": {}}
    for label, p in (("by validation error", p_err), ("by validation Sharpe", p_sh), ("by BIC on dev", bic)):
        if label != "by validation error" and p in [v["p"] for v in out["test"].values()]:
            continue
        c, A = var_fit(lr, v_end, p)
        fc = var_forecast(lr, c, A)
        t_adj, t_sh = window_sharpe(bt, positions_from(fc), v_end, n, dates)
        out["test"][label] = {"p": p, "test_adj": t_adj, "test_sharpe": t_sh, "test_err": rel_err(fc, v_end, n - 2)}
        print(f"  test, p chosen {label} (p = {p}): adj Sharpe {t_adj:+.2f}  Sharpe {t_sh:+.2f}  error {out['test'][label]['test_err']:.5f}")
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bt = Backtest(PRICES, MACRO)
    dates = pd.DatetimeIndex(pd.to_datetime(bt.prices["AsOfDate"].to_pandas()))
    n = len(dates)
    print(f"dev {dates[0].date()}..{dates[int(n * .75) - 1].date()} | validation {dates[int(n * .75)].date()}.."
          f"{dates[int(n * .9) - 1].date()} | test {dates[int(n * .9)].date()}..{dates[n - 1].date()}\n")
    tik = tikhonov_part(bt, dates, n)
    print("\nVAR lag tuning (fit on dev, chosen on validation):")
    var = var_part(bt, dates, n)
    (OUT / "split_v2.json").write_text(json.dumps({"tikhonov": tik, "var": var}, indent=1))
    print(f"\nwrote {OUT / 'split_v2.json'} and {OUT / 'split_v2_tikhonov.png'}")


if __name__ == "__main__":
    main()
