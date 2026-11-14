"""The strategy as a linear fit of previous returns, kept from overfitting by Tikhonov regularisation.

    prediction[t] = sum_j w[j] * z[t - j],   j = 0 .. L-1          (z = vol-normalised daily return)
    minimise   ||X w - y||^2 + lam * ||G w||^2                       (strategy return - actual return)

y is the return a position decided on day t actually earns: z[t+2] + ... + z[t+1+H].
One filter w is shared by all 27 assets (pooled), so there is enough data for L = 250 weights.

G, the Tikhonov matrix:
    "ridge":  identity                      -> shrinks every lag weight towards 0
    "smooth": second differences of w (+ a small identity) -> penalises a jagged kernel, so the
              learned filter is a smooth curve, like a real trend filter

An EWMA crossover is a special case: fast EWMA - slow EWMA of the price equals
sum_j [(1-a_slow)^(j+1) - (1-a_fast)^(j+1)] * r[t-j], a fixed hump-shaped kernel. The figure compares
the learned kernels with that one.

Chronological split, with H + 2 rows dropped at each boundary so no label reaches into the next block:
    train 60%: fit w for every lam      validation 20%: pick lam and G      test 20%: scored once

Run: uv run python YANNICK/lag_tikhonov.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from macro_strategy import long_stocks_bonds, macro_prior, slow_trend
from portfolio import Portfolio

L = 250
H = 20
LAMS = np.logspace(-1, 8, 73)
OUT = Path(__file__).resolve().parent / "output"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def z_returns(px: pd.DataFrame) -> pd.DataFrame:
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    return (ret / vol.shift(1)).clip(-6, 6)


def lag_matrix(z: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """X[i, j] = z[rows[i] - j] for j = 0 .. L-1 (rows must be >= L-1)."""
    return np.stack([z[rows - j] for j in range(L)], axis=1)


def gamma(kind: str) -> np.ndarray:
    if kind == "ridge":
        return np.eye(L)
    D = np.zeros((L - 2, L))
    for i in range(L - 2):
        D[i, i : i + 3] = (1, -2, 1)
    return np.vstack([10 * D, 0.1 * np.eye(L)])


def ewma_kernel(fast: int = 16, slow: int = 64) -> np.ndarray:
    af, as_ = 2 / (fast + 1), 2 / (slow + 1)
    j = np.arange(L)
    return (1 - as_) ** (j + 1) - (1 - af) ** (j + 1)


def build(z: pd.DataFrame, y: pd.DataFrame, rows: np.ndarray):
    """Normal equations X'X, X'y and y'y pooled over assets, for the given rows."""
    XtX, Xty, yty, n = np.zeros((L, L)), np.zeros(L), 0.0, 0
    for a in z.columns:
        za, ya = np.nan_to_num(z[a].to_numpy()), y[a].to_numpy()
        r = rows[(rows >= L - 1) & np.isfinite(ya[rows])]
        r = r[np.isfinite(z[a].to_numpy()[r - L + 1])]  # full history available
        X, b = lag_matrix(za, r), ya[r]
        XtX += X.T @ X
        Xty += X.T @ b
        yty += b @ b
        n += len(b)
    return XtX, Xty, yty, n


def mse(w: np.ndarray, XtX, Xty, yty, n) -> float:
    return float((w @ XtX @ w - 2 * w @ Xty + yty) / n)


def filter_forecast(prices: pl.DataFrame, w: np.ndarray) -> pl.DataFrame:
    """Apply the filter: forecast[t] = sum_j w[j] z[t-j]. Uses only returns up to t."""
    px = prices.to_pandas().set_index("AsOfDate")
    z = z_returns(px)
    out = {}
    for a in z.columns:
        za = np.nan_to_num(z[a].to_numpy())
        f = np.convolve(za, w)[: len(za)]
        f[: L + 100] = np.nan
        out[a] = f
    return pl.from_pandas(pd.DataFrame(out, index=px.index).rename_axis("AsOfDate").reset_index())


def main() -> None:
    import matplotlib.pyplot as plt

    pd.set_option("display.width", 200)
    bt = Backtest(PRICES, MACRO)
    px = bt.prices.to_pandas().set_index("AsOfDate")
    dates = pd.to_datetime(px.index)
    z = z_returns(px)
    y = sum(z.shift(-k) for k in range(2, 2 + H))
    n = len(z)
    gap = H + 2
    t_end, v_end = int(n * 0.6), int(n * 0.8)
    blocks = {"train": np.arange(0, t_end - gap), "val": np.arange(t_end, v_end - gap),
              "train+val": np.arange(0, v_end - gap), "test": np.arange(v_end, n - gap)}
    print(f"train ..{dates[t_end - gap - 1].date()} | validation {dates[t_end].date()}..{dates[v_end - gap - 1].date()} "
          f"| test {dates[v_end].date()}..{dates[-1].date()}   (L = {L} lags, target = next {H} days)\n")
    nm = {k: build(z, y, r) for k, r in blocks.items()}
    base = {k: v[2] / v[3] for k, v in nm.items()}  # error of predicting 0

    results, curves = {}, {}
    for kind in ("ridge", "smooth"):
        G = gamma(kind)
        GtG = G.T @ G
        XtX, Xty, yty, cnt = nm["train"]
        errt, errv, size, W = [], [], [], []
        for lam in LAMS:
            w = np.linalg.solve(XtX + lam * GtG, Xty)
            W.append(w)
            errt.append(mse(w, XtX, Xty, yty, cnt) / base["train"])
            errv.append(mse(w, *nm["val"]) / base["val"])
            size.append(float((G @ w) @ (G @ w)))
        errt, errv, size = map(np.array, (errt, errv, size))
        b = int(np.argmin(errv))
        results[kind] = {"λ": LAMS[b], "val err": errv[b], "val err OLS": errv[0], "train err": errt[b]}
        curves[kind] = (errt, errv, size, W, b)
    print("Validation (error relative to predicting 0; below 1 = the filter helps):")
    print(pd.DataFrame(results).T.round(5).to_string())

    kind = min(results, key=lambda k: results[k]["val err"])
    lam = results[kind]["λ"]
    G = gamma(kind)
    XtX, Xty, _, _ = nm["train+val"]
    w = np.linalg.solve(XtX + lam * G.T @ G, Xty)
    test_err = mse(w, *nm["test"]) / base["test"]
    print(f"\nChosen on validation: Γ = {kind}, λ = {lam:.3g}. Refit on train + validation.")
    print(f"Test block, scored once: error relative to predicting 0 = {test_err:.5f}")

    # Trading: the filter on its own, and as the trend leg of equal thirds
    def learned(prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
        return filter_forecast(prices, w)

    def window(f, is_pos, a, b_):
        run = bt.run_positions if is_pos else bt.run
        res = run(f, period="full", check_leakage=False)
        r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
        r = r[(pd.to_datetime(r.index) >= a) & (pd.to_datetime(r.index) < b_)].dropna().to_numpy()
        return float(adjusted_sharpe(r)), float(sharpe(r))

    print("\nTrading the test block (weights fixed from train + validation), adjusted Sharpe / Sharpe:")
    a, b_ = dates[v_end], dates[-1] + pd.Timedelta(days=1)
    for name, f, is_pos in [
        ("learned filter alone", learned, False),
        ("slow trend alone (EWMA vote)", slow_trend, False),
        ("equal thirds (slow trend leg)", Portfolio({"t": slow_trend, "l": long_stocks_bonds, "m": macro_prior}, {k: 1 / 3 for k in "tlm"}), True),
        ("equal thirds, learned filter as trend leg", Portfolio({"t": learned, "l": long_stocks_bonds, "m": macro_prior}, {k: 1 / 3 for k in "tlm"}), True),
    ]:
        adj, s = window(f, is_pos, a, b_)
        print(f"  {name:<44} {adj:+.2f} / {s:+.2f}")

    # ------------------------------------------------------------------------------------ figure
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURF)
    for a_ in ax:
        a_.set_facecolor(SURF)
        a_.grid(True, color=GRID, linewidth=0.6)
        for s_ in a_.spines.values():
            s_.set_visible(False)
        a_.tick_params(colors=INK2, labelsize=8)
    for k, col in (("ridge", S1), ("smooth", S2)):
        errt, errv, size, W, b = curves[k]
        ax[0].plot(errt, size, color=col, linewidth=2, label=f"Γ = {k}")
        ax[0].scatter([errt[b]], [size[b]], s=50, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        ax[1].semilogx(LAMS, errv, color=col, linewidth=2, label=f"Γ = {k}, validation")
        ax[1].semilogx(LAMS, errt, color=col, linewidth=1, linestyle="--", label=f"Γ = {k}, train")
        ax[1].scatter([LAMS[b]], [errv[b]], s=50, color=col, edgecolor=SURF, linewidth=2, zorder=3)
    ax[0].set_yscale("log")
    ax[0].set_title("L-curve (dot = λ chosen on validation)", color=INK, fontsize=11, loc="left")
    ax[0].set_xlabel("training error / error of predicting 0", color=INK2, fontsize=9)
    ax[0].set_ylabel("‖Γw‖²", color=INK2, fontsize=9)
    ax[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    ax[1].axhline(1, color=INK2, linewidth=0.8)
    ax[1].set_title("Train vs validation error by λ", color=INK, fontsize=11, loc="left")
    ax[1].set_xlabel("λ", color=INK2, fontsize=9)
    ax[1].legend(frameon=False, fontsize=7, labelcolor=INK)
    lags = np.arange(L)
    ols = curves["ridge"][3][0]
    ax[2].plot(lags, ols, color=GRID, linewidth=1, label="near-OLS (λ = 0.1), ridge")
    for k, col in (("ridge", S1), ("smooth", S2)):
        wk = curves[k][3][curves[k][4]]
        ax[2].plot(lags, wk, color=col, linewidth=2, label=f"Γ = {k}, chosen λ")
    ek = ewma_kernel()
    ref = curves["smooth"][3][curves["smooth"][4]]
    ax[2].plot(lags, ek * (np.abs(ref).max() / np.abs(ek).max()), color=S3, linewidth=2, linestyle=":",
               label="EWMA 16/64 crossover (scaled)")
    ax[2].axhline(0, color=INK2, linewidth=0.8)
    ax[2].set_ylim(np.array([-1.5, 1.5]) * max(np.abs(ref).max(), 1e-9) * 1.5)
    ax[2].set_title("Learned weight on each past day's return", color=INK, fontsize=11, loc="left")
    ax[2].set_xlabel("lag j (days ago)", color=INK2, fontsize=9)
    ax[2].legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "lag_tikhonov.png", dpi=120)
    print(f"\nFigure: {OUT / 'lag_tikhonov.png'}")


if __name__ == "__main__":
    main()
