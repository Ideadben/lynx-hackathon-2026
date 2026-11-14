"""Tikhonov lag filter with a Γ that weighs recent days higher than far-away days.

Model as in src/strategies/tikhonov.py (L = 250, H = 5, expanding window, yearly refit), but Γ adds
a ridge penalty that grows with the lag j, so far-away betas are shrunk harder towards zero:

    Γ = [ s * D_smooth ; diag(d_j) ]      d_j increasing in j

Bayesian reading: the prior variance of beta_j is proportional to 1 / d_j^2, i.e. older returns are
believed to matter less unless the data says otherwise.

For each Γ, α is chosen on the dev blocks 2003-2010 (walk-forward trading adjusted Sharpe); the
validation block (Jun 2011 - Oct 2014) is reported for that α. The test block is not used.

Writes YANNICK/output/tikhonov_gamma.png.
Run: uv run python YANNICK/tikhonov_gamma.py
"""

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from strategies import tikhonov as tk
from tikhonov_overfit import prepare

L = tk.L
J = np.arange(L)
ALPHAS = np.logspace(0, 5, 11)


def second_diff(scale: np.ndarray | float = 1.0) -> np.ndarray:
    D = np.zeros((L - 2, L))
    for i in range(L - 2):
        D[i, i:i + 3] = (1, -2, 1)
    return D * (np.asarray(scale)[: L - 2, None] if np.ndim(scale) else scale)


def first_diff(scale: np.ndarray | float = 1.0) -> np.ndarray:
    """Rows w_j - w_{j+1}: penalises changes between neighbouring betas (a flat filter costs nothing)."""
    D = np.zeros((L - 1, L))
    for i in range(L - 1):
        D[i, i:i + 2] = (1, -1)
    return D * (np.asarray(scale)[: L - 1, None] if np.ndim(scale) else scale)


GAMMAS = {
    "smooth (current)": (10 * second_diff(), 0.1 * np.ones(L)),
    "first order": (10 * first_diff(), 0.1 * np.ones(L)),
    "first order, 10x vs ridge": (100 * first_diff(), 0.1 * np.ones(L)),
    "first + second order": (np.vstack([10 * first_diff(), 10 * second_diff()]), 0.1 * np.ones(L)),
    "first order growing with lag": (10 * first_diff(1 + J / 50), 0.1 * np.ones(L)),
    "smooth + ridge growing linearly": (10 * second_diff(), 0.1 * (1 + J / 50)),
    "smooth + ridge growing quadratically": (10 * second_diff(), 0.1 * (1 + J / 50) ** 2),
    "smooth + ridge growing exp(j/100)": (10 * second_diff(), 0.1 * np.exp(J / 100)),
    "smooth + ridge growing exp(j/50)": (10 * second_diff(), 0.1 * np.exp(J / 50)),
    "smoothness growing with lag": (10 * second_diff(1 + J / 50), 0.1 * np.ones(L)),
    "ridge only, growing linearly": (None, 1 + J / 50),
    "ridge only, growing exp(j/100)": (None, np.exp(J / 100)),
}


def gamma(name: str) -> np.ndarray:
    D, d = GAMMAS[name]
    return np.diag(d) if D is None else np.vstack([D, np.diag(d)])


def normal_equations(X, y, usable, years):
    """Expanding-window X'X and X'y for every refit year (rows whose label is known by then)."""
    gap = tk.H + 2
    out = {}
    for year in range(2000, int(years.max()) + 1):
        start = int(np.argmax(years == year))
        rows = np.arange(0, start - gap)
        mask = usable[rows]
        Xr, yr = X[rows][mask], y[rows][mask]
        out[year] = (Xr.T @ Xr, Xr.T @ yr, len(set(years[rows])))
    return out


def main() -> None:
    import matplotlib.pyplot as plt

    pd.set_option("display.width", 220)
    bt = Backtest(PRICES, MACRO)
    X, y, usable, years = prepare(bt.prices)
    assets = [c for c in bt.prices.columns if c != "AsOfDate"]
    dates = pd.DatetimeIndex(pd.to_datetime(bt.prices["AsOfDate"].to_pandas()))
    n = len(dates)
    val = (dates[int(n * .75)], dates[int(n * .9)])
    NE = normal_equations(X, y, usable, years)

    def run(G, alpha):
        GtG = G.T @ G
        fc = np.full(y.shape, np.nan)
        betas, dfs = {}, []
        for year, (XtX, Xty, ny) in NE.items():
            lam = alpha * np.trace(XtX) / L / ny
            A = XtX + lam * GtG
            w = np.linalg.solve(A, Xty)
            betas[year] = w
            dfs.append(np.trace(np.linalg.solve(A, XtX)))
            fc[years == year] = X[years == year] @ w
        frame = bt.prices.select("AsOfDate").with_columns(pl.Series(a, fc[:, i], nan_to_null=True) for i, a in enumerate(assets))
        r = bt.run(lambda prices, *e: frame, period="full", check_leakage=False).returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        cut = lambda a, b: float(adjusted_sharpe(r[(r.index >= a) & (r.index < b)].dropna().to_numpy()))  # noqa: E731
        yr = r[r.index.year >= 2000]
        yearly = yr.groupby(yr.index.year).apply(lambda x: float(sharpe(x.dropna().to_numpy())))
        return {"dev blocks adj": cut(pd.Timestamp("2003-01-01"), pd.Timestamp("2011-01-01")), "validation adj": cut(*val),
                "mean/yr 2000-10": yearly[yearly.index <= 2010].mean(), "mean/yr 2011-16": yearly[yearly.index >= 2011].mean(),
                "eff. params": float(np.mean(dfs))}, betas

    rows, chosen = {}, {}
    for name in GAMMAS:
        G = gamma(name)
        best = None
        for alpha in ALPHAS:
            s, betas = run(G, alpha)
            if best is None or s["dev blocks adj"] > best[0]["dev blocks adj"]:
                best = (s, betas, alpha)
        s, betas, alpha = best
        w = betas[2010]  # fitted on 1995-2009, inside dev
        mass = np.abs(w)
        s.update({"α": alpha, "weight share, last 60 days": mass[:60].sum() / mass.sum(),
                  "weight share, 150+ days": mass[150:].sum() / mass.sum()})
        rows[name], chosen[name] = s, w
        print(f"  {name:<38} α={alpha:<8.3g} dev blocks {s['dev blocks adj']:+.2f}  validation {s['validation adj']:+.2f}", flush=True)
    t = pd.DataFrame(rows).T
    print("\nEach Γ at its best α (chosen on the dev blocks):")
    print(t[["α", "dev blocks adj", "validation adj", "mean/yr 2000-10", "mean/yr 2011-16", "eff. params",
             "weight share, last 60 days", "weight share, 150+ days"]].round(2).to_string())

    # Figure: penalty profile d_j | betas at the chosen α (2010 refit)
    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), facecolor=SURF)
    for a in ax:
        a.set_facecolor(SURF)
        a.grid(True, color=GRID, linewidth=0.6)
        for s_ in a.spines.values():
            s_.set_visible(False)
        a.tick_params(colors=INK2, labelsize=8)
    profiles = {  # the distinct shapes; several Γ share one
        "flat (current; also 'smoothness growing' for the ridge part)": np.ones(L),
        "linear 1 + j/50 (also the curvature weight in 'smoothness growing')": 1 + J / 50,
        "quadratic (1 + j/50)²": (1 + J / 50) ** 2,
        "exp(j/100)": np.exp(J / 100),
        "exp(j/50)": np.exp(J / 50),
    }
    for (label, d), col in zip(profiles.items(), colors):
        ax[0].plot(J, d, color=col, linewidth=2, label=label)
    shown = ["smooth (current)", "smoothness growing with lag", "first order", "first + second order",
             "smooth + ridge growing exp(j/100)"]
    for name, col in zip(shown, colors):
        ax[1].plot(J, chosen[name], color=col, linewidth=2.2 if name == "smooth (current)" else 1.5,
                   label=f"{name} (dev {rows[name]['dev blocks adj']:+.2f}, val {rows[name]['validation adj']:+.2f})")
    ax[0].set_yscale("log")
    ax[0].set_title("Penalty profile over lags, relative to day 0", color=INK, fontsize=11, loc="left")
    ax[0].set_xlabel("lag j (days ago)", color=INK2, fontsize=9)
    ax[0].legend(frameon=False, fontsize=7, labelcolor=INK)
    ax[1].axhline(0, color=INK2, linewidth=0.8)
    ax[1].set_title("Betas at the chosen α (2010 refit)", color=INK, fontsize=11, loc="left")
    ax[1].set_xlabel("lag j (days ago)", color=INK2, fontsize=9)
    ax[1].legend(frameon=False, fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig("YANNICK/output/tikhonov_gamma.png", dpi=120)
    print("\nwrote YANNICK/output/tikhonov_gamma.png")

    import json
    from pathlib import Path

    out = Path(__file__).resolve().parent / "output" / "tikhonov_gamma"
    out.mkdir(parents=True, exist_ok=True)
    meta = [{"label": "tikhonov_gamma.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"}]
    key = {"α": "alpha", "dev blocks adj": "dev_blocks_adj", "validation adj": "validation_adj", "mean/yr 2000-10": "mean_yr_2000_10",
           "mean/yr 2011-16": "mean_yr_2011_16", "eff. params": "eff_params", "weight share, last 60 days": "share_last_60",
           "weight share, 150+ days": "share_150_plus"}
    summary = [{"gamma": k, **{key[c]: float(v) for c, v in r.items() if c in key}} for k, r in rows.items()]
    betas = [{"gamma": k, "lag": int(j), "beta": float(w[j])} for k, w in chosen.items() for j in range(L)]
    prof = [{"profile": label, "lag": int(j), "penalty": float(d[j])} for label, d in profiles.items() for j in range(L)]
    for name, data in {"summary": summary, "betas": betas, "profiles": prof}.items():
        (out / f"{name}.json").write_text(json.dumps({"rows": data, "meta": meta}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
