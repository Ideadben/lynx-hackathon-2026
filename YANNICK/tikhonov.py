"""Tikhonov regression with L-curve selection of the penalty, as a walk-forward strategy.

    minimise  ||X w - y||^2 + lam * ||G w||^2     ->     w = (X'X + lam G'G)^-1 X'y

Every lam gives a solution, from least squares (lam -> 0) to w = 0 (lam -> inf): the bias-variance
tradeoff. The L-curve plots the training error ||Xw - y||^2 against the weight size ||Gw||^2 for
every lam; we pick the corner, where both are small (the point closest to the origin once both
log axes are scaled to [0, 1]).

Only trainable rows are used: the target is the vol-normalised move over days t+2 .. t+21, so at a
refit date the last 22 rows have no known label yet and are dropped.

Features per asset (standardised on the training rows):
    macro:  gdp_accel, cpi_accel, gdp_12m, cpi_12m  (own planet; FX: planet minus titan;
            commodities: planet average)
    trend:  strength at spans 16, 32, 64
    moves:  20- and 60-day vol-normalised move
One regression per asset class (stocks, bonds, FX, commodities), pooled over its assets.

G (the Tikhonov matrix):
    "ridge":  identity
    "smooth": identity, plus first differences between neighbouring trend speeds, so the model
              cannot give adjacent speeds wild opposite weights

Run: uv run python YANNICK/tikhonov.py      (dev period only; writes YANNICK/output/*.png)
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import paired_bootstrap
from data import MACRO, PRICES
from macro_strategy import asset_class, long_stocks_bonds, macro_features, macro_prior, slow_trend
from portfolio import Portfolio

HORIZON = 20
LAMS = np.logspace(-2, 7, 91)
MACRO_F = ["gdp_accel", "cpi_accel", "gdp_12m", "cpi_12m"]
NAMES = MACRO_F + ["trend16", "trend32", "trend64", "move20", "move60"]
OUT = Path(__file__).resolve().parent / "output"


def gamma(kind: str) -> np.ndarray:
    p = len(NAMES)
    G = np.eye(p)
    if kind == "smooth":
        i16, i32, i64 = NAMES.index("trend16"), NAMES.index("trend32"), NAMES.index("trend64")
        D = np.zeros((2, p))
        D[0, i16], D[0, i32] = 1, -1
        D[1, i32], D[1, i64] = 1, -1
        G = np.vstack([G, 3 * D])
    return G


def design(prices: pl.DataFrame, extras: pl.DataFrame) -> tuple[dict, pd.DataFrame, pd.DatetimeIndex]:
    """Per asset an array (days, features), and the target frame."""
    px = prices.to_pandas().set_index("AsOfDate")
    idx = pd.to_datetime(px.index)
    px.index = idx
    mf = macro_features(extras, idx)
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    z = ret / vol.shift(1)
    y = sum(z.shift(-k) for k in range(2, 2 + HORIZON))
    trend = {}
    for s in (16, 32, 64):
        fast = px.ewm(span=s, adjust=False, min_periods=s).mean()
        slow = px.ewm(span=4 * s, adjust=False, min_periods=4 * s).mean()
        trend[f"trend{s}"] = ((fast - slow) / (vol * np.sqrt(s))).clip(-3, 3)
    moves = {f"move{n}": (z.rolling(n).sum() / np.sqrt(n)).clip(-4, 4) for n in (20, 60)}
    X = {}
    for a in px.columns:
        cls, p = asset_class(a)
        p = p or "world"
        cols = []
        for f in MACRO_F:
            col = mf[f][p] - mf[f]["titan"] if cls == "titan_fx" else mf[f][p]
            cols.append(col.to_numpy())
        cols += [trend[k][a].to_numpy() for k in ("trend16", "trend32", "trend64")]
        cols += [moves[k][a].to_numpy() for k in ("move20", "move60")]
        X[a] = np.column_stack(cols)
    return X, y, idx


def stack(X: dict, y: pd.DataFrame, members: list[str], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = np.vstack([X[a][rows] for a in members])
    b = np.concatenate([y[a].to_numpy()[rows] for a in members])
    ok = ~np.isnan(A).any(1) & ~np.isnan(b)
    return A[ok], b[ok]


def path(A: np.ndarray, b: np.ndarray, G: np.ndarray, lams=LAMS):
    """Weights, training error and weight size along the lam grid (A already standardised)."""
    AtA, Atb, GtG, n = A.T @ A, A.T @ b, G.T @ G, len(b)
    W, err, size = [], [], []
    for lam in lams:
        w = np.linalg.solve(AtA + lam * GtG, Atb)
        W.append(w)
        err.append(((A @ w - b) ** 2).mean())
        size.append(float((G @ w) @ (G @ w)))
    return np.array(W), np.array(err), np.array(size), n


def corner(err: np.ndarray, size: np.ndarray) -> int:
    """Index of the L-curve corner: closest to the origin after scaling both log axes to [0, 1]."""
    x, yy = np.log(err), np.log(np.maximum(size, 1e-300))
    x = (x - x.min()) / (np.ptp(x) or 1)
    yy = (yy - yy.min()) / (np.ptp(yy) or 1)
    return int(np.argmin(x**2 + yy**2))


def tikhonov(prices: pl.DataFrame, extras: pl.DataFrame, kind: str = "ridge", lam: float | str = "corner",
             min_years: int = 5) -> pl.DataFrame:
    """Walk-forward: refit each January on earlier, labelled rows; lam from that year's L-curve."""
    X, y, idx = design(prices, extras)
    assets = list(X)
    G = gamma(kind)
    years = idx.year
    groups: dict[str, list[str]] = {}
    for a in assets:
        groups.setdefault(asset_class(a)[0], []).append(a)
    out = np.full((len(idx), len(assets)), np.nan)
    for year in sorted(set(years)):
        past = np.where(years < year)[0]
        if len(set(years[past])) < min_years:
            continue
        train = past[: -(HORIZON + 2)]  # only rows whose label is known
        now = np.where(years == year)[0]
        for cls, members in groups.items():
            A, b = stack(X, y, members, train)
            if len(b) < 500:
                continue
            sd = A.std(0)
            sd[sd == 0] = 1
            A = A / sd
            if lam == "corner":
                W, err, size, _ = path(A, b, G)
                w = W[corner(err, size)]
            else:
                w = np.linalg.solve(A.T @ A + float(lam) * G.T @ G, A.T @ b)
            for a in members:
                out[now, assets.index(a)] = (X[a][now] / sd) @ w
    frame = pd.DataFrame(out, index=prices["AsOfDate"].to_pandas(), columns=assets)
    return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())


# --------------------------------------------------------------------------------------- figures
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
S1, S2 = "#2a78d6", "#eb6834"


def _style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)


def figures(bt: Backtest, kind: str = "ridge") -> pd.DataFrame:
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    dev_rows = int(bt.prices.height * 0.75)
    prices, extras = bt.prices.head(dev_rows), bt.extras
    X, y, idx = design(prices, extras)
    G = gamma(kind)
    groups: dict[str, list[str]] = {}
    for a in X:
        groups.setdefault(asset_class(a)[0], []).append(a)
    split = int(len(idx) * 0.7)
    train_rows = np.arange(0, split - (HORIZON + 2))  # labels known before the validation block
    val_rows = np.arange(split, len(idx))
    full_rows = np.arange(0, len(idx) - (HORIZON + 2))

    fig1, ax1 = plt.subplots(2, 2, figsize=(10, 8), facecolor=SURF)
    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 8), facecolor=SURF)
    summary = {}
    for k, (cls, members) in enumerate(groups.items()):
        A, b = stack(X, y, members, full_rows)
        sd = A.std(0)
        W, err, size, n = path(A / sd, b, G)
        c = corner(err, size)
        a = ax1.flat[k]
        _style(a, f"{cls}  ({len(members)} assets, {n:,} rows)")
        a.plot(err, size, color=S1, linewidth=2)
        a.set_yscale("log")  # the error spans a narrow range, so its axis stays linear
        a.xaxis.set_major_locator(plt.MaxNLocator(4))
        a.scatter([err[c]], [size[c]], s=60, color=S1, edgecolor=SURF, linewidth=2, zorder=3)
        a.annotate(f"corner  λ = {LAMS[c]:.3g}", (err[c], size[c]), textcoords="offset points",
                   xytext=(10, 6), color=INK, fontsize=9)
        for j, off in ((0, (8, -4)), (70, (8, 4))):  # λ = 0.01 (least squares) and λ = 1e5
            a.annotate(f"λ = {LAMS[j]:.0e}", (err[j], size[j]), textcoords="offset points", xytext=off,
                       color=INK2, fontsize=8)
        a.set_xlabel("training error  mean (Xw − y)²", color=INK2, fontsize=9)
        a.set_ylabel("weight size  ‖Γw‖²", color=INK2, fontsize=9)

        # bias-variance: train on the first 70% of dev, validate on the rest
        At, bt_ = stack(X, y, members, train_rows)
        Av, bv = stack(X, y, members, val_rows)
        sdt = At.std(0)
        Wt, errt, _, _ = path(At / sdt, bt_, G)
        errv = np.array([((Av / sdt @ w - bv) ** 2).mean() for w in Wt])
        ct = corner(errt, np.array([float((G @ w) @ (G @ w)) for w in Wt]))
        base_t, base_v = (bt_**2).mean(), (bv**2).mean()  # error of predicting 0 (lam -> inf)
        a = ax2.flat[k]
        _style(a, f"{cls}: error relative to predicting 0")
        a.semilogx(LAMS, errt / base_t, color=S1, linewidth=2, label="training (first 70% of dev)")
        a.semilogx(LAMS, errv / base_v, color=S2, linewidth=2, label="validation (last 30% of dev)")
        a.axvline(LAMS[ct], color=INK2, linewidth=1, linestyle=":")
        a.annotate("L-curve corner", (LAMS[ct], a.get_ylim()[1]), textcoords="offset points", xytext=(4, -12),
                   color=INK2, fontsize=8)
        a.axhline(1.0, color=INK2, linewidth=0.8)
        a.set_xlabel("λ (Tikhonov parameter)", color=INK2, fontsize=9)
        a.set_ylabel("relative mean squared error", color=INK2, fontsize=9)
        if k == 0:
            a.legend(frameon=False, fontsize=8, labelcolor=INK)
        best_v = int(np.argmin(errv))
        summary[cls] = {
            "corner λ": LAMS[c], "best validation λ": LAMS[best_v],
            "validation err at corner": errv[ct] / base_v, "best validation err": errv[best_v] / base_v,
            "validation err OLS (λ=0.01)": errv[0] / base_v,
        }
        summary[cls].update({f"w_{nm}": W[c][i] for i, nm in enumerate(NAMES)})
    fig1.suptitle(f"L-curves, Γ = {kind}, dev period", color=INK, x=0.02, ha="left")
    fig2.suptitle(f"Bias-variance, Γ = {kind}: below 1 beats predicting 0", color=INK, x=0.02, ha="left")
    for f, name in ((fig1, "lcurve"), (fig2, "bias_variance")):
        f.tight_layout()
        f.savefig(OUT / f"tikhonov_{name}_{kind}.png", dpi=130)
    return pd.DataFrame(summary).T


def _ret(r) -> pd.Series:
    return r.returns.to_pandas().set_index("AsOfDate")["returns"].dropna()


# ---------------------------------------------------------------- train / validation / test split
def split_study(bt: Backtest, kinds=("ridge", "smooth"), train: float = 0.6, val: float = 0.2) -> None:
    """Chronological 60/20/20 split on all data.

    train:      fit the weights for every lam (and every G)
    validation: pick lam and G by validation error; also report the L-curve corner for comparison
    test:       refit on train + validation with the chosen lam, then score once on test
    Between blocks, the last HORIZON + 2 rows are dropped, because their labels reach into the next block.
    """
    import matplotlib.pyplot as plt

    from backtest.stats import adjusted_sharpe, sharpe

    X, y, idx = design(bt.prices, bt.extras)
    n = len(idx)
    gap = HORIZON + 2
    t_end, v_end = int(n * train), int(n * (train + val))
    rows = {
        "train": np.arange(0, t_end - gap),
        "val": np.arange(t_end, v_end - gap),
        "train+val": np.arange(0, v_end - gap),
        "test": np.arange(v_end, n - gap),
    }
    print(f"train {idx[0].date()} .. {idx[t_end - gap - 1].date()} | validation {idx[t_end].date()} .. "
          f"{idx[v_end - gap - 1].date()} | test {idx[v_end].date()} .. {idx[n - 1].date()}\n")
    groups: dict[str, list[str]] = {}
    for a in X:
        groups.setdefault(asset_class(a)[0], []).append(a)

    choice, table = {}, []
    fig, axes = plt.subplots(len(kinds), len(groups), figsize=(14, 3.6 * len(kinds)), facecolor=SURF, squeeze=False)
    for r, kind in enumerate(kinds):
        G = gamma(kind)
        for c, (cls, members) in enumerate(groups.items()):
            At, bt_ = stack(X, y, members, rows["train"])
            Av, bv = stack(X, y, members, rows["val"])
            sd = At.std(0)
            W, errt, size, _ = path(At / sd, bt_, G)
            errv = np.array([((Av / sd @ w - bv) ** 2).mean() for w in W]) / (bv**2).mean()
            errt = errt / (bt_**2).mean()
            best, corn = int(np.argmin(errv)), corner(errt, size)
            table.append({"Γ": kind, "class": cls, "λ by validation": LAMS[best], "λ L-curve corner": LAMS[corn],
                          "val err at best": errv[best], "val err at corner": errv[corn], "val err OLS": errv[0]})
            choice[(kind, cls)] = (LAMS[best], errv[best])
            a = axes[r, c]
            _style(a, f"Γ = {kind}, {cls}")
            a.semilogx(LAMS, errt, color=S1, linewidth=2, label="train")
            a.semilogx(LAMS, errv, color=S2, linewidth=2, label="validation")
            a.scatter([LAMS[best]], [errv[best]], s=50, color=S2, edgecolor=SURF, linewidth=2, zorder=3)
            a.axvline(LAMS[corn], color=INK2, linewidth=1, linestyle=":")
            a.axhline(1.0, color=INK2, linewidth=0.8)
            a.set_xlabel("λ", color=INK2, fontsize=9)
            if c == 0:
                a.set_ylabel("error / error of predicting 0", color=INK2, fontsize=9)
            if r == 0 and c == 0:
                a.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Train vs validation error by λ (dot: chosen λ; dotted: L-curve corner). Test is not shown.",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "tikhonov_train_val.png", dpi=120)
    print("Choosing λ on validation (errors are relative to predicting 0; below 1 is useful):")
    print(pd.DataFrame(table).round(4).to_string(index=False))

    # Pick G per class by validation error too, then refit on train+val and score once on test.
    weights, scales, test_rows = {}, {}, []
    for cls, members in groups.items():
        kind = min(kinds, key=lambda k: choice[(k, cls)][1])
        lam = choice[(kind, cls)][0]
        A, b = stack(X, y, members, rows["train+val"])
        sd = A.std(0)
        G = gamma(kind)
        w = np.linalg.solve((A / sd).T @ (A / sd) + lam * G.T @ G, (A / sd).T @ b)
        weights[cls], scales[cls] = w, sd
        Ate, bte = stack(X, y, members, rows["test"])
        test_rows.append({"class": cls, "Γ": kind, "λ": lam,
                          "test err (vs predicting 0)": ((Ate / sd @ w - bte) ** 2).mean() / (bte**2).mean(),
                          "test corr(forecast, outcome)": np.corrcoef(Ate / sd @ w, bte)[0, 1]})
    print("\nTest block, scored once with the validation-chosen λ and Γ (refit on train + validation):")
    print(pd.DataFrame(test_rows).round(4).to_string(index=False))

    # Trade it on the test block: fixed weights, so positions only use data up to each date.
    def fixed(prices: pl.DataFrame, extras: pl.DataFrame) -> pl.DataFrame:
        Xf, _, _ = design(prices, extras)
        out = pd.DataFrame({a: (Xf[a] / scales[asset_class(a)[0]]) @ weights[asset_class(a)[0]] for a in Xf},
                           index=prices["AsOfDate"].to_pandas())
        return pl.from_pandas(out.rename_axis("AsOfDate").reset_index())

    start = idx[v_end]
    print(f"\nTrading the test block ({start.date()} .. {idx[-1].date()}), adjusted Sharpe:")
    thirds = Portfolio({"t": slow_trend, "l": long_stocks_bonds, "m": macro_prior}, {k: 1 / 3 for k in "tlm"})
    for name, f, is_pos in [("Tikhonov, validation-chosen", fixed, False), ("macro prior", macro_prior, False),
                            ("equal thirds", thirds, True),
                            ("thirds with Tikhonov as 4th quarter",
                             Portfolio({"t": slow_trend, "l": long_stocks_bonds, "m": macro_prior, "k": fixed}, {k: 0.25 for k in "tlmk"}), True)]:
        run = bt.run_positions if is_pos else bt.run
        res = run(f, period="full", check_leakage=False, name=name)
        r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
        r = r[pd.to_datetime(r.index) >= start].dropna().to_numpy()
        print(f"  {name:<38} adj {float(adjusted_sharpe(r)):+.2f}  sharpe {float(sharpe(r)):+.2f}")


def main() -> None:
    pd.set_option("display.width", 220)
    bt = Backtest(PRICES, MACRO)
    for kind in ("ridge", "smooth"):
        s = figures(bt, kind)
        print(f"\n===== Γ = {kind}: L-curve on all of dev (trainable rows), and a 70/30 check inside dev")
        print(s[["corner λ", "best validation λ", "validation err at corner", "best validation err",
                 "validation err OLS (λ=0.01)"]].round(4).to_string())
        print("weights at the corner (per standardised feature):")
        print(s[[c for c in s.columns if c.startswith("w_")]].round(4).to_string())

    for scoring in ("pdf", "lynx"):
        bt = Backtest(PRICES, MACRO, scoring=scoring)
        thirds = bt.run_positions(Portfolio({"t": slow_trend, "l": long_stocks_bonds, "m": macro_prior}, {k: 1 / 3 for k in "tlm"}))
        print(f"\n===== walk-forward strategies, dev, scoring={scoring}  (equal thirds: {thirds.adjusted_sharpe:.2f})")
        tests = {
            "OLS (λ = 0)": partial(tikhonov, lam=0.0),
            "ridge, λ from L-curve": partial(tikhonov, kind="ridge"),
            "smooth Γ, λ from L-curve": partial(tikhonov, kind="smooth"),
            "ridge, λ = 1e4 (fixed)": partial(tikhonov, lam=1e4),
            "ridge, λ = 1e6 (fixed)": partial(tikhonov, lam=1e6),
            "macro prior (for comparison)": macro_prior,
        }
        for name, f in tests.items():
            r = bt.run(f, name=name)
            x = _ret(r)
            h = len(x) // 2
            sr = lambda v: v.mean() / v.std() * np.sqrt(252)  # noqa: E731
            both = pd.concat([x, _ret(thirds)], axis=1, join="inner")
            print(f"  {name:<30} adj {r.adjusted_sharpe:+.2f}  gross {r.sharpe_ex_slippage:+.2f}  halves {sr(x[:h]):+.2f}/{sr(x[h:]):+.2f}"
                  f"  hold {r.av_holding:3.0f}d  corr(thirds) {both.corr().iloc[0, 1]:+.2f}")
        for name, f in {"thirds with Tikhonov (smooth) as macro leg": partial(tikhonov, kind="smooth"),
                        "four quarters: thirds + Tikhonov (smooth)": None}.items():
            if f is not None:
                P = Portfolio({"t": slow_trend, "l": long_stocks_bonds, "m": f}, {k: 1 / 3 for k in "tlm"})
            else:
                P = Portfolio({"t": slow_trend, "l": long_stocks_bonds, "m": macro_prior, "k": partial(tikhonov, kind="smooth")},
                              {k: 0.25 for k in "tlmk"})
            r = bt.run_positions(P, name=name)
            both = pd.concat([_ret(r), _ret(thirds)], axis=1, join="inner")
            d = paired_bootstrap(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())
            print(f"  {name:<44} adj {r.adjusted_sharpe:+.2f}  vs thirds {str(d)[:48]}")


if __name__ == "__main__":
    import sys

    if "--split" in sys.argv:
        split_study(Backtest(PRICES, MACRO))
    else:
        main()
