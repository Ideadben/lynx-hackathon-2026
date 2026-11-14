"""Does ML add anything? Ridge regression and XGBoost on engineered features, walk-forward.

Following Lynx's advice: engineered features (not raw prices), and the two-day lag between features
and labels. Pooled across assets (5x the data), refit each January on earlier years only, with the
last `horizon + 2` days before the refit dropped because their labels were not yet known.

Target: the vol-normalised move over the `horizon` days a position decided today would earn,
z[t+2] + ... + z[t+1+horizon].

Run: uv run --with xgboost python YANNICK/ml_models.py
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.stats import paired_bootstrap
from strategies import trend

ROOT = Path(__file__).resolve().parents[1]
from data import PRICES  # noqa: E402


def build(prices: pl.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    px = prices.to_pandas().set_index("AsOfDate")
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    z = ret / vol.shift(1)
    f = {}
    for span in (8, 16, 32, 64):
        fast = px.ewm(span=span, adjust=False, min_periods=span).mean()
        slow = px.ewm(span=4 * span, adjust=False, min_periods=4 * span).mean()
        f[f"trend{span}"] = ((fast - slow) / (vol * np.sqrt(span))).clip(-3, 3)
    for n in (1, 5, 20, 60):
        f[f"move{n}"] = (z.rolling(n).sum() / np.sqrt(n)).clip(-4, 4)
    f["shock"] = z.where(z.abs() > 2, 0).rolling(5, min_periods=1).sum().clip(-6, 6)
    f["er40"] = (px - px.shift(40)).abs() / ret.abs().rolling(40).sum()
    f["vol_ratio"] = (vol / vol.rolling(250, min_periods=100).mean()).clip(0, 3)
    hi, lo = px.rolling(80).max(), px.rolling(80).min()
    f["range80"] = (px - (hi + lo) / 2) / (hi - lo)
    return f, z


def walk_forward(prices: pl.DataFrame, model: str = "ridge", horizon: int = 5, min_years: int = 4,
                 l2: float = 100.0, features: tuple[str, ...] | None = None) -> pl.DataFrame:
    feats, z = build(prices)
    names = list(features or feats)
    assets = list(z.columns)
    # Label at t: sum of z[t+2 .. t+1+horizon]
    y = sum(z.shift(-k) for k in range(2, 2 + horizon))
    X3 = np.stack([feats[n][assets].to_numpy() for n in names], -1)  # (days, assets, features)
    Y = y[assets].to_numpy()
    years = z.index.year
    out = np.full(Y.shape, np.nan)
    for year in sorted(set(years)):
        past = np.where(years < year)[0]
        if len(set(years[past])) < min_years:
            continue
        train = past[: -(horizon + 2)]  # labels of these rows were fully known by the refit date
        Xt = X3[train].reshape(-1, len(names))
        yt = Y[train].reshape(-1)
        ok = ~np.isnan(Xt).any(1) & ~np.isnan(yt)
        Xt, yt = Xt[ok], yt[ok]
        mu, sd = Xt.mean(0), Xt.std(0)
        rows = np.where(years == year)[0]
        Xn = (X3[rows].reshape(-1, len(names)) - mu) / sd
        if model == "ridge":
            A = (Xt - mu) / sd
            beta = np.linalg.solve(A.T @ A + l2 * np.eye(len(names)), A.T @ (yt - yt.mean()))
            pred = Xn @ beta
        else:
            import xgboost as xgb

            m = xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.7,
                                 colsample_bytree=0.8, min_child_weight=200, reg_lambda=10.0,
                                 objective="reg:pseudohubererror", random_state=0)
            m.fit((Xt - mu) / sd, yt)
            pred = m.predict(np.nan_to_num(Xn)) - np.median(m.predict((Xt - mu) / sd))
        pred = pred.reshape(len(rows), len(assets))
        pred[np.isnan(Xn.reshape(len(rows), len(assets), -1)).any(-1)] = np.nan
        out[rows] = pred
    frame = pd.DataFrame(out, index=z.index, columns=assets)
    return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())


def ridge_coefficients(prices: pl.DataFrame, horizon: int = 5) -> pd.Series:
    feats, z = build(prices)
    names = list(feats)
    y = sum(z.shift(-k) for k in range(2, 2 + horizon))
    X = np.column_stack([feats[n].stack() for n in names])
    yy = y.stack().reindex(feats[names[0]].stack().index).to_numpy()
    ok = ~np.isnan(X).any(1) & ~np.isnan(yy)
    A = (X[ok] - X[ok].mean(0)) / X[ok].std(0)
    beta = np.linalg.solve(A.T @ A + 100 * np.eye(len(names)), A.T @ (yy[ok] - yy[ok].mean()))
    t = beta / (np.sqrt(((yy[ok] - A @ beta) ** 2).mean() / len(A)) * np.sqrt(horizon))
    return pd.DataFrame({"coef x100": beta * 100, "approx t (overlap-adjusted)": t}, index=names).round(2)


def _ret(r) -> pd.Series:
    return r.returns.to_pandas().set_index("AsOfDate")["returns"].dropna()


def main() -> None:
    bt = Backtest(PRICES)
    dev = bt.prices.head(int(bt.prices.height * 0.75))
    print("What a ridge fit on all of dev says (reading only, not a strategy):")
    print(ridge_coefficients(dev))

    base = bt.run(trend.forecast, name="trend")
    tests = {
        "ridge, all features, h=5": partial(walk_forward, model="ridge", horizon=5),
        "ridge, all features, h=20": partial(walk_forward, model="ridge", horizon=20),
        "ridge, trend features only, h=20": partial(walk_forward, model="ridge", horizon=20,
                                                    features=("trend8", "trend16", "trend32", "trend64")),
        "xgboost, all features, h=5": partial(walk_forward, model="xgb", horizon=5),
        "xgboost, all features, h=20": partial(walk_forward, model="xgb", horizon=20),
    }
    print("\nWalk-forward, dev period, PDF scoring. 'vs trend' compares on the same days.")
    rets = {"trend.py": _ret(base)}
    for name, f in tests.items():
        r = bt.run(f, name=name)
        ret = _ret(r)
        rets[name] = ret
        both = pd.concat([ret, _ret(base)], axis=1, join="inner")
        h = len(ret) // 2
        sr = lambda x: x.mean() / x.std() * np.sqrt(252)  # noqa: E731
        diff = paired_bootstrap(both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy())
        print(f"  {name:<34} adj {r.adjusted_sharpe:.2f}  halves {sr(ret[:h]):.2f}/{sr(ret[h:]):.2f}  "
              f"hold {r.av_holding:.0f}d  corr {both.corr().iloc[0, 1]:.2f}  vs trend {diff}")
    lynx = Backtest(PRICES, scoring="lynx")
    print(f"\nLynx costs: ridge h=20 {lynx.run(tests['ridge, all features, h=20']).adjusted_sharpe:.2f}, "
          f"xgboost h=20 {lynx.run(tests['xgboost, all features, h=20']).adjusted_sharpe:.2f}, "
          f"trend {lynx.run(trend.forecast).adjusted_sharpe:.2f}")


if __name__ == "__main__":
    main()
