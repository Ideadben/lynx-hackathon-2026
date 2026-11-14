"""Which macro features predict which assets? Dev period only.

Macro values are daily (calendar days) and used as of each price date. Features per planet:
growth over 3 and 12 months, growth acceleration (3m growth now minus 3 months ago), and the same for
inflation (CPI). FX features use the planet-minus-titan difference, since each FX rate is quoted
against titan. Target: the vol-normalised move over the 20 trading days a position decided today
earns (days t+2 .. t+21). t-stats are divided by sqrt(20) for the overlapping windows.

Run: uv run python YANNICK/macro_scan.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data import MACRO, PRICES

HORIZON = 20
PLANETS = ["ganymede", "io", "titan", "callisto", "europa"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    px = pd.read_csv(PRICES, index_col=0, parse_dates=True)
    mac = pd.read_csv(MACRO, index_col=0, parse_dates=True)
    dev_end = px.index[int(len(px) * 0.75)]
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    z = ret / vol.shift(1)
    target = sum(z.shift(-k) for k in range(2, 2 + HORIZON))
    return px.loc[:dev_end], mac.loc[:dev_end], target.loc[:dev_end]


def features(mac: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Macro features per planet, computed on calendar days and sampled on price dates."""
    lg = np.log(mac)
    out = {}
    for kind in ("gdp", "cpi"):
        cols = {p: f"{kind}|{p}" for p in PLANETS}
        x = lg[[cols[p] for p in PLANETS]].set_axis(PLANETS, axis=1)
        g3 = x - x.shift(91)
        g12 = x - x.shift(365)
        out[f"{kind}_3m"] = g3
        out[f"{kind}_12m"] = g12
        out[f"{kind}_accel"] = g3 - g3.shift(91)
    return {k: v.reindex(index, method="ffill") for k, v in out.items()}


def ic(f: pd.Series, y: pd.Series) -> tuple[float, float]:
    both = pd.concat([f, y], axis=1).dropna()
    if len(both) < 300:
        return np.nan, np.nan
    r = both.iloc[:, 0].corr(both.iloc[:, 1])
    return r, r * np.sqrt(len(both) / HORIZON)


def main() -> None:
    px, mac, target = load()
    feats = features(mac, px.index)
    rows = []
    for name, f in feats.items():
        for p in PLANETS:
            for kind in ("stocks", "bonds"):
                asset = f"{p}_{kind}"
                if asset in target:
                    r, t = ic(f[p], target[asset])
                    rows.append((name, "own", asset, r, t))
            fx = f"{p}_titan_fx"
            if fx in target:
                r, t = ic(f[p] - f["titan"], target[fx])
                rows.append((name, "minus titan", fx, r, t))
        # commodities: against the average across planets
        avg = f.mean(axis=1)
        for asset in target.columns:
            if not any(asset.startswith(p + "_") for p in PLANETS):
                r, t = ic(avg, target[asset])
                rows.append((name, "planet avg", asset, r, t))
    res = pd.DataFrame(rows, columns=["feature", "version", "asset", "ic", "t"])
    pd.set_option("display.width", 200)
    print(f"Dev period {px.index[0].date()} to {px.index[-1].date()}, target = next {HORIZON} days.\n")
    print("Mean IC per feature and asset class (positive = the feature predicts a rise):")
    res["class"] = res["asset"].str.extract(r"_(stocks|bonds|titan_fx)$")[0].fillna("commodity")
    print(res.pivot_table(index="feature", columns="class", values="ic", aggfunc="mean").round(3).to_string())
    print("\nAverage |t| per feature and class (about 2 is meaningful for one cell; there are many cells):")
    res["abs_t"] = res["t"].abs()
    print(res.pivot_table(index="feature", columns="class", values="abs_t", aggfunc="mean").round(2).to_string())
    print("\nStrongest 20 cells by |t|:")
    print(res.reindex(res["t"].abs().sort_values(ascending=False).index).head(20)[
        ["feature", "version", "asset", "ic", "t"]].round(3).to_string(index=False))
    print("\nConsistency: share of planets where the sign agrees with the class average")
    for cls in ("stocks", "bonds", "titan_fx"):
        sub = res[res["class"] == cls]
        agree = sub.groupby("feature").apply(lambda g: (np.sign(g["ic"]) == np.sign(g["ic"].mean())).mean())
        print(f"  {cls:<9}", agree.round(2).to_dict())


if __name__ == "__main__":
    main()
