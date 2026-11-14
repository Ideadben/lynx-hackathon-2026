"""Choose the Tikhonov vs triple share, and the data for the portfolio dashboard.

Two strategies: Tikhonov + GARCH (src/strategies/tikhonov.py, alpha = 46) and the triple
(slow trend, long stocks+bonds, macro; equal thirds inside). Tikhonov gets share w, each triple
part (1 - w) / 3. Every mix blends the same per-strategy positions, risk-scaled once more, exactly
as Portfolio does.

Choice: the share with the best adjusted Sharpe over dev + validation together (1995 - Oct 2014,
all data not reserved for test). The test block is scored once for the chosen share.

Writes YANNICK/output/portfolio_dashboard/data.json and prints the chosen weights.
Run: uv run python YANNICK/portfolio_dashboard.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from backtest.pipeline import risk_scale
from backtest.stats import adjusted_sharpe, sharpe
from data import MACRO, PRICES
from portfolio import Portfolio
from portfolio.blend import _weighted_sum
from portfolio.final import STRATEGIES

OUT = Path(__file__).resolve().parent / "output" / "portfolio_dashboard"
SHARES = np.round(np.arange(0, 0.625, 0.05), 2)
TRIPLE = ["slow_trend", "long_stocks_bonds", "macro"]
LABELS = {"tikhonov_garch": "Tikhonov + GARCH", "slow_trend": "Slow trend", "long_stocks_bonds": "Long stocks + bonds", "macro": "Macro"}


def asset_class(a: str) -> str:
    for k in ("stocks", "bonds", "titan_fx"):
        if a.endswith("_" + k):
            return {"titan_fx": "FX"}.get(k, k.capitalize())
    return "Commodities"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bt = Backtest(PRICES, MACRO)
    dates = pd.DatetimeIndex(pd.to_datetime(bt.prices["AsOfDate"].to_pandas()))
    n = len(dates)
    end = dates[-1] + pd.Timedelta(days=1)
    W = {"dev": (dates[0], dates[int(n * .75)]), "validation": (dates[int(n * .75)], dates[int(n * .9)]),
         "test": (dates[int(n * .9)], end), "dev + validation": (dates[0], dates[int(n * .9)]), "full": (dates[0], end)}

    t0 = time.time()
    positions = Portfolio(STRATEGIES).positions(bt.prices, bt.extras)
    print(f"positions ({time.time() - t0:.0f}s)")
    names = list(positions)

    def mix(weights: dict) -> pl.DataFrame:
        rows = np.tile([weights.get(k, 0.0) for k in names], (bt.prices.height, 1))
        m = _weighted_sum(positions, rows)
        cols = m.columns[1:]
        m = m.with_columns(pl.col(c).fill_nan(None) for c in cols)
        live = pl.any_horizontal(pl.col(c).is_not_null() for c in cols)
        m = m.with_columns(pl.when(live).then(pl.col(c).fill_null(0.0)).otherwise(None).alias(c) for c in cols)
        return risk_scale(m, bt.prices)

    def returns(frame) -> tuple[pd.Series, object]:
        res = bt.run_positions(frame, period="full", check_leakage=False)
        r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        return r, res

    def stats(r: pd.Series, w: str) -> float:
        a, b = W[w]
        return float(adjusted_sharpe(r[(r.index >= a) & (r.index < b)].dropna().to_numpy()))

    def weights_for(share: float) -> dict:
        return {"tikhonov_garch": float(share), **{k: (1 - share) / 3 for k in TRIPLE}}

    # 1. Share sweep (test not used for the choice)
    sweep = []
    for s in SHARES:
        r, _ = returns(mix(weights_for(s)))
        sweep.append({"tikhonov_share": float(s), **{w: stats(r, w) for w in ("dev", "validation", "dev + validation")}})
    best = max(sweep, key=lambda x: x["dev + validation"])
    share = best["tikhonov_share"]
    final_w = weights_for(share)
    print("share sweep (adjusted Sharpe):")
    print(pd.DataFrame(sweep).round(3).to_string(index=False))
    print(f"\nchosen Tikhonov share: {share:.0%}  ->  weights {', '.join(f'{k} {v:.4f}' for k, v in final_w.items())}")

    # 2. Final (leakage-checked through Portfolio), old final, triple, and each strategy
    final_p = Portfolio(STRATEGIES, final_w)
    final_res = bt.run_positions(final_p, period="full", name="final")  # leakage test on the whole blend
    rf = final_res.returns.to_pandas().set_index("AsOfDate")["returns"]
    rf.index = pd.to_datetime(rf.index)
    series = {"Final portfolio": (rf, final_res)}
    series["Old final 40/20/20/20"] = returns(mix(weights_for(0.4)))
    series["Triple only"] = returns(mix(weights_for(0.0)))
    for k in names:  # each strategy alone, scored the normal way (one risk scaling, as bt.run does)
        res = bt.run(STRATEGIES[k], period="full", check_leakage=False, name=k)
        r = res.returns.to_pandas().set_index("AsOfDate")["returns"]
        r.index = pd.to_datetime(r.index)
        series[LABELS[k]] = (r, res)

    summary = []
    for label, (r, res) in series.items():
        x = r.dropna()
        summary.append({"strategy": label, "weight": None if label not in LABELS.values() else final_w[[k for k, v in LABELS.items() if v == label][0]],
                        **{f"{w}_adj": stats(r, w) for w in W}, "full_sharpe": float(sharpe(x.to_numpy())),
                        "skew": float(x.skew()), "excess_kurtosis": float(x.kurt()), "holding_days": float(res.av_holding)})
    print("\n" + pd.DataFrame(summary).round(2).to_string(index=False))

    # 3. Curves (weekly), yearly, correlations, exposure by asset class
    curve, yearly = [], []
    for label in ["Final portfolio", "Old final 40/20/20/20"] + list(LABELS.values()):
        r = series[label][0]
        rr = r[r.index >= r.dropna().index[0]].fillna(0.0)
        cum = rr.cumsum()
        dd = cum - cum.cummax()
        roll = rr.rolling(252).mean() / rr.rolling(252).std() * np.sqrt(252)
        for i in range(0, len(cum), 5):
            curve.append({"strategy": label, "date": cum.index[i].strftime("%Y-%m-%d"), "cum_pnl": round(float(cum.iloc[i]), 3),
                          "drawdown": round(float(dd.iloc[i]), 3), "rolling_1y_sharpe": None if np.isnan(roll.iloc[i]) else round(float(roll.iloc[i]), 3)})
        for y_, x in rr.groupby(rr.index.year):
            if len(x) > 100:
                yearly.append({"strategy": label, "year": int(y_), "sharpe": float(sharpe(x.to_numpy()))})
    rets = pd.DataFrame({LABELS[k]: series[LABELS[k]][0] for k in names}).dropna()
    corr = [{"a": a, "b": b, "corr": float(rets[a].corr(rets[b]))} for a in rets for b in rets]

    pos = final_res.positions.to_pandas().set_index("AsOfDate")
    pos.index = pd.to_datetime(pos.index)
    gross = pos.abs()
    by_class = gross.T.groupby([asset_class(c) for c in gross.columns]).sum().T
    share_cls = by_class.div(by_class.sum(axis=1), axis=0).resample("ME").mean().dropna()
    exposure = [{"date": d.strftime("%Y-%m-%d"), "class": c, "share": round(float(v), 4)} for d, row in share_cls.iterrows() for c, v in row.items()]
    net_long = (pos.sum(axis=1) / gross.sum(axis=1)).resample("ME").mean().dropna()

    meta = {"weights": final_w, "share": share, "windows": {w: [str(a.date()), str((b - pd.Timedelta(days=1)).date())] for w, (a, b) in W.items()},
            "net_long_mean": float(net_long.mean())}
    data = {"meta": meta, "sweep": sweep, "summary": summary, "curve": curve, "yearly": yearly, "corr": corr, "exposure": exposure}
    clean = json.loads(json.dumps(data, default=float).replace("NaN", "null"))
    (OUT / "data.json").write_text(json.dumps(clean))
    print(f"\nwrote {OUT / 'data.json'}")


if __name__ == "__main__":
    main()
