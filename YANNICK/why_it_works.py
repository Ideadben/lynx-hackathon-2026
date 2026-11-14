"""Why did the things that worked, work? Dev period only.

A 'hand' is one asset on one day where trend has a side. Its payoff is the vol-normalised move the
trend position earns: side * z[t+2] (x100 in the tables). Everything here is diagnosis, not a strategy.

1. Trend: where its payoff comes from (strength, assets, long vs short, horizon).
2. Folding (meta-labelling): which hands it folded and what those hands paid.
3. Efficiency-ratio sizing: the same question for Kaufman's filter.
4. Extreme-day reversal: an event study of what happens after a >2 sigma day.
5. Seasonality: is the month pattern stable across years, or one lucky stretch?
6. The blend: why weak, uncorrelated pieces raise the Sharpe ratio.

Run: uv run python YANNICK/why_it_works.py
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import Backtest  # noqa: E402
from book_strategies import _ret, efficiency_ratio, meta_labelled  # noqa: E402
from edge_tests import extreme_reversal, seasonal  # noqa: E402
from strategies import trend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from data import PRICES  # noqa: E402


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def table(payoff: pd.DataFrame, key: pd.DataFrame, label: str) -> pd.DataFrame:
    """Mean payoff, its t-stat, a per-hand Sharpe and the share of hands, per bucket of `key`."""
    df = pd.concat([payoff.stack().rename("p"), key.stack().rename("k")], axis=1).dropna()
    g = df.groupby("k")["p"]
    out = pd.DataFrame({
        "hands %": g.size() / len(df) * 100,
        "mean x100": g.mean() * 100,
        "t": g.apply(t_stat),
        "annual Sharpe/hand": g.mean() / g.std() * np.sqrt(252),
    })
    out.index.name = label
    return out.round(2)


def main() -> None:
    pd.set_option("display.width", 200)
    bt = Backtest(PRICES)
    prices = bt.prices.head(int(bt.prices.height * 0.75))  # dev only
    px = prices.to_pandas().set_index("AsOfDate")
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    z = ret / vol.shift(1)
    earned = z.shift(-2)  # what a position decided today earns
    tf = trend.forecast(prices).to_pandas().set_index("AsOfDate")
    side = np.sign(tf).where(tf.ne(0))
    payoff = side * earned

    # 1. Trend ---------------------------------------------------------------------------------------
    print("=" * 100, "\n1. TREND: where does the payoff come from?\n")
    print(table(payoff, tf.round(2), "vote (agreement of the 3 speeds)"))
    print("\nLong vs short hands")
    print(table(payoff, side.replace({1.0: "long", -1.0: "short"}), "side"))
    print("\nPer asset")
    print(table(payoff, pd.DataFrame({a: a for a in px.columns}, index=px.index), "asset"))
    print("\nHow long the edge lasts: mean side * z[t+k] (x100), k days after the decision")
    print(pd.Series({k: (side * z.shift(-k)).stack().mean() * 100 for k in (1, 2, 5, 10, 20, 40, 60)}).round(2).to_string())

    # 2. Folding -------------------------------------------------------------------------------------
    print("\n" + "=" * 100, "\n2. FOLDING (meta-labelled trend): which hands were folded, and what did they pay?\n")
    meta = meta_labelled(prices).to_pandas().set_index("AsOfDate")
    size = (meta.abs()).where(side.notna() & meta.notna())
    bucket = pd.cut(size.stack(), [-0.001, 0.0, 0.25, 0.5, 1.0, 10], labels=["fold", "0-0.25", "0.25-0.5", "0.5-1", ">1"])
    print(table(payoff.where(size.notna()), bucket.unstack(), "bet size"))
    played = payoff.where(size > 0)
    folded = payoff.where(size == 0)
    allh = payoff.where(size.notna())
    s = lambda x: x.stack().mean() / x.stack().std() * np.sqrt(252)  # noqa: E731
    print(f"\nPer-hand Sharpe: all hands {s(allh):.2f}, played only {s(played):.2f}, folded only {s(folded):.2f}")
    print("\nWhat drives the fold decision? Trend payoff by quintile of each input the model sees:")
    er = efficiency_ratio(px, 40)
    vr = vol / vol.rolling(250, min_periods=100).mean()
    for name, feat in {"efficiency ratio (40d)": er, "vol / its 1y average": vr, "|trend vote|": tf.abs()}.items():
        q = feat.where(side.notna()).stack()
        qb = pd.qcut(q.rank(method="first"), 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]).unstack()
        print(f"\n  {name}")
        print(table(payoff, qb, "quintile").to_string().replace("\n", "\n  "))

    # 3. ER sizing -----------------------------------------------------------------------------------
    print("\n" + "=" * 100, "\n3. TREND x EFFICIENCY RATIO: does a clean trend pay more per unit of risk?\n")
    rel = er / er.expanding(min_periods=250).mean()
    qb = pd.qcut(rel.where(side.notna()).stack().rank(method="first"), 5,
                 labels=["Q1 noisy", "Q2", "Q3", "Q4", "Q5 clean"]).unstack()
    print(table(payoff, qb, "ER vs own history"))

    # 4. Extreme-day reversal ------------------------------------------------------------------------
    print("\n" + "=" * 100, "\n4. EXTREME DAYS: what happens after a move larger than 2 sigma?\n")
    shock = z.where(z.abs() > 2)
    sgn = np.sign(shock)
    rows = {}
    for k in range(1, 11):
        after = (-sgn * z.shift(-k)).stack()  # positive = the move reverses
        rows[k] = {"mean reversal x100": after.mean() * 100, "t": t_stat(after)}
    ev = pd.DataFrame(rows).T
    ev["cumulative x100"] = ev["mean reversal x100"].cumsum()
    print(f"{int(shock.notna().sum().sum())} shocks. Day k after the shock (the position starts earning on day 2):")
    print(ev.round(2).to_string())
    print("\nDoes the reversal pay when trend loses? Daily returns of each strategy, by trend's daily result:")
    r_t = _ret(bt.run(trend.forecast, name="t"))
    r_e = _ret(bt.run(partial(extreme_reversal, k=2.0, hold=5), name="e"))
    r_s = _ret(bt.run(seasonal, name="s"))
    both = pd.concat({"trend": r_t, "reversal": r_e, "seasonal": r_s}, axis=1, join="inner")
    both = both[both.index <= px.index[-1]]
    both["trend day"] = pd.qcut(both["trend"], 5, labels=["worst 20%", "Q2", "Q3", "Q4", "best 20%"])
    print(both.groupby("trend day", observed=True)[["trend", "reversal", "seasonal"]].mean().round(3).to_string())
    dd = (both["trend"].cumsum() - both["trend"].cumsum().cummax()) < -both["trend"].std() * 10
    print(f"\nIn trend drawdowns ({dd.mean():.0%} of days): mean daily return trend {both.loc[dd, 'trend'].mean():+.3f}, "
          f"reversal {both.loc[dd, 'reversal'].mean():+.3f}, seasonal {both.loc[dd, 'seasonal'].mean():+.3f}")

    # 5. Seasonality ---------------------------------------------------------------------------------
    print("\n" + "=" * 100, "\n5. SEASONALITY: same months every year, or one lucky stretch?\n")
    m = earned.groupby([earned.index.year, earned.index.month]).mean()
    yrs = sorted(set(earned.index.year))
    odd, even = m.loc[[y for y in yrs if y % 2]], m.loc[[y for y in yrs if not y % 2]]
    print("Correlation of the 12 monthly means between odd and even years (a stable pattern repeats):")
    print(pd.Series({a: odd[a].groupby(level=1).mean().corr(even[a].groupby(level=1).mean()) for a in px.columns}).round(2).to_string())
    early = m.loc[[y for y in yrs if y < 2009]]
    late = m.loc[[y for y in yrs if y >= 2009]]
    print("\nSame, first half of the years vs second half:")
    print(pd.Series({a: early[a].groupby(level=1).mean().corr(late[a].groupby(level=1).mean()) for a in px.columns}).round(2).to_string())

    # 6. Blend ---------------------------------------------------------------------------------------
    print("\n" + "=" * 100, "\n6. THE BLEND: diversification arithmetic\n")
    rets = both[["trend", "reversal", "seasonal"]]
    sr = rets.mean() / rets.std() * np.sqrt(252)
    c = rets.corr()
    print("Sharpe on common days:", sr.round(2).to_dict())
    print("Correlations:\n", c.round(2).to_string())
    w = np.array([0.7, 0.15, 0.15])
    # equal-risk pieces, weighted: Sharpe = w.SR / sqrt(w' C w)
    pred = w @ sr.to_numpy() / np.sqrt(w @ c.to_numpy() @ w)
    print(f"\nPredicted Sharpe of 70/15/15 on equal-risk pieces: {pred:.2f}   (trend alone {sr['trend']:.2f})")
    print("Rule: adding a piece helps if its Sharpe > (its correlation with the blend) x (blend Sharpe).")
    print(f"For reversal: {sr['reversal']:.2f} vs {c.loc['reversal', 'trend']:.2f} x {sr['trend']:.2f} "
          f"= {c.loc['reversal', 'trend'] * sr['trend']:.2f}  -> {'helps' if sr['reversal'] > c.loc['reversal', 'trend'] * sr['trend'] else 'hurts'}")
    print(f"For seasonal: {sr['seasonal']:.2f} vs {c.loc['seasonal', 'trend']:.2f} x {sr['trend']:.2f} "
          f"= {c.loc['seasonal', 'trend'] * sr['trend']:.2f}  -> {'helps' if sr['seasonal'] > c.loc['seasonal', 'trend'] * sr['trend'] else 'hurts'}")


if __name__ == "__main__":
    main()
