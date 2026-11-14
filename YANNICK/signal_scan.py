"""Signal scan on the dev period: which simple features predict the return we actually earn?

The position decided at T earns the price change from T+1 to T+2, so every feature at T is compared
with z[T+2] = r[T+2] / sigma[T], the return two days later in units of the asset's own volatility.
A predictive correlation of 0.02-0.03 on daily data is already a usable edge.

Run: uv run python YANNICK/signal_scan.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from data import PRICES  # noqa: E402
prices = pd.read_csv(PRICES, index_col=0, parse_dates=True)
dev = prices.iloc[: int(len(prices) * 0.75)]
assets = list(dev.columns)

ret = dev.diff()  # price changes, as in the PDF's return formula
vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
z = ret / vol.shift(1)  # today's move in units of yesterday's volatility
target = ret.shift(-2) / vol  # what the position decided today earns


def ic(feature: pd.DataFrame) -> pd.Series:
    """Correlation of the feature with the earned return, per asset, plus the pooled value."""
    out = {a: feature[a].corr(target[a]) for a in assets}
    stacked = pd.concat([feature.stack(), target.stack()], axis=1).dropna()
    out["POOLED"] = stacked.iloc[:, 0].corr(stacked.iloc[:, 1])
    return pd.Series(out)


rows = {}
# 1. Reversal after a single day, and after a few days.
for h in (1, 2, 3, 5, 10):
    rows[f"reversal {h}d"] = ic(-z.rolling(h).sum())
# 2. Reversal only after extreme days: the overreaction story.
for k in (1.5, 2.0, 2.5):
    shock = z.where(z.abs() > k, 0.0)
    for h in (1, 5):
        rows[f"extreme>{k} rev {h}d"] = ic(-shock.rolling(h).sum())
# 3. Trend, for reference.
for span in (8, 16, 32, 64):
    fast = dev.ewm(span=span, adjust=False).mean()
    slow = dev.ewm(span=4 * span, adjust=False).mean()
    rows[f"trend {span}"] = ic((fast - slow) / (vol * np.sqrt(span)))
print("Information coefficient with the return earned (dev period)")
print(pd.DataFrame(rows).T.round(3).to_string())

# 4. Seasonality: average z-return by calendar month and weekday. Bigger than noise?
print("\nMonth of year: mean z-return per day (x100), and t-stat of the month effect")
month = target.groupby(target.index.month).mean() * 100
print(month.round(2).to_string())
n_years = dev.index.year.nunique()
print(f"(about {n_years} years; a month mean needs |value| > ~{200 / np.sqrt(21 * n_years):.1f} to be 2 s.e.)")
print("\nWeekday (of the earning day): mean z-return x100")
earn_day = pd.Series(target.index, index=target.index).shift(-2)
wd = target.groupby(earn_day.dt.dayofweek).mean() * 100
print(wd.round(2).to_string())

# 5. Lead-lag: does asset A's recent move predict asset B's earned return?
print("\nLead-lag: IC of A's last-5-day z-move with B's earned return (rows A, cols B)")
move = z.rolling(5).sum()
ll = pd.DataFrame({b: {a: move[a].corr(target[b]) for a in assets} for b in assets})
print(ll.round(3).to_string())

# 6. Trend exhaustion: does the trend payoff fade when the trend is very strong?
print("\nTrend 16 strength decile vs mean signed payoff (z x100): does it fade at the extremes?")
fast = dev.ewm(span=16, adjust=False).mean()
slow = dev.ewm(span=64, adjust=False).mean()
strength = ((fast - slow) / (vol * 4)).stack()
earned = target.stack()
both = pd.concat([strength.rename("s"), earned.rename("y")], axis=1).dropna()
both["bucket"] = pd.qcut(both["s"], 10, labels=False)
table = both.groupby("bucket").agg(s=("s", "mean"), y=("y", "mean"), n=("y", "size"))
table["y"] *= 100
print(table.round(3).to_string())

# 7. Relative value: does the spread between two assets' vol-normalised paths revert?
print("\nPair spread reversion: IC of -(cum z_A - cum z_B over 20d) with (z_A - z_B) earned")
for i, a in enumerate(assets):
    for b in assets[i + 1 :]:
        spread = z[a].rolling(20).sum() - z[b].rolling(20).sum()
        y = target[a] - target[b]
        print(f"  {a:>8} - {b:<8} {(-spread).corr(y):+.3f}   return corr {z[a].corr(z[b]):+.2f}")
