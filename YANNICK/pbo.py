"""Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu, 2015), via CSCV.

Question: if we pick the strategy with the best in-sample Sharpe out of everything we tried, how
often does that pick land in the bottom half out of sample?

Combinatorially symmetric cross-validation: cut the days into S blocks, and for every way of choosing
S/2 blocks as 'in-sample' (the rest out-of-sample), pick the in-sample winner and record its
out-of-sample rank. PBO is the share of splits where the winner ranks below the median out of sample.
PBO near 0: selection works. Near 0.5: selection is a coin flip. Above 0.5: selection hurts.

Run: uv run python YANNICK/pbo.py
"""

import sys
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import Backtest  # noqa: E402
from book_strategies import STRATEGIES, _ret  # noqa: E402
from edge_tests import extreme_reversal, seasonal  # noqa: E402
from portfolio import Portfolio  # noqa: E402
from strategies import trend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from data import PRICES  # noqa: E402


def pbo(returns: pd.DataFrame, blocks: int = 16) -> tuple[float, pd.Series]:
    """PBO and the distribution of the logit of the winner's out-of-sample relative rank."""
    x = returns.dropna().to_numpy()
    edges = np.linspace(0, len(x), blocks + 1).astype(int)
    parts = [x[a:b] for a, b in zip(edges[:-1], edges[1:])]
    n = x.shape[1]
    logits = []
    for chosen in combinations(range(blocks), blocks // 2):
        ins = np.vstack([parts[i] for i in chosen])
        oos = np.vstack([parts[i] for i in range(blocks) if i not in chosen])
        sr_in = ins.mean(0) / ins.std(0)
        sr_out = oos.mean(0) / oos.std(0)
        best = np.argmax(sr_in)
        rank = (sr_out < sr_out[best]).sum() + 1  # 1 = worst, n = best
        w = rank / (n + 1)
        logits.append(np.log(w / (1 - w)))
    logits = pd.Series(logits)
    return float((logits <= 0).mean()), logits


def main() -> None:
    bt = Backtest(PRICES)
    candidates = dict(STRATEGIES)
    candidates.update(
        {
            "extreme reversal": partial(extreme_reversal, k=2.0, hold=5),
            "seasonal": seasonal,
            "blend 70/15/15": Portfolio(
                {"t": trend.forecast, "e": partial(extreme_reversal, k=2.0, hold=5), "s": seasonal},
                {"t": 0.7, "e": 0.15, "s": 0.15},
            ),
        }
    )
    rets = {}
    for name, f in candidates.items():
        r = bt.run_positions(f, name=name) if isinstance(f, Portfolio) else bt.run(f, name=name)
        rets[name] = _ret(r)
    frame = pd.concat(rets, axis=1, join="inner")
    print(f"{frame.shape[1]} strategies, {len(frame)} common days ({frame.index[0]} to {frame.index[-1]})\n")
    sr = frame.mean() / frame.std() * np.sqrt(252)
    print("Sharpe on the common days:")
    print(sr.sort_values(ascending=False).round(2).to_string())

    for label, cols in {
        "all candidates": list(frame.columns),
        "trend variants only (EWMAC, breakout, slope, ER, KAMA, momentum, meta)": [
            c for c in frame.columns if c not in ("extreme reversal", "seasonal", "blend 70/15/15",
                                                  "WorldQuant cross-sectional")
        ],
    }.items():
        p, logits = pbo(frame[cols])
        print(f"\nPBO, {label}: {p:.2f}  (median logit {logits.median():+.2f}; "
              f"{len(logits)} splits)")


if __name__ == "__main__":
    main()
