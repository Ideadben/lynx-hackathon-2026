"""The 'poker bot': bet in proportion to the expected value it has learned so far.

How a poker bot thinks, and the piece of this strategy that does it:

- Hands: each day, each asset shows a few signals (trend strength, a recent shock, the season).
- Prior: before seeing any results, the bot believes trend pays a little and the rest pay nothing.
- Population read: it learns how much each signal pays across all assets (pooled regression).
- Player read: each asset may deviate from the population, but only as far as its own evidence
  supports (shrinkage towards the pooled estimate).
- Online: it refits every month on results that were already known at the time, so it never peeks.
- Bet sizing: the forecast is the expected vol-adjusted return. The pipeline divides it by
  volatility, which makes the position mu / sigma^2, the Kelly bet.

Run: uv run python YANNICK/poker_bot.py
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import Backtest  # noqa: E402
from backtest.stats import paired_bootstrap  # noqa: E402
from edge_tests import extreme_reversal, seasonal  # noqa: E402
from portfolio import Portfolio  # noqa: E402
from strategies import trend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
from data import PRICES  # noqa: E402
FEATURES = ["trend", "shock", "season"]


def features(prices: pl.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Three signals per asset, each scaled to roughly unit size, plus the vol-normalised returns."""
    px = prices.to_pandas().set_index("AsOfDate")
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    z = ret / vol.shift(1)
    strength = []
    for span in (8, 16, 32):
        fast = px.ewm(span=span, adjust=False, min_periods=span).mean()
        slow = px.ewm(span=4 * span, adjust=False, min_periods=4 * span).mean()
        strength.append(((fast - slow) / (vol * np.sqrt(span))).clip(-2, 2))
    feats = {
        "trend": sum(strength) / len(strength),
        "shock": extreme_reversal(prices, k=2.0, hold=5).to_pandas().set_index("AsOfDate") / 3,
        "season": seasonal(prices).to_pandas().set_index("AsOfDate") / 3,
    }
    return z, feats


def poker_bot(
    prices: pl.DataFrame,
    pool_strength: float = 1000.0,
    asset_strength: float = 3000.0,
    trend_prior: float = 0.02,
    refit: int = 21,
    use: tuple[str, ...] = tuple(FEATURES),
) -> pl.DataFrame:
    """Expected vol-adjusted return two days ahead, learned online with shrinkage.

    Args:
        pool_strength: how many days of evidence the prior is worth, for the pooled estimate.
        asset_strength: how many days of an asset's own evidence it takes to move halfway from
            the pooled estimate. Infinity means every asset uses the pooled estimate.
        trend_prior: the prior payoff per unit of trend strength.
        refit: refit every this many days.
        use: which signals the bot may play.
    """
    z, feats = features(prices)
    assets = list(z.columns)
    n, p = len(z), len(use)
    X = np.stack([feats[f][assets].to_numpy() for f in use], axis=-1)  # (n, assets, p)
    X = np.nan_to_num(X)
    y = z.shift(-2).to_numpy()  # what a position decided at t earns; known at t + 2
    prior = np.array([trend_prior if f == "trend" else 0.0 for f in use])

    xx = np.zeros((len(assets), p, p))
    xy = np.zeros((len(assets), p))
    seen = 0  # rows whose outcome has been added
    beta = np.tile(prior, (len(assets), 1))
    out = np.full((n, len(assets)), np.nan)
    eye = np.eye(p)
    for t in range(n):
        if t % refit == 0:
            known = t - 2  # outcomes y[s] for s <= t - 2 are known at the end of day t
            for s in range(seen, max(known + 1, seen)):
                ok = ~np.isnan(y[s])
                xs = X[s][ok]
                xx[ok] += xs[:, :, None] * xs[:, None, :]
                xy[ok] += xs * y[s][ok][:, None]
            seen = max(seen, known + 1)
            pooled = np.linalg.solve(xx.sum(0) + pool_strength * eye, xy.sum(0) + pool_strength * prior)
            for j in range(len(assets)):
                if np.isinf(asset_strength):
                    beta[j] = pooled
                else:
                    beta[j] = np.linalg.solve(
                        xx[j] + asset_strength * eye, xy[j] + asset_strength * pooled
                    )
        out[t] = (X[t] * beta).sum(-1)
    out[np.isnan(z.to_numpy())] = np.nan
    frame = pd.DataFrame(out, index=z.index, columns=assets)
    frame.iloc[:150] = np.nan  # warm-up: signals not ready
    return pl.from_pandas(frame.rename_axis("AsOfDate").reset_index())


def learned_weights(prices: pl.DataFrame, **kwargs) -> pd.DataFrame:
    """The bot's final read, per asset: payoff per unit of each signal (x100)."""
    z, feats = features(prices)
    y = z.shift(-2)
    rows = {}
    for a in z.columns:
        X = np.nan_to_num(np.column_stack([feats[f][a].to_numpy() for f in FEATURES]))
        ok = ~np.isnan(y[a].to_numpy())
        rows[a] = np.linalg.lstsq(X[ok], y[a].to_numpy()[ok], rcond=None)[0] * 100
    return pd.DataFrame(rows, index=FEATURES).T.round(2)


def main() -> None:
    bt = Backtest(PRICES)
    base = bt.run(trend.forecast, name="trend")
    print(f"trend baseline: adjusted Sharpe {base.adjusted_sharpe:.2f}")

    print("\nWhat the bot would learn from all of dev (no shrinkage), payoff x100 per unit signal:")
    dev = bt.prices.head(int(bt.prices.height * 0.75))
    print(learned_weights(dev))

    print("\nSweep: how much it trusts each asset's own record, and the prior")
    print(
        bt.sweep(
            poker_bot,
            asset_strength=[500.0, 3000.0, float("inf")],
            trend_prior=[0.0, 0.02],
        ).select(["asset_strength", "trend_prior", "adjusted_sharpe", "sharpe", "av_holding"])
    )

    variants = {
        "trend only (continuous)": partial(poker_bot, use=("trend",)),
        "bot: all three": poker_bot,
        "bot: pooled only": partial(poker_bot, asset_strength=float("inf")),
    }
    results = {name: bt.run(f, name=name) for name, f in variants.items()}
    print()
    for name, r in results.items():
        diff = paired_bootstrap(r.returns, base.returns)
        print(f"{name:<26} adj {r.adjusted_sharpe:.2f}  hold {r.av_holding:.0f}d  vs trend: {diff}")

    best = results["bot: all three"]
    print("\nBy year (bot vs trend)")
    print(pd.DataFrame({"bot": best.by_year().to_pandas().set_index("year")["sharpe"],
                        "trend": base.by_year().to_pandas().set_index("year")["sharpe"]}).round(2).T)
    lynx = Backtest(PRICES, scoring="lynx")
    print(f"\nLynx costs: bot {lynx.run(poker_bot).adjusted_sharpe:.2f}, "
          f"trend {lynx.run(trend.forecast).adjusted_sharpe:.2f}")

    hindsight = Portfolio(
        {"trend": trend.forecast, "extreme": partial(extreme_reversal, k=2.0, hold=5), "seasonal": seasonal},
        {"trend": 0.6, "extreme": 0.2, "seasonal": 0.2},
    )
    print(f"For reference, the hand-picked 60/20/20 blend: {bt.run_positions(hindsight).adjusted_sharpe:.2f}")


if __name__ == "__main__":
    main()
