"""Every strategy we tried, scored the same way on the new data, written to YANNICK/output/all_results.json.

For each strategy: one full run (positions use only past data), scored on
    dev       first 75% of dates      (what everything was built on; the training data)
    val       next 15%                 (validation: choosing hyperparameters)
    test      last 10%                 (test: scored once)
plus dev-period figures from the scorer (gross Sharpe, skew, kurtosis, holding, position bias).
PDF scoring throughout. Leakage tests are run in each strategy's own script; here they are skipped
for speed, except where noted.

Run: uv run --with xgboost --with scikit-learn python YANNICK/all_results.py
"""

import json
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import Backtest  # noqa: E402
from backtest.stats import adjusted_sharpe, sharpe  # noqa: E402
from data import MACRO, PRICES  # noqa: E402
from portfolio import Portfolio  # noqa: E402
from strategies import trend  # noqa: E402

import book_strategies as bs  # noqa: E402
import edge_tests as et  # noqa: E402
import macro_strategy as ms  # noqa: E402
from oos_test import thirds as thirds_subset, ALL, FX  # noqa: E402

OUT = Path(__file__).resolve().parent / "output" / "all_results.json"


def P(f):
    """Make a prices-only strategy accept the macro extras."""
    def g(prices, *extras):
        return f(prices)
    g.__name__ = getattr(f, "__name__", "f")
    return g


def long_only(which):
    def f(prices, *extras):
        cols = [c for c in prices.columns if c != "AsOfDate"]
        return prices.with_columns(pl.lit(1.0 if c.endswith(which) else 0.0).alias(c) for c in cols)
    return f


def swing_he3(prices, *extras, n=120):
    px = prices.to_pandas().set_index("AsOfDate")
    lp = np.log(px)
    zz = (lp - lp.rolling(n).mean()) / lp.rolling(n).std()
    out = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    out["helium3"] = (-zz["helium3"]).clip(-2, 2)
    return pl.from_pandas(out.rename_axis("AsOfDate").reset_index())


def strategies() -> list[tuple[str, str, object, bool, str]]:
    """(group, name, function, is_positions, description)."""
    import fft_scan
    import ml_models
    import poker_bot
    import tikhonov
    import var_strategy
    import garch_timing
    import lag_tikhonov_wf

    slow = ms.slow_trend
    lsb = ms.long_stocks_bonds
    mp = ms.macro_prior
    thirds = Portfolio({"t": slow, "l": lsb, "m": mp}, {k: 1 / 3 for k in "tlm"})
    S = []
    add = lambda g, n, f, d, pos=False: S.append((g, n, f, pos, d))  # noqa: E731

    g = "Trend"
    add(g, "trend.py (spans 8/16/32)", P(trend.forecast), "Vote of 3 EWMA crossovers (fast vs 4x slower)")
    add(g, "Slow trend (16/32/64)", slow, "Same vote at slower speeds")
    add(g, "Trend 32/64/128", P(partial(trend.forecast, spans=(32, 64, 128))), "Even slower speeds")
    add(g, "Trend + position inertia 10%", None, "Skip trades under 10% of the position (Carver)")
    add(g, "Carver EWMAC, 6 speeds", P(bs.ewmac), "Continuous EWMAC 2/8 .. 64/256, scaled and capped")
    add(g, "Carver EWMAC 8-64", P(partial(bs.ewmac, fasts=(8, 16, 32, 64))), "Continuous EWMAC, 4 speeds")
    add(g, "Kaufman N-day breakout", P(bs.breakout), "Position in the 20-160 day range")
    add(g, "Kaufman N-day momentum", P(bs.momentum), "Sign of the 20-250 day change")
    add(g, "Kaufman regression slope", P(bs.regression_slope), "t-stat of the price slope, 20-160 days")
    add(g, "Kaufman trend x efficiency ratio", P(bs.trend_er_filter), "Trend sized by how clean the trend is")
    add(g, "Kaufman KAMA", P(bs.kama), "Adaptive moving average slope")
    add(g, "Meta-labelled trend (folding)", P(bs.meta_labelled), "Logistic model decides bet size on trend's side")
    add(g, "Tikhonov lag filter (walk-forward)", lag_tikhonov_wf.walk_forward, "Linear filter of 250 past returns, smooth Tikhonov, refit yearly")
    add(g, "Poker bot (online Bayesian)", P(poker_bot.poker_bot), "Learns payoff of trend/shock/season per asset")

    g = "Reversion & patterns"
    add(g, "5-day reversal", P(et.reversal), "Bet against the last 5 days' move")
    add(g, "Extreme-day reversal (>2σ, 5d)", P(partial(et.extreme_reversal, k=2.0, hold=5)), "Bet against shocks for 5 days")
    add(g, "Regime switcher", P(et.regime), "Trend or reversal per asset by past autocorrelation")
    add(g, "Swing trade helium3", swing_he3, "Bet against distance from 120-day mean")
    add(g, "Commodity pair spreads (60d)", ms.commodity_pairs, "Spread reversion on 3 commodity pairs")
    add(g, "Commodity pair spreads (120d)", partial(ms.commodity_pairs, n=120), "Same, slower")
    add(g, "Seasonality (month of year)", P(et.seasonal), "Same month in earlier years")
    add(g, "Cross-sectional ranking", P(bs.cross_sectional), "Long leaders, short laggards (WorldQuant)")
    add(g, "FFT cycle forecast", P(partial(fft_scan.fft_forecast, window=256, k=3)), "Extend the 3 strongest waves")

    g = "Risk premia"
    add(g, "Long stocks + bonds", lsb, "Always long, vol-scaled")
    add(g, "Long bonds", long_only(("bonds",)), "Always long bonds")
    add(g, "Long stocks", long_only(("stocks",)), "Always long stocks")

    g = "Macro & lead-lag"
    add(g, "Macro prior, all classes", mp, "Textbook signs: growth/inflation acceleration")
    add(g, "Macro prior, stocks only", partial(mp, classes=("stocks",)), "Long stocks when growth accelerates")
    add(g, "Macro prior, bonds only", partial(mp, classes=("bonds",)), "Short bonds when inflation accelerates")
    add(g, "Macro prior, commodities only", partial(mp, classes=("commodity",)), "Long when world growth/inflation accelerate")
    add(g, "Macro prior, FX only", partial(mp, classes=("titan_fx",)), "Relative growth vs titan")
    add(g, "Macro walk-forward ridge", ms.macro_walk_forward, "Ridge on macro features, refit yearly")
    add(g, "Tikhonov feature regression", tikhonov.tikhonov, "Macro+trend+moves, λ from L-curve, refit yearly")
    add(g, "VAR daily (T+2), followers", P(var_strategy.var_daily), "Leaders predict followers; earnable horizon")
    add(g, "Seb's weekly VAR", P(var_strategy.seb_weekly), "Weekly VAR on stocks and bonds")

    g = "ML"
    add(g, "Ridge, 20-day target", P(partial(ml_models.walk_forward, model="ridge", horizon=20)), "12 features, pooled, refit yearly")
    add(g, "XGBoost, 5-day target", P(partial(ml_models.walk_forward, model="xgb", horizon=5)), "Same features, shallow trees")

    g = "Blends"
    add(g, "Base: slow trend 50 / long 50", Portfolio({"t": slow, "l": lsb}, {"t": .5, "l": .5}), "Trend + risk premium", True)
    add(g, "35 / 35 / 30 (trend / long / macro)", Portfolio({"t": slow, "l": lsb, "m": mp}, {"t": .35, "l": .35, "m": .3}), "", True)
    add(g, "Equal thirds (RECOMMENDED)", thirds, "Slow trend + long stocks/bonds + macro, 1/3 risk each", True)
    add(g, "Equal thirds, no FX", thirds_subset([c for c in ALL if c not in FX]), "Drop the 4 FX rates", True)
    add(g, "Equal thirds x GARCH timing", garch_timing.thirds("garch"), "Søren's GARCH vol timing on all parts", True)
    add(g, "Thirds + commodity pairs 10%", Portfolio({"t": slow, "l": lsb, "m": mp, "p": ms.commodity_pairs}, {"t": .3, "l": .3, "m": .3, "p": .1}), "", True)
    add(g, "Old blend 70/15/15 (trend/reversal/season)", Portfolio({"t": P(trend.forecast), "e": P(partial(et.extreme_reversal, k=2.0, hold=5)), "s": P(et.seasonal)}, {"t": .7, "e": .15, "s": .15}), "Best on the old example data", True)
    return S


def inertia(prices, *extras):
    return bs.with_inertia(trend.forecast)(prices)


def main() -> None:
    bt = Backtest(PRICES, MACRO)
    dates = pd.to_datetime(bt.prices["AsOfDate"].to_pandas())
    n = len(dates)
    W = {"dev": (0, int(n * .75)), "val": (int(n * .75), int(n * .9)), "test": (int(n * .9), n)}  # non-overlapping 75/15/10
    rows, daily = [], {}
    for group, name, f, is_pos, desc in strategies():
        t0 = time.time()
        try:
            if f is None:  # inertia is a positions function
                f, is_pos = inertia, True
            run = bt.run_positions if is_pos else bt.run
            full = run(f, period="full", check_leakage=False, name=name)
            dev = run(f, period="dev", check_leakage=False, name=name)
            r = full.returns.to_pandas().set_index("AsOfDate")["returns"]
            r.index = pd.to_datetime(r.index)
            daily[name] = r
            row = {"group": group, "name": name, "desc": desc}
            for w, (a, b) in W.items():
                end = dates[b] if b < n else dates.iloc[-1] + pd.Timedelta(days=1)
                x = r[(r.index >= dates[a]) & (r.index < end)].dropna()
                row[f"{w}_adj"] = float(adjusted_sharpe(x.to_numpy())) if len(x) > 50 else None
                row[f"{w}_sharpe"] = float(sharpe(x.to_numpy())) if len(x) > 50 else None
            fig = dev.figures
            row.update({"dev_gross": fig["sharpe_ex_slippage"], "dev_skew": fig["skewness"], "dev_kurt": fig["kurtosis"],
                        "dev_hold": fig["av_holding"], "dev_bias": fig["position_bias"], "dev_lag1": fig["sharpe_1d_lag"]})
            rows.append(row)
            print(f"  {time.time() - t0:5.0f}s  {name}: dev {row['dev_adj']:+.2f}  test {row['test_adj']:+.2f}", flush=True)
        except Exception:
            print(f"  FAILED {name}\n{traceback.format_exc()}", flush=True)
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()} for row in rows]
    meta = {"windows": {w: [str(dates[a].date()), str(dates[min(b, n - 1)].date())] for w, (a, b) in W.items()},
            "instruments": len(ALL), "scoring": "pdf"}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "rows": clean}, indent=1))
    pd.DataFrame(daily).rename_axis("AsOfDate").to_csv(OUT.with_name("all_returns.csv"))  # daily net returns, for block validation
    print(f"wrote {OUT} and {OUT.with_name('all_returns.csv')}")


if __name__ == "__main__":
    main()
