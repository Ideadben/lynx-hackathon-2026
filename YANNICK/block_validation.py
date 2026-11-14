"""Block validation for every strategy: Sharpe per calendar year, from the daily returns that
all_results.py saves (YANNICK/output/all_returns.csv).

Dev years 1996-2010 are summarised (mean, spread, share of positive years, worst year); 2011 is
split between dev and validation, so the validation (Jun 2011 - Oct 2014) and test (Oct 2014 - 2016)
blocks are shown as their own columns. Writes YANNICK/output/block_validation.csv.

Run: uv run python YANNICK/block_validation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.stats import sharpe

OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    r = pd.read_csv(OUT / "all_returns.csv", index_col=0, parse_dates=True)
    meta = json.loads((OUT / "all_results.json").read_text())
    groups = {row["name"]: row["group"] for row in meta["rows"]}
    w = meta["meta"]["windows"]
    yearly = r.groupby(r.index.year).apply(lambda g: g.apply(lambda s: float(sharpe(s.dropna().to_numpy())) if s.notna().sum() > 100 else np.nan))
    dev_years = [y for y in yearly.index if 1996 <= y <= 2010]
    dev = yearly.loc[dev_years]
    block = lambda a, b: r[(r.index >= a) & (r.index < b)].apply(lambda s: float(sharpe(s.dropna().to_numpy())))  # noqa: E731
    summary = pd.DataFrame({
        "group": pd.Series(groups),
        "dev yrs": dev.notna().sum(),
        "mean/yr": dev.mean(),
        "spread": dev.std(),
        "% yrs > 0": (dev > 0).sum() / dev.notna().sum() * 100,
        "worst yr": dev.min(),
        "worst": dev.idxmin(),
        "val block": block(w["val"][0], w["val"][1]),
        "test block": block(w["test"][0], pd.Timestamp(w["test"][1]) + pd.Timedelta(days=1)),
    })
    summary["t (mean/yr)"] = summary["mean/yr"] / (summary["spread"] / np.sqrt(summary["dev yrs"]))
    summary = summary.sort_values("mean/yr", ascending=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    print("Block validation: Sharpe per calendar year (dev years 1996-2010), then the validation and test blocks")
    print(summary.round({"mean/yr": 2, "spread": 2, "% yrs > 0": 0, "worst yr": 2, "val block": 2, "test block": 2, "t (mean/yr)": 1}).to_string())
    full = yearly.T.loc[summary.index]
    full.insert(0, "group", summary["group"])
    full.round(2).to_csv(OUT / "block_validation.csv")
    print("\nSharpe per year for selected strategies:")
    pick = ["Equal thirds", "35 / 35 / 30 (trend / long / macro)", "Long stocks + bonds", "Macro prior, all classes", "Slow trend (16/32/64)",
            "trend.py (spans 8/16/32)", "Tikhonov lag filter (walk-forward)", "Seasonality (month of year)"]
    print(yearly[[p for p in pick if p in yearly]].T.round(1).to_string())
    print(f"\nwrote {OUT / 'block_validation.csv'} (all strategies x all years)")


if __name__ == "__main__":
    main()
