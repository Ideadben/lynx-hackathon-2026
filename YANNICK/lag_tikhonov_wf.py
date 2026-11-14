"""Walk-forward version of the Tikhonov lag filter (see lag_tikhonov.py).

Each January the filter is refit on all earlier years, using only rows whose 20-day label is already
known, with the Γ and λ chosen on the 75/15/10 validation block (ridge, 2.37e5). Trading starts
after 5 years of history. Passes the backtester's leakage test.
"""

import numpy as np
import pandas as pd
import polars as pl

import lag_tikhonov as lt

LAM = 2.37e5  # chosen on the 75/15/10 validation block (split_v2.py)
G = lt.gamma("ridge")


def walk_forward(prices: pl.DataFrame, *extras: pl.DataFrame) -> pl.DataFrame:
    p = prices.to_pandas().set_index("AsOfDate")
    d = pd.to_datetime(p.index)
    zz = lt.z_returns(p)
    yy = sum(zz.shift(-k) for k in range(2, 2 + lt.H))
    gap = lt.H + 2
    out = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for year in sorted(set(d.year)):
        if year - d.year[0] < 5:
            continue
        start = int(np.argmax(d.year == year))
        XtX, Xty, _, _ = lt.build(zz, yy, np.arange(0, start - gap))
        w = np.linalg.solve(XtX + LAM * G.T @ G, Xty)
        f = lt.filter_forecast(prices, w).to_pandas().set_index("AsOfDate")
        out[d.year == year] = f[d.year == year].to_numpy()
    return pl.from_pandas(out.rename_axis("AsOfDate").reset_index())
