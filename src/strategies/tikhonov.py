"""Tikhonov lag filter: a linear fit of past returns, regularised so it doesn't fit noise.

The forecast for each asset is a weighted sum of its last ``L`` vol-normalised daily price changes:

    forecast[t] = sum_{j < L} w_j * z[t - j]

The weights ``w`` are shared by all assets and fitted to the move a position actually earns, the
next ``H`` days after the two-day lag, ``y[t] = z[t+2] + ... + z[t+1+H]``, by minimising

    ||X w - y||^2 + lam * ||G w||^2

``G`` is the "smooth" Tikhonov matrix (second differences of ``w`` plus a little identity), so
neighbouring days get similar weights and the filter comes out as a smooth curve. ``lam`` is
``alpha`` times the average diagonal of ``X'X`` per year, so ``alpha`` means the same penalty
strength whatever the size of the data.

Walk-forward: every January the weights are refitted on all earlier years. The last ``H + 2``
days before the refit are dropped, because their targets reach into the year being traded. The
first ``MIN_YEARS`` years only train. Defaults: L = 250, H = 5, second-order (smooth) G from the
Tikhonov lab (YANNICK/tikhonov_features.py); alpha = 46, chosen on the validation block among the
full alpha sweep (YANNICK/tikhonov_lagweight.py), since the dev blocks could not tell alphas apart.
alpha = 46 gives about 20 effective parameters; the earlier default alpha = 1000 gave about 7.

``forecast_garch`` is the same forecast times the GARCH vol-timing multiplier from
``strategies.garch``: exposure is cut when GARCH expects more turbulence than usual. The learned
filter itself puts about 3x less weight on past returns in turbulent markets, so this is where
the GARCH risk management belongs.
"""

import numpy as np
import polars as pl

from strategies import garch

L = 250
H = 5
ALPHA = 46.0
MIN_YEARS = 5
VOL_WINDOW = 100
DATE = "AsOfDate"


def _assets(prices: pl.DataFrame) -> list[str]:
    return [c for c in prices.columns if c != DATE]


def z_returns(prices: pl.DataFrame) -> np.ndarray:
    """Daily price change divided by the trailing volatility known the day before, capped at ±6."""
    px = prices.select(_assets(prices)).to_numpy().astype(float)
    d = np.vstack([np.full((1, px.shape[1]), np.nan), np.diff(px, axis=0)])
    sq = np.where(np.isfinite(d), d * d, np.nan)
    vol = np.full_like(d, np.nan)
    for t in range(len(d)):
        window = sq[max(0, t - VOL_WINDOW + 1) : t + 1]
        n = np.sum(np.isfinite(window), axis=0)
        with np.errstate(invalid="ignore"):
            vol[t] = np.where(n >= VOL_WINDOW // 2, np.sqrt(np.nansum(window, axis=0) / np.maximum(n, 1)), np.nan)
    z = np.full_like(d, np.nan)
    z[1:] = d[1:] / vol[:-1]
    return np.clip(z, -6, 6)


def smooth_gamma(lags: int) -> np.ndarray:
    """Second differences of the weights (scaled by 10) stacked on a small identity."""
    D = np.zeros((lags - 2, lags))
    for i in range(lags - 2):
        D[i, i : i + 3] = (1, -2, 1)
    return np.vstack([10 * D, 0.1 * np.eye(lags)])


def _lagged(z: np.ndarray, lags: int) -> np.ndarray:
    """X[t, k, j] = z[t - j, k]: the last ``lags`` values for every day and asset (NaN if missing)."""
    n, m = z.shape
    X = np.full((n, m, lags), np.nan)
    for j in range(lags):
        X[j:, :, j] = z[: n - j]
    return X


def weights_by_year(prices: pl.DataFrame, lags: int = L, horizon: int = H, alpha: float = ALPHA,
                    min_years: int = MIN_YEARS) -> dict[int, np.ndarray]:
    """The filter weights used in each calendar year, each fitted on earlier years only."""
    z = z_returns(prices)
    n, m = z.shape
    years = np.array([d.year for d in prices[DATE].to_list()])
    # Target: z[t+2] + ... + z[t+1+horizon], NaN where any of those days is missing or past the data.
    pad = lambda k: np.vstack([z[k:], np.full((k, m), np.nan)])  # noqa: E731  row t holds z[t+k]
    y = sum(pad(k) for k in range(2, 2 + horizon))
    # Rows whose last `lags` values are all real numbers.
    bad = np.cumsum(~np.isfinite(z), axis=0)
    bad_before = np.vstack([np.zeros((lags, m)), bad[:-lags]])
    have = (bad - bad_before == 0) & (np.arange(n) >= lags - 1)[:, None]
    X = _lagged(np.nan_to_num(z), lags)
    usable = have & np.isfinite(y)

    G = smooth_gamma(lags)
    GtG = G.T @ G
    out = {}
    first_year = years[0] + min_years
    for year in sorted(set(years)):
        if year < first_year:
            continue
        start = int(np.argmax(years == year))
        rows = np.arange(0, max(0, start - (horizon + 2)))  # targets known before the year starts
        mask = usable[rows]
        if not mask.any():
            continue
        Xr = X[rows][mask]  # (samples, lags)
        yr = y[rows][mask]
        XtX = Xr.T @ Xr
        n_years = max(1, len(set(years[rows])))
        lam = alpha * np.trace(XtX) / lags / n_years
        out[int(year)] = np.linalg.solve(XtX + lam * GtG, Xr.T @ yr)
    return out


def forecast(prices: pl.DataFrame, *extras: pl.DataFrame, lags: int = L, horizon: int = H,
             alpha: float = ALPHA, min_years: int = MIN_YEARS) -> pl.DataFrame:
    """Walk-forward Tikhonov forecast, one column per asset. Null until the first refit."""
    z = np.nan_to_num(z_returns(prices))
    years = np.array([d.year for d in prices[DATE].to_list()])
    X = _lagged(z, lags)
    out = np.full(z.shape, np.nan)
    for year, w in weights_by_year(prices, lags, horizon, alpha, min_years).items():
        rows = years == year
        out[rows] = X[rows] @ w
    return prices.select(DATE).with_columns(
        pl.Series(a, out[:, i], nan_to_null=True) for i, a in enumerate(_assets(prices))
    )


def forecast_garch(prices: pl.DataFrame, *extras: pl.DataFrame, power: float = 1.0, **kwargs) -> pl.DataFrame:
    """The Tikhonov forecast times the GARCH vol-timing multiplier (risk management).

    The multiplier is ``clip((long-run vol / GARCH vol) ** power, 0.5, 2)`` per asset and day,
    from GARCH(1,1) fits refitted every 63 days on past data only.
    """
    fc = forecast(prices, **kwargs)
    timing = garch.vol_timing(prices, power=power)
    return fc.select(DATE).with_columns((fc[a] * timing[a]).alias(a) for a in _assets(prices))
