"""GARCH(1,1) volatility forecasts and the vol-timing multiplier, used as risk management.

Copied from Søren's ``Søren/garch_lib.py`` (his research file stays as it is), without its
trend-specific ``forecast``, so the strategies package does not depend on a notebook folder.

A GARCH(1,1) model is fitted per asset to daily price changes on an expanding window, refitted
every ``refit_every`` days (63 by default). Between refits the recursion runs with frozen
parameters, so ``sigma[T]`` (the forecast for day T + 1) uses data up to and including T only.
``vol_timing`` turns it into an exposure multiplier ``clip((long-run vol / sigma) ** power, floor, cap)``:
below 1 when GARCH expects more turbulence than usual, above 1 when it expects calm.

Fits are cached on the exact bytes of their input window, so the leakage checker's reruns on
shorter or altered data reuse them.
"""

import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
from arch import arch_model


DATE = "AsOfDate"
REFIT_EVERY = 63
MIN_TRAIN = 500
MAX_PERSISTENCE = 0.999
EWMA_LAMBDA = 0.94

_FIT_CACHE: dict[tuple[bytes, str], tuple[float, float, float, float] | None] = {}


@dataclass
class GarchPath:
    """GARCH output for one asset, aligned with the price rows (NaN where undefined)."""

    sigma: np.ndarray  # forecast vol for the next day, in price-change units
    longrun: np.ndarray  # unconditional vol of the active fit
    omega: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray


def _instruments(frame: pl.DataFrame) -> list[str]:
    return [column for column in frame.columns if column != DATE]


def _fit(window: np.ndarray, dist: str) -> tuple[float, float, float, float] | None:
    """Fits a zero-mean GARCH(1,1). Returns (omega, alpha, beta, last variance) in data units.

    The data is scaled to unit variance for numerical stability and the results scaled back.
    Returns None if the fit fails or is (near) non-stationary.
    """
    key = (hashlib.sha1(window.tobytes()).digest(), dist)
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]

    scale = float(np.std(window))
    result = None
    if scale > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(
                    window / scale, mean="Zero", vol="GARCH", p=1, q=1, dist=dist,
                    rescale=False,
                ).fit(disp="off", show_warning=False)
            omega, alpha, beta = (float(res.params[n]) for n in ("omega", "alpha[1]", "beta[1]"))
            last_var = float(res.conditional_volatility[-1]) ** 2
            if (
                res.convergence_flag == 0
                and np.isfinite([omega, alpha, beta, last_var]).all()
                and omega > 0
                and alpha + beta < MAX_PERSISTENCE
            ):
                result = (omega * scale**2, alpha, beta, last_var * scale**2)
        except (ValueError, np.linalg.LinAlgError):
            result = None
    _FIT_CACHE[key] = result
    return result


def garch_path(
    changes: np.ndarray,
    refit_every: int = REFIT_EVERY,
    min_train: int = MIN_TRAIN,
    dist: str = "t",
) -> GarchPath:
    """Runs the refit-and-recurse GARCH scheme on one asset's daily price changes.

    Args:
        changes: Price changes, one per row; NaN before the asset starts.
        refit_every: Days between refits.
        min_train: Changes needed before the first fit.
        dist: Innovation distribution for ``arch`` ("normal", "t", "skewt", ...).

    Returns:
        The path; row ``t`` only depends on ``changes[: t + 1]``.
    """
    n = len(changes)
    out = {name: np.full(n, np.nan) for name in ("sigma", "longrun", "omega", "alpha", "beta")}
    finite = np.flatnonzero(np.isfinite(changes))
    if len(finite) < min_train:
        return GarchPath(**out)
    start = finite[0]
    x = np.nan_to_num(changes[start:])  # forward-filled prices: no gaps after the start

    params: tuple[float, float, float] | None = None
    var = np.nan  # conditional variance of the current day, h_t
    for k in range(min_train - 1, len(x)):
        if (k - (min_train - 1)) % refit_every == 0:
            window = x[: k + 1]
            fitted = _fit(window, dist)
            if fitted is not None:
                omega, alpha, beta, last_var = fitted
                params = (omega, alpha, beta)
                var = last_var
            elif params is None:
                # No usable fit yet: RiskMetrics EWMA, anchored at the sample variance.
                sample_var = float(np.var(window))
                params = (0.0, 1 - EWMA_LAMBDA, EWMA_LAMBDA)
                var = sample_var
            # else: keep the previous parameters and keep recursing from the current variance.
        omega, alpha, beta = params
        var_next = omega + alpha * x[k] ** 2 + beta * var
        t = start + k
        out["sigma"][t] = np.sqrt(var_next)
        persistence = alpha + beta
        out["longrun"][t] = (
            np.sqrt(omega / (1 - persistence)) if omega > 0 and persistence < 1
            else np.sqrt(np.mean(x[: k + 1] ** 2))
        )
        out["omega"][t], out["alpha"][t], out["beta"][t] = omega, alpha, beta
        var = var_next
    return GarchPath(**out)


def garch_paths(prices: pl.DataFrame, **garch_kw) -> dict[str, GarchPath]:
    """GARCH paths for every asset. Keyword arguments go to ``garch_path``."""
    return {
        asset: garch_path(prices[asset].diff().to_numpy().astype(float), **garch_kw)
        for asset in _instruments(prices)
    }


def garch_vol(prices: pl.DataFrame, **garch_kw) -> pl.DataFrame:
    """One-step-ahead GARCH volatility per asset (null during warm-up)."""
    paths = garch_paths(prices, **garch_kw)
    return prices.select(DATE).with_columns(
        pl.Series(asset, path.sigma, nan_to_null=True) for asset, path in paths.items()
    )


def vol_timing(
    prices: pl.DataFrame,
    power: float = 1.0,
    floor: float = 0.5,
    cap: float = 2.0,
    ref: str = "longrun",
    warmup: float | None = 1.0,
    **garch_kw,
) -> pl.DataFrame:
    """Exposure multiplier ``clip((sigma_ref / sigma) ** power, floor, cap)`` per asset.

    Args:
        prices: ``AsOfDate`` plus one price column per asset.
        power: Strength of the timing; 0 turns it off.
        floor, cap: Bounds on the multiplier.
        ref: "longrun" (unconditional vol of the current fit) or "expanding" (expanding mean of
            the GARCH forecast so far).
        warmup: Multiplier before GARCH is ready; None leaves those days null.
        **garch_kw: Passed to ``garch_path``.
    """
    paths = garch_paths(prices, **garch_kw)
    columns = []
    for asset, path in paths.items():
        if ref == "longrun":
            reference = path.longrun
        elif ref == "expanding":
            valid = np.isfinite(path.sigma)
            counts = np.cumsum(valid)
            sums = np.cumsum(np.where(valid, path.sigma, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                reference = np.where(valid, sums / counts, np.nan)
        else:
            raise ValueError(f"Unknown ref: {ref!r}")
        with np.errstate(invalid="ignore", divide="ignore"):
            multiplier = np.clip((reference / path.sigma) ** power, floor, cap)
        if warmup is not None:
            multiplier = np.where(np.isfinite(multiplier), multiplier, warmup)
        columns.append(pl.Series(asset, multiplier, nan_to_null=True))
    return prices.select(DATE).with_columns(columns)
