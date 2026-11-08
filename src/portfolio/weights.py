"""Choosing functions that look at how the strategies move together.

They have the same form as those in ``backtest.choose``, so they work with ``Portfolio`` and with
``Backtest.walk_forward``. Every one returns weights of at least 0 that add up to 1.

Estimates from a few years of daily returns are noisy, the means far more than the covariances.
``shrink`` pulls the correlations towards 0, and ``mean_shrink`` pulls the strategies' average
returns towards each other, which both make the weights steadier.
"""

import numpy as np
import polars as pl
from scipy.optimize import minimize

from backtest.choose import Choose, Weights, stateless


def inverse_vol() -> Choose:
    """Weights in inverse proportion to each strategy's volatility in training.

    Each strategy then adds about the same risk if they don't move together. Ignores
    correlations and average returns.
    """

    def choose(train: pl.DataFrame) -> Weights:
        weights = 1 / train.to_numpy().std(axis=0, ddof=1)
        return _named(train, weights)

    return stateless(choose)


def risk_parity(shrink: float = 0.0) -> Choose:
    """Weights where every strategy adds the same share of the blend's risk.

    Strategies that move together share their risk, so each gets less weight. Ignores average
    returns. Without correlations (``shrink=1``), this is the same as ``inverse_vol``.

    Args:
        shrink: How far to pull correlations towards 0, from 0 (as measured) to 1 (none).

    Returns:
        A choosing function.
    """
    _check_share(shrink, "shrink")

    def choose(train: pl.DataFrame) -> Weights:
        return _named(train, _risk_parity(_covariance(train.to_numpy(), shrink)))

    return stateless(choose)


def shrunk_max_sharpe(shrink: float = 0.5, mean_shrink: float = 0.5) -> Choose:
    """The weights with the best Sharpe ratio in training, from steadier estimates.

    With ``shrink=0`` and ``mean_shrink=0``, this is ``backtest.choose.max_sharpe()``. With
    ``mean_shrink=1``, every strategy is assumed to earn the same, so only risk and correlation
    decide, and the weights favour strategies that don't move with the rest.

    Args:
        shrink: How far to pull correlations towards 0, from 0 to 1.
        mean_shrink: How far to pull each strategy's average return towards the average of all
            strategies, from 0 to 1.

    Returns:
        A choosing function.
    """
    _check_share(shrink, "shrink")
    _check_share(mean_shrink, "mean_shrink")

    def choose(train: pl.DataFrame) -> Weights:
        returns = train.to_numpy()
        means = returns.mean(axis=0)
        means = (1 - mean_shrink) * means + mean_shrink * means.mean()
        covariance = _covariance(returns, shrink)
        return _named(train, _max_sharpe(means, covariance))

    return stateless(choose)


def _covariance(returns: np.ndarray, shrink: float) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(returns, rowvar=False))
    return (1 - shrink) * covariance + shrink * np.diag(np.diag(covariance))


def _risk_parity(covariance: np.ndarray) -> np.ndarray:
    """Equal risk contributions, from the convex form of Spinu (2013)."""
    count = len(covariance)
    scale = np.sqrt(np.diag(covariance))
    scale = np.where(scale > 0, scale, 1.0)
    # Solve in units of each strategy's volatility, so the optimiser sees numbers near 1.
    correlation = covariance / np.outer(scale, scale)

    def loss(x: np.ndarray) -> float:
        return 0.5 * x @ correlation @ x - np.log(x).sum() / count

    def gradient(x: np.ndarray) -> np.ndarray:
        return correlation @ x - 1 / (count * x)

    start = np.full(count, 1 / np.sqrt(count))
    result = minimize(
        loss, start, jac=gradient, method="L-BFGS-B", bounds=[(1e-9, None)] * count
    )
    weights = result.x / scale
    return weights / weights.sum()


def _max_sharpe(means: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    count = len(means)

    def loss(weights: np.ndarray) -> float:
        risk = np.sqrt(weights @ covariance @ weights)
        return -(means @ weights) / risk if risk > 0 else 0.0

    result = minimize(
        loss,
        np.full(count, 1 / count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
    )
    weights = np.clip(result.x, 0, None)
    return weights / weights.sum()


def _named(train: pl.DataFrame, weights: np.ndarray) -> Weights:
    weights = weights / weights.sum()
    return dict(zip(train.columns, weights.tolist(), strict=True))


def _check_share(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, not {value}.")
