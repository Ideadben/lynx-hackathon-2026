"""Choosing functions for ``Backtest.walk_forward``.

A choosing function decides how much weight each rule gets in the next block. It sees each rule's
daily net returns over the training days, as a frame with one column per rule, and returns a
weight per rule. State is passed in and out explicitly, so a choice can depend on earlier ones:

    def choose(train: pl.DataFrame, state: State) -> tuple[Weights, State]: ...

``state`` is ``None`` before the first block. Use ``stateless`` for a function that needs no state.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

from backtest.stats import adjusted_sharpe, sharpe

Weights = dict[str, float]
State = Any
Choose = Callable[[pl.DataFrame, State], tuple[Weights, State]]
Objective = Callable[[np.ndarray], float | np.ndarray]


def stateless(choose: Callable[[pl.DataFrame], Weights]) -> Choose:
    """Turns ``train -> weights`` into a choosing function."""

    def stateless_choose(train: pl.DataFrame, state: State) -> tuple[Weights, State]:
        return choose(train), state

    return stateless_choose


def best_single(objective: Objective = adjusted_sharpe) -> Choose:
    """All weight on the rule that scored best in training."""

    def choose(train: pl.DataFrame) -> Weights:
        scores = _scores(objective, train.to_numpy())
        best = int(np.nanargmax(np.where(np.isnan(scores), -np.inf, scores)))
        return {name: float(i == best) for i, name in enumerate(train.columns)}

    return stateless(choose)


def equal_weights() -> Choose:
    """The same weight on every rule. Learns nothing, which makes it a good baseline."""

    def choose(train: pl.DataFrame) -> Weights:
        return {name: 1 / train.width for name in train.columns}

    return stateless(choose)


def max_sharpe(long_only: bool = True) -> Choose:
    """The weights with the best Sharpe ratio in training.

    With ``long_only``, weights are at least 0 and add up to 1, found with an optimiser.
    Otherwise they come from the covariance of the rules' returns, and can be negative.
    """

    def choose(train: pl.DataFrame) -> Weights:
        returns = train.to_numpy()
        if long_only:
            weights = _long_only_max_sharpe(returns)
        else:
            raw = np.linalg.pinv(np.cov(returns, rowvar=False)) @ returns.mean(axis=0)
            weights = raw / np.abs(raw).sum()
        return dict(zip(train.columns, weights.tolist(), strict=True))

    return stateless(choose)


def random_search(
    candidates: int = 500,
    keep: int = 50,
    validation: float = 0.3,
    objective: Objective = adjusted_sharpe,
    seed: int = 0,
) -> Choose:
    """Random long-only weights, refined from block to block.

    Before each block:

    1. Draws ``candidates`` new random weightings, and adds the ``keep`` kept from the last block.
    2. Splits the training days: the last ``validation`` share is for validation.
    3. Keeps the ``keep`` weightings that score best on the first part.
    4. Of those, picks the one that scores best on the validation part.
    """

    def choose(train: pl.DataFrame, state: State) -> tuple[Weights, State]:
        returns = train.to_numpy()
        kept, rng = state if state is not None else (None, np.random.default_rng(seed))
        fresh = rng.dirichlet(np.ones(train.width), candidates)
        pool = fresh if kept is None else np.vstack([kept, fresh])

        split = int(len(returns) * (1 - validation))
        fit = _scores(objective, returns[:split] @ pool.T)
        shortlist = pool[np.argsort(-np.nan_to_num(fit, nan=-np.inf))[:keep]]
        checked = _scores(objective, returns[split:] @ shortlist.T)
        best = shortlist[int(np.argmax(np.nan_to_num(checked, nan=-np.inf)))]
        return dict(zip(train.columns, best.tolist(), strict=True)), (shortlist, rng)

    return choose


def drop_small(choose: Choose, below: float = 0.05) -> Choose:
    """Sets weights smaller than ``below`` to 0, and scales the rest up to the same total."""

    def dropping(train: pl.DataFrame, state: State) -> tuple[Weights, State]:
        weights, state = choose(train, state)
        total = sum(abs(w) for w in weights.values())
        kept = {name: (w if abs(w) >= below else 0.0) for name, w in weights.items()}
        kept_total = sum(abs(w) for w in kept.values())
        if kept_total == 0:
            return weights, state
        return {name: w * total / kept_total for name, w in kept.items()}, state

    return dropping


def _scores(objective: Objective, returns: np.ndarray) -> np.ndarray:
    """The objective of every column. Falls back to one column at a time."""
    scores = np.asarray(objective(returns), dtype=float)
    if scores.shape != (returns.shape[1],):
        scores = np.array([float(objective(column)) for column in returns.T])
    return scores


def _long_only_max_sharpe(returns: np.ndarray) -> np.ndarray:
    count = returns.shape[1]

    def loss(weights: np.ndarray) -> float:
        value = sharpe(returns @ weights)
        return -value if np.isfinite(value) else 0.0

    result = minimize(
        loss,
        np.full(count, 1 / count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
    )
    weights = np.clip(result.x, 0, None)
    return weights / weights.sum()
