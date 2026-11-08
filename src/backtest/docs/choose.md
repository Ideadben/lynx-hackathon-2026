# Module backtest.choose

```python
from backtest.choose import best_single, drop_small, equal_weights, max_sharpe, random_search, stateless
```

Choosing functions for `Backtest.walk_forward`.

---

```python
Weights = dict[str, float]
Choose = Callable[[pl.DataFrame, State], tuple[Weights, State]]
```

A choosing function for `walk_forward`. `choose(train, state)` takes each rule's daily net returns in training, one
column per rule, and returns a weight per rule and the state for the next block. `state` is `None` at first.

```python
def best_single(objective: Objective = adjusted_sharpe) -> Choose
```

`best_single()` puts all weight on the rule with the best `objective` in training.

```python
def equal_weights() -> Choose
```

`equal_weights()` puts the same weight on every rule.

```python
def max_sharpe(long_only: bool = True) -> Choose
```

`max_sharpe()` returns the weights with the best Sharpe ratio in training.

```python
def random_search(
    candidates: int = 500,
    keep: int = 50,
    validation: float = 0.3,
    objective: Objective = adjusted_sharpe,
    seed: int = 0,
) -> Choose
```

`random_search()` tries random weights, keeps the best `keep` for the next block, and picks one on validation days.

```python
def drop_small(choose: Choose, below: float = 0.05) -> Choose
```

`drop_small(c)` sets weights from `c` below `below` to 0, and scales the rest up to the same total.

```python
def stateless(choose: Callable[[pl.DataFrame], Weights]) -> Choose
```

`stateless(f)` turns a function from training returns to weights into a choosing function.
