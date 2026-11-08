# Module backtest.scoring

```python
from backtest.scoring import Scoring
```

How positions are scored: by the formulas in the PDF, or with Lynx's `evaluation.py`.

---

```python
ScoringKind = Literal["pdf", "lynx"]
```

The PDF's formulas, or Lynx's `evaluation.py`.

```python
class Scoring:
    kind: ScoringKind = "pdf"
    delta: float = 0.0002
    costs: float = -0.02
```

The scoring formula and its cost setting.

- **Raises** `ValueError` if `kind` is unknown.

```python
def figures(self, positions: pl.DataFrame, prices: pl.DataFrame, names: Sequence[str]) -> dict[str, object]
```

`scoring.figures(positions, prices, names)` calculates the named key figures.

```python
def plot(self, positions: pl.DataFrame, prices: pl.DataFrame) -> Figure
```

`scoring.plot(positions, prices)` returns the dashboard.
