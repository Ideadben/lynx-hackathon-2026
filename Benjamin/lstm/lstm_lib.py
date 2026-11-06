"""LSTM trend strategy: features, purged cross-validation, training and position sizing.

Timing convention, identical to ``evaluation.py``: a position dated T uses data through T, is
traded on T + 1 and earns the return of T + 2. Every array here is indexed (day, instrument).
"""

import itertools
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import calc_key_figures

DATE_COL = "AsOfDate"
VOL_WINDOW = 100
MOMENTUM_WINDOWS = (5, 21, 63, 126)
SHORT_VOL_WINDOW = 20
# Number of daily returns that feed the features of one day. It sets the size of the embargo.
FEATURE_LOOKBACK = max(*MOMENTUM_WINDOWS, VOL_WINDOW, SHORT_VOL_WINDOW)
FIRST_FEATURE_DAY = FEATURE_LOOKBACK  # Returns start at day 1, so the longest window is full here.
FEATURE_CLIP = 5.0
LABEL_LAG = 2
COST_PER_UNIT_RISK = 0.02  # Matches costs=-0.02 in evaluation.py once positions are in risk units.

DEV_FRACTION = 0.75
N_GROUPS = 6
N_TEST_GROUPS = 2  # Set to 1 for ordinary purged k-fold.
MAX_SEQ_LEN = 126
WF_FIRST_TRAIN_DAY = 1300  # About five years of history before the first model.
WF_RETRAIN_EVERY = 260


# ----------------------------------------------------------------------------------------------
# Data and features
# ----------------------------------------------------------------------------------------------
def load_prices() -> pl.DataFrame:
    return pl.read_csv(ROOT / "example_prices.csv", try_parse_dates=True).rename({"dates": DATE_COL})


def price_matrix(prices: pl.DataFrame) -> np.ndarray:
    matrix = prices.drop(DATE_COL).to_numpy().astype(np.float64)
    if np.isnan(matrix).any():
        raise ValueError("Missing prices are not handled yet; decide on a fill rule first.")
    return matrix


def daily_returns(prices: np.ndarray) -> np.ndarray:
    returns = np.full_like(prices, np.nan)
    returns[1:] = np.diff(prices, axis=0)
    return returns


def _rolling(values: np.ndarray, window: int, func) -> np.ndarray:
    """Rolling ``func`` over days. Each window is computed directly, so a day depends only on it."""
    out = np.full_like(values, np.nan)
    if len(values) >= window:
        out[window - 1 :] = func(sliding_window_view(values, window, axis=0), axis=-1)
    return out


def rolling_vol(returns: np.ndarray, window: int = VOL_WINDOW) -> np.ndarray:
    """sigma_T = sqrt(mean of r^2 over the last ``window`` days), as in trend_follower.ipynb."""
    return np.sqrt(_rolling(returns**2, window, np.mean))


def build_features(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return features (day, instrument, feature), labels (day, instrument) and sigma.

    All features are scaled by volatility, so instruments are comparable and one network can be
    shared between them. Labels are the return earned by a position taken today, r_{T+2} / sigma_T.
    """
    sigma = rolling_vol(returns)
    columns = [returns / sigma]
    for window in MOMENTUM_WINDOWS:
        columns.append(_rolling(returns, window, np.sum) / (sigma * np.sqrt(window)))
    columns.append(np.log(rolling_vol(returns, SHORT_VOL_WINDOW) / sigma))
    features = np.clip(np.stack(columns, axis=-1), -FEATURE_CLIP, FEATURE_CLIP)

    labels = np.full_like(returns, np.nan)
    labels[:-LABEL_LAG] = returns[LABEL_LAG:] / sigma[:-LABEL_LAG]
    return features.astype(np.float32), labels.astype(np.float32), sigma


# ----------------------------------------------------------------------------------------------
# Purged cross-validation
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    groups: tuple[int, ...]  # Which groups are tested.
    test: tuple[tuple[int, int], ...]  # Inclusive position-day range of each test block.
    train_ends: tuple[tuple[int, int], ...]  # Inclusive ranges of allowed window end days.


def first_window_end(seq_len: int) -> int:
    return FIRST_FEATURE_DAY + seq_len - 1


def make_groups(start: int, last_position_day: int, n_groups: int = N_GROUPS) -> list[tuple[int, int]]:
    edges = np.linspace(start, last_position_day + 1, n_groups + 1).astype(int)
    return [(int(lo), int(hi) - 1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]


def allowed_ends(start: int, last: int, seq_len: int, blocks) -> tuple[tuple[int, int], ...]:
    """Window end days that may be used for training when ``blocks`` are the test blocks.

    A training sample is a window of ``seq_len`` days ending on t, with labels up to r_{t+2}.
    For each test block [a, b]:

    * Purge, before the block: keep t only if t + 2 <= a, so every label had been realized on the
      first test day. This is exactly what live trading would have known.
    * Embargo, after the block: the inputs of a later window reach back seq_len + FEATURE_LOOKBACK
      days and can hold test labels. Keep t only if its first input return is after r_{b+2}.
    """
    keep = np.zeros(last + 1, dtype=bool)
    keep[first_window_end(seq_len) :] = True
    for a, b in blocks:
        keep[max(a - LABEL_LAG + 1, 0) : b + seq_len + FEATURE_LOOKBACK + 1] = False
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(int), [0]])))
    return tuple((int(lo), int(hi) - 1) for lo, hi in zip(edges[::2], edges[1::2], strict=True))


def make_splits(start, last, seq_len, n_groups=N_GROUPS, n_test=N_TEST_GROUPS) -> list[Split]:
    """Every choice of ``n_test`` test groups. With n_test=1 this is ordinary purged k-fold."""
    groups = make_groups(start, last, n_groups)
    splits = []
    for chosen in itertools.combinations(range(n_groups), n_test):
        blocks = tuple(groups[g] for g in chosen)
        splits.append(Split(chosen, blocks, allowed_ends(start, last, seq_len, blocks)))
    return splits


def make_paths(splits: list[Split], n_groups: int = N_GROUPS) -> list[dict[int, int]]:
    """Assemble complete backtest paths, each mapping every group to one split that tested it."""
    tested_by = {g: [i for i, s in enumerate(splits) if g in s.groups] for g in range(n_groups)}
    n_paths = len(tested_by[0])
    return [{g: tested_by[g][p] for g in range(n_groups)} for p in range(n_paths)]


def check_purging(splits: list[Split], seq_len: int) -> None:
    """Check every training window against the rules, independently of ``allowed_ends``."""
    for split in splits:
        ends = np.concatenate([np.arange(lo, hi + 1) for lo, hi in split.train_ends])
        assert (ends - seq_len + 1 >= FIRST_FEATURE_DAY).all(), "Window starts before features exist"
        first_label, last_label = ends - seq_len + 1 + LABEL_LAG, ends + LABEL_LAG
        first_input = ends - seq_len + 1 - FEATURE_LOOKBACK + 1
        for a, b in split.test:
            overlap = (first_label <= b + LABEL_LAG) & (last_label >= a + LABEL_LAG)
            assert not overlap.any(), f"Label overlap with test block {(a, b)}"
            before, after = ends < a, ends >= a
            assert (last_label[before] <= a).all(), f"Purge violated before {(a, b)}"
            assert (first_input[after] > b + LABEL_LAG).all(), f"Embargo violated after {(a, b)}"


def check_feature_footprint(returns: np.ndarray, day: int = 2000) -> None:
    """A shock to one return must only change features in the FEATURE_LOOKBACK days it can reach."""
    shocked = returns.copy()
    shocked[day] += 10 * np.nanstd(returns, axis=0)
    base, _, _ = build_features(returns)
    new, _, _ = build_features(shocked)
    changed = np.flatnonzero(np.any(base != new, axis=(1, 2)) & ~np.isnan(base).any(axis=(1, 2)))
    assert changed.min() == day and changed.max() <= day + FEATURE_LOOKBACK - 1, (changed.min(), changed.max())


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class TrendLSTM(nn.Module):
    """One LSTM shared by all instruments, with one output per day of the input window."""

    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(self.dropout(output)).squeeze(-1)


def sharpe_loss(raw: torch.Tensor, label: torch.Tensor, burn_in: int, turnover_mult: float) -> torch.Tensor:
    """Minus the Sharpe of net returns in risk units, using the evaluation.py cost per unit of risk."""
    position = torch.tanh(raw)
    trades = (position[:, burn_in:] - position[:, burn_in - 1 : -1]).abs()
    net = position[:, burn_in:] * label[:, burn_in:] - turnover_mult * COST_PER_UNIT_RISK * trades
    return -net.mean() / (net.std() + 1e-6)


def mse_loss(raw: torch.Tensor, label: torch.Tensor, burn_in: int) -> torch.Tensor:
    return ((raw[:, burn_in:] - label[:, burn_in:].clamp(-FEATURE_CLIP, FEATURE_CLIP)) ** 2).mean()


def range_days(ranges) -> np.ndarray:
    return np.concatenate([np.arange(low, high + 1) for low, high in ranges])


def _windows(values: np.ndarray, ends: np.ndarray, instruments: np.ndarray, seq_len: int) -> torch.Tensor:
    index = ends[:, None] + np.arange(-seq_len + 1, 1)
    return torch.from_numpy(values[index, instruments[:, None]].copy())


def fit_lstm(features, labels, ranges, config: dict, seed: int) -> tuple[TrendLSTM, float]:
    """Train on windows ending in ``ranges``. Returns the model and the scale of its raw output."""
    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    seq_len = config["seq_len"]
    burn_in = max(1, seq_len // 4)
    ends = range_days(ranges)
    n_instruments = features.shape[1]

    model = TrendLSTM(features.shape[2], config["hidden"], config["layers"], config["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # One epoch shows each (day, instrument) about once, since each window covers seq_len days.
    batches_per_epoch = max(5, len(ends) * n_instruments // seq_len // config["batch_size"])
    model.train()
    for _ in range(config["epochs"]):
        for _ in range(batches_per_epoch):
            batch_ends = generator.choice(ends, config["batch_size"])
            batch_instruments = generator.integers(n_instruments, size=config["batch_size"])
            x = _windows(features, batch_ends, batch_instruments, seq_len)
            y = _windows(labels, batch_ends, batch_instruments, seq_len)
            raw = model(x)
            if config["loss"] == "sharpe":
                loss = sharpe_loss(raw, y, burn_in, config["turnover_mult"])
            else:
                loss = mse_loss(raw, y, burn_in)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    scale = 1.0
    if config["loss"] == "mse":
        # Regression output is tiny next to the noise, so scale it before squashing to a position.
        sample = generator.choice(ends, min(2000, len(ends)), replace=False)
        scale = float(np.std(predict_raw(model, features, sample, seq_len))) + 1e-8
    return model, scale


@torch.no_grad()
def predict_raw(model: TrendLSTM, features: np.ndarray, days: np.ndarray, seq_len: int, batch: int = 256) -> np.ndarray:
    """Raw output for the last step of the window ending on each day, for every instrument."""
    model.eval()
    n_instruments = features.shape[1]
    out = np.empty((len(days), n_instruments), dtype=np.float32)
    for lo in range(0, len(days), batch):
        chunk = days[lo : lo + batch]
        ends = np.repeat(chunk, n_instruments)
        instruments = np.tile(np.arange(n_instruments), len(chunk))
        raw = model(_windows(features, ends, instruments, seq_len))[:, -1].numpy()
        out[lo : lo + batch] = raw.reshape(len(chunk), n_instruments)
    return out


def task_seed(seed: int, *key) -> int:
    """A stable seed per fit, so a fit does not depend on how much data follows it."""
    return zlib.crc32(repr((seed, *key)).encode()) % (2**31)


def fit_predict(features, labels, ranges, predict_ranges, config, seed) -> np.ndarray:
    """Fit one model and return positions in [-1, 1] for the days in ``predict_ranges``."""
    model, scale = fit_lstm(features, labels, ranges, config, seed)
    return np.tanh(predict_raw(model, features, range_days(predict_ranges), config["seq_len"]) / scale)


def fit_predict_ridge(features, labels, ranges, predict_ranges, alpha: float) -> np.ndarray:
    """Linear control: ridge regression of the label on today's features, with the same purging."""
    ends = range_days(ranges)
    x = features[ends].reshape(-1, features.shape[2]).astype(np.float64)
    y = np.clip(labels[ends].reshape(-1).astype(np.float64), -FEATURE_CLIP, FEATURE_CLIP)
    x_mean, y_mean = x.mean(axis=0), y.mean()
    x_c, y_c = x - x_mean, y - y_mean
    beta = np.linalg.solve(x_c.T @ x_c + alpha * np.eye(x.shape[1]), x_c.T @ y_c)
    scale = np.std(x_c @ beta) + 1e-12
    days = range_days(predict_ranges)
    return np.tanh(((features[days].astype(np.float64) - x_mean) @ beta + y_mean) / scale)


# ----------------------------------------------------------------------------------------------
# From model output to positions
# ----------------------------------------------------------------------------------------------
def smooth(signal: np.ndarray, halflife: float) -> np.ndarray:
    """Causal exponential smoothing of positions, used to cut turnover."""
    if not halflife:
        return signal
    alpha = 1 - 0.5 ** (1 / halflife)
    out = signal.copy()
    for t in range(1, len(out)):
        if not np.isnan(out[t - 1]).any() and not np.isnan(signal[t]).any():
            out[t] = alpha * signal[t] + (1 - alpha) * out[t - 1]
    return out


def size_positions(signal: np.ndarray, returns: np.ndarray, sigma: np.ndarray, window: int = VOL_WINDOW) -> np.ndarray:
    """Inverse-volatility and portfolio risk scaling, exactly as in trend_follower.ipynb."""
    equal_risk = signal / sigma
    portfolio_return = np.full(len(signal), np.nan)
    portfolio_return[2:] = (equal_risk[:-2] * returns[2:]).sum(axis=1)
    portfolio_risk = _rolling(portfolio_return, window, lambda w, axis: np.std(w, axis=axis, ddof=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return equal_risk / portfolio_risk[:, None]


def to_frame(positions: np.ndarray, prices: pl.DataFrame) -> pl.DataFrame:
    instruments = [c for c in prices.columns if c != DATE_COL]
    frame = pl.DataFrame(positions, schema=instruments).fill_nan(None)
    return frame.with_columns(prices[DATE_COL]).select(DATE_COL, *instruments)


def trend_signal(prices: np.ndarray, M: int) -> np.ndarray:
    """Signal of Benjamin/trend_follower.ipynb: sign(P_T - P_{T-M})."""
    signal = np.full_like(prices, np.nan)
    signal[M:] = np.sign(prices[M:] - prices[:-M])
    return signal


def ewma_signal(prices: pl.DataFrame, M: int) -> np.ndarray:
    """Signal of Benjamin/ewma.ipynb: sign(EWMA(M) - EWMA(4M))."""
    instruments = [c for c in prices.columns if c != DATE_COL]
    fast = prices.select(pl.col(instruments).ewm_mean(span=M, adjust=False, min_samples=M))
    slow = prices.select(pl.col(instruments).ewm_mean(span=4 * M, adjust=False, min_samples=4 * M))
    return (fast - slow).select(pl.all().sign()).to_numpy().astype(np.float64)


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def dev_last_position_day(n_days: int) -> int:
    """The last position whose label still falls inside the development period."""
    return int(n_days * DEV_FRACTION) - 1 - LABEL_LAG


def holdout_first_position_day(n_days: int) -> int:
    return dev_last_position_day(n_days) + 1


def slice_figures(positions: np.ndarray, prices: pl.DataFrame, first: int, last: int, **kwargs) -> dict:
    """Key figures from evaluation.py on positions dated first through last."""
    return calc_key_figures(to_frame(positions, prices)[first : last + 1], prices, **kwargs)


def score_paths(split_signals, splits, paths, returns, sigma, prices, halflife) -> dict:
    """Score each complete out-of-sample path with the full sizing pipeline. Sharpe is net of costs.

    ``split_signals[i]`` holds the positions of split i, concatenated over its test blocks.
    """
    groups = sorted({block for split in splits for block in split.test})
    figures = []
    for path in paths:
        signal = np.full(returns.shape, np.nan)
        for g, split_id in path.items():
            split = splits[split_id]
            offset = 0
            for block in split.test:
                size = block[1] - block[0] + 1
                if block == groups[g]:
                    signal[block[0] : block[1] + 1] = split_signals[split_id][offset : offset + size]
                offset += size
        positions = size_positions(smooth(signal, halflife), returns, sigma)
        figures.append(slice_figures(positions, prices, groups[0][0], groups[-1][1]))

    def values(key):
        return np.array([float(f[key]) if f[key] is not None else np.nan for f in figures])

    sharpes = np.nan_to_num(values("sharpe"), nan=-1.0)
    return {
        "objective": float(sharpes.mean()),
        "path_sharpe_min": float(sharpes.min()),
        "path_sharpe_max": float(sharpes.max()),
        "path_sharpes": sharpes.tolist(),
        "adjusted_sharpe": float(np.nanmean(values("adjusted_sharpe"))),
        "av_holding": float(np.nanmean(values("av_holding"))),
        "position_bias": float(np.nanmean(values("position_bias"))),
    }


def block_bootstrap_sharpe(returns_a: np.ndarray, returns_b: np.ndarray | None = None, block=20, n=5000, seed=1):
    """Circular block bootstrap of the annualized Sharpe of a, or of the difference a - b."""
    rng = np.random.default_rng(seed)
    size = len(returns_a)
    starts = rng.integers(size, size=(n, size // block + 1))
    index = ((starts[..., None] + np.arange(block)) % size).reshape(n, -1)[:, :size]

    def sharpe(values):
        sample = values[index]
        return sample.mean(axis=1) / sample.std(axis=1, ddof=1) * np.sqrt(260)

    draws = sharpe(returns_a) if returns_b is None else sharpe(returns_a) - sharpe(returns_b)
    return {"low": float(np.quantile(draws, 0.025)), "high": float(np.quantile(draws, 0.975))}
