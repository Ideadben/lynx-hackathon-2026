"""Spectral look at the example data: are there cycles in returns or in volatility?

Part 1 (diagnosis, dev period only):
  - Periodogram of vol-normalised daily returns per asset. Pure noise has a flat spectrum, so a peak
    means a repeating pattern. Peaks are judged against 1000 shuffles of the same returns, which keep
    the distribution and destroy any timing, so the threshold accounts for testing every frequency.
  - Low-frequency power vs the flat level: >1 means trends (slow cycles), <1 means reversion.
  - The same for |returns|, where cycles would mean predictable calm and busy periods.

Part 2 (strategy): a rolling FFT forecast. Every day, take the last `window` days of returns, keep
the `k` strongest frequencies, and extend those waves two days ahead. Only past data is used.

Run: uv run python YANNICK/fft_scan.py
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from backtest import Backtest
from strategies import trend

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "prices" / "example_prices.csv"
rng = np.random.default_rng(0)


def z_returns(px: pd.DataFrame) -> pd.DataFrame:
    ret = px.diff()
    vol = np.sqrt((ret**2).rolling(100, min_periods=50).mean())
    return ret / vol.shift(1)


def periodogram(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x)) ** 2 / len(x)
    freq = np.fft.rfftfreq(len(x))
    return freq[1:], power[1:]  # drop the zero frequency (the mean)


def scan(series: pd.Series, label: str, shuffles: int = 1000) -> None:
    x = series.dropna().to_numpy()
    freq, power = periodogram(x)
    flat = power.mean()
    # Max-peak null: the largest peak in each shuffle, so the threshold covers every frequency.
    null_max = np.array([periodogram(rng.permutation(x))[1].max() for _ in range(shuffles)])
    threshold = np.quantile(null_max, 0.95)
    top = np.argsort(power)[::-1][:3]
    peaks = ", ".join(f"{1 / freq[i]:.1f}d ({power[i] / flat:.1f}x)" for i in top)
    sig = (power > threshold).sum()
    # Band power relative to flat: periods > 20 days (slow) and 2-5 days (fast).
    period = 1 / freq
    slow = power[period > 20].mean() / flat
    fast = power[(period >= 2) & (period <= 5)].mean() / flat
    print(f"  {label:<10} top peaks: {peaks:<42} significant: {sig:>2}   "
          f"slow>20d {slow:.2f}x   fast 2-5d {fast:.2f}x")


# ---------------------------------------------------------------------------------------------------
def fft_forecast(prices: pl.DataFrame, window: int = 256, k: int = 3, step: int = 5) -> pl.DataFrame:
    """Extend the k strongest waves of the last `window` days of returns two days ahead.

    Recomputed every `step` days and held in between, to keep it fast. Uses only past returns.
    """
    px = prices.to_pandas().set_index("AsOfDate")
    z = z_returns(px).fillna(0.0).to_numpy()
    n, m = z.shape
    out = np.full((n, m), np.nan)
    t_ahead = window + 1  # index of the return two days after the last observed one (window - 1)
    taper = np.hanning(window)
    for t in range(window + 100, n, step):
        seg = z[t - window + 1 : t + 1]
        seg = (seg - seg.mean(axis=0)) * taper[:, None]
        spec = np.fft.rfft(seg, axis=0)
        spec[0] = 0
        keep = np.argsort(np.abs(spec), axis=0)[::-1][:k]
        mask = np.zeros_like(spec, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=0)
        spec = np.where(mask, spec, 0)
        freqs = np.fft.rfftfreq(window)[:, None]
        # Evaluate the kept sinusoids at the future index (the series continues periodically
        # in an FFT, so this is the 'extend the wave' forecast).
        phase = np.exp(2j * np.pi * freqs * t_ahead)
        weight = np.where((freqs == 0) | (freqs == 0.5), 1.0, 2.0)
        value = (weight * spec * phase).real.sum(axis=0) / window
        out[t : t + step] = value
    return pl.from_pandas(pd.DataFrame(out, index=px.index, columns=px.columns).rename_axis("AsOfDate").reset_index())


def main() -> None:
    px = pd.read_csv(PRICES, index_col=0, parse_dates=True)
    dev = px.iloc[: int(len(px) * 0.75)]
    z = z_returns(dev)

    print("Periodogram of vol-normalised returns (dev). 'significant' = peaks above the 95% shuffle max.")
    for a in dev.columns:
        scan(z[a], a)
    print("\nPeriodogram of |returns| (volatility clustering and cycles)")
    for a in dev.columns:
        scan(z[a].abs(), a)
    print("\nSame-day returns across assets: the market-wide part (mean z of all five)")
    scan(z.mean(axis=1), "market")

    print("\nBacktest of the rolling FFT forecast, dev period")
    bt = Backtest(PRICES)
    print(bt.sweep(fft_forecast, window=[128, 256, 512], k=[1, 3, 8]).select(
        ["window", "k", "adjusted_sharpe", "av_holding"]))
    cmp = bt.compare({"trend": trend.forecast, "fft 256 k3": partial(fft_forecast, window=256, k=3)})
    print(cmp.correlations)


if __name__ == "__main__":
    main()
