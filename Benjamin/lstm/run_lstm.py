"""Run the LSTM experiment. Results go to Benjamin/lstm/results, which lstm.ipynb reads.

    uv run python Benjamin/lstm/run_lstm.py search --trials 96   # Purged CPCV on the dev period only
    uv run python Benjamin/lstm/run_lstm.py walkforward          # Positions from the frozen config
    uv run python Benjamin/lstm/run_lstm.py leakage              # Refit on altered data and compare
    uv run python Benjamin/lstm/run_lstm.py placebo              # Refit on returns with no structure
    uv run python Benjamin/lstm/run_lstm.py seeds                # Refit with single networks
    uv run python Benjamin/lstm/run_lstm.py report               # Scores, including the holdout
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import lstm_lib as lib
import numpy as np
import optuna
import polars as pl
import torch

RESULTS = Path(__file__).resolve().parent / "results"
WF_SEEDS = 5
_data: dict = {}


def _init_worker(features, labels):
    torch.set_num_threads(1)
    _data.update(features=features, labels=labels)


def _fit_task(args):
    ranges, predict_ranges, config, seed = args
    return lib.fit_predict(_data["features"], _data["labels"], ranges, predict_ranges, config, seed)


def make_pool(features, labels, workers):
    return ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(features, labels))


def suggest(trial: optuna.Trial) -> dict:
    return {
        "loss": trial.suggest_categorical("loss", ["sharpe", "mse"]),
        "seq_len": trial.suggest_categorical("seq_len", [21, 42, 63, lib.MAX_SEQ_LEN]),
        "hidden": trial.suggest_categorical("hidden", [8, 16, 32, 64]),
        "layers": trial.suggest_categorical("layers", [1, 2]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "epochs": trial.suggest_int("epochs", 3, 30),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "turnover_mult": trial.suggest_categorical("turnover_mult", [1.0, 2.0, 5.0, 10.0]),
        "halflife": trial.suggest_categorical("halflife", [0, 2, 5, 10, 20]),
    }


def search(args):
    prices = lib.load_prices()
    matrix = lib.price_matrix(prices)
    returns = lib.daily_returns(matrix)
    features, labels, sigma = lib.build_features(returns)
    lib.check_feature_footprint(returns)

    # Everything after this day stays out of the search. Labels past it are blanked to enforce that.
    last = lib.dev_last_position_day(len(matrix))
    labels = labels.copy()
    labels[last + 1 :] = np.nan
    start = lib.first_window_end(lib.MAX_SEQ_LEN)
    dates = prices[lib.DATE_COL]
    print(f"Dev position days {start}..{last}, {dates[start]} to {dates[last]}")

    for seq_len in (21, 42, 63, lib.MAX_SEQ_LEN):
        lib.check_purging(lib.make_splits(start, last, seq_len), seq_len)
    print("Purging and embargo checks passed", flush=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, constant_liar=True, n_startup_trials=24)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    rows = []
    started = time.time()
    RESULTS.mkdir(exist_ok=True)
    with make_pool(features, labels, args.workers) as pool:
        done = 0
        while done < args.trials:
            batch = [study.ask() for _ in range(min(args.batch, args.trials - done))]
            configs = [suggest(trial) for trial in batch]
            all_splits = [lib.make_splits(start, last, c["seq_len"]) for c in configs]
            tasks = [
                (split.train_ends, split.test, config, lib.task_seed(0, "cv", split_id))
                for config, splits in zip(configs, all_splits, strict=True)
                for split_id, split in enumerate(splits)
            ]
            outputs = iter(pool.map(_fit_task, tasks, chunksize=1))
            for trial, config, splits in zip(batch, configs, all_splits, strict=True):
                signals = [next(outputs) for _ in splits]
                score = lib.score_paths(signals, splits, lib.make_paths(splits), returns, sigma, prices, config["halflife"])
                study.tell(trial, score["objective"])
                rows.append({"trial": trial.number, **config, **score, "path_sharpes": json.dumps(score["path_sharpes"])})
                print(
                    f"#{trial.number:3d} cv={score['objective']:+.3f} [{score['path_sharpe_min']:+.2f},"
                    f"{score['path_sharpe_max']:+.2f}] hold={score['av_holding']:5.1f} bias={score['position_bias']:+.2f} "
                    f"{config['loss']:6s} L={config['seq_len']:3d} h={config['hidden']:2d}x{config['layers']} "
                    f"ep={config['epochs']:2d} bs={config['batch_size']:3d} [{time.time() - started:5.0f}s]",
                    flush=True,
                )
            done += len(batch)
            pl.DataFrame(rows).sort("objective", descending=True).write_csv(RESULTS / "lstm_trials.csv")

    best = max(rows, key=lambda row: row["objective"])
    config = {key: best[key] for key in configs[0]}
    (RESULTS / "best_config.json").write_text(json.dumps(config, indent=2))
    print("Best:", best["objective"], config)
    ridge_search(prices, returns, features, labels, sigma, start, last)


def ridge_search(prices, returns, features, labels, sigma, start, last):
    splits = lib.make_splits(start, last, seq_len=1)
    paths = lib.make_paths(splits)
    rows = []
    for alpha in (1e0, 1e2, 1e4, 1e6):
        signals = [lib.fit_predict_ridge(features, labels, s.train_ends, s.test, alpha) for s in splits]
        for halflife in (0, 2, 5, 10, 20):
            score = lib.score_paths(signals, splits, paths, returns, sigma, prices, halflife)
            rows.append({"alpha": alpha, "halflife": halflife, **score, "path_sharpes": json.dumps(score["path_sharpes"])})
    table = pl.DataFrame(rows).sort("objective", descending=True)
    table.write_csv(RESULTS / "ridge_trials.csv")
    best = table.row(0, named=True)
    (RESULTS / "best_ridge.json").write_text(json.dumps({"alpha": best["alpha"], "halflife": best["halflife"]}, indent=2))
    print("Best ridge:", best["objective"], best["alpha"], best["halflife"])


def retrain_days(n_days):
    return list(range(lib.WF_FIRST_TRAIN_DAY, n_days, lib.WF_RETRAIN_EVERY))


def walk_forward_signal(prices_frame, config, workers, seeds=range(WF_SEEDS)) -> np.ndarray:
    """Expanding-window retraining. A model fitted on day D only sees labels realized by day D."""
    matrix = lib.price_matrix(prices_frame)
    features, labels, _ = lib.build_features(lib.daily_returns(matrix))
    n_days = len(matrix)
    tasks, spans = [], []
    for k, day in enumerate(retrain_days(n_days)):
        ranges = ((lib.first_window_end(config["seq_len"]), day - lib.LABEL_LAG),)
        span = (day, min(day + lib.WF_RETRAIN_EVERY, n_days) - 1)
        spans.append(span)
        tasks.extend((ranges, (span,), config, lib.task_seed(seed, "wf", k)) for seed in seeds)
    signal = np.full(matrix.shape, np.nan)
    with make_pool(features, labels, workers) as pool:
        outputs = iter(pool.map(_fit_task, tasks, chunksize=1))
        for span in spans:
            signal[span[0] : span[1] + 1] = np.mean([next(outputs) for _ in seeds], axis=0)
    return signal


def lstm_positions(prices_frame, config, workers, seeds=range(WF_SEEDS)) -> np.ndarray:
    returns = lib.daily_returns(lib.price_matrix(prices_frame))
    _, _, sigma = lib.build_features(returns)
    signal = walk_forward_signal(prices_frame, config, workers, seeds)
    return lib.size_positions(lib.smooth(signal, config["halflife"]), returns, sigma)


def ridge_positions(prices_frame, config) -> np.ndarray:
    matrix = lib.price_matrix(prices_frame)
    returns = lib.daily_returns(matrix)
    features, labels, sigma = lib.build_features(returns)
    signal = np.full(matrix.shape, np.nan)
    n_days = len(matrix)
    for day in retrain_days(n_days):
        span = (day, min(day + lib.WF_RETRAIN_EVERY, n_days) - 1)
        ranges = ((lib.first_window_end(1), day - lib.LABEL_LAG),)
        signal[span[0] : span[1] + 1] = lib.fit_predict_ridge(features, labels, ranges, (span,), config["alpha"])
    return lib.size_positions(lib.smooth(signal, config["halflife"]), returns, sigma)


def walkforward(args):
    config = json.loads((RESULTS / "best_config.json").read_text())
    prices = lib.load_prices()
    started = time.time()
    positions = lstm_positions(prices, config, args.workers)
    lib.to_frame(positions, prices).write_csv(RESULTS / "positions_lstm.csv")
    print(f"Walk-forward finished in {time.time() - started:.0f}s")


def leakage(args):
    """Apply both tests from the instructions to the real pipeline, refitting every model."""
    config = json.loads((RESULTS / "best_config.json").read_text())
    prices = lib.load_prices()
    instruments = [c for c in prices.columns if c != lib.DATE_COL]
    full = lstm_positions(prices, config, args.workers)
    truncated = lstm_positions(prices.head(prices.height - 20), config, args.workers)

    cut = lib.holdout_first_position_day(prices.height) + 300
    is_future = pl.int_range(pl.len()) > cut
    altered_prices = prices.with_columns(
        pl.when(is_future).then(pl.col(c) * 1.5 + 100).otherwise(pl.col(c)).alias(c) for c in instruments
    )
    altered = lstm_positions(altered_prices, config, args.workers)

    typical = float(np.nanmedian(np.abs(full)))

    def compare(a, b):
        return {
            "same_missing": bool((np.isnan(a) == np.isnan(b)).all()),
            "max_diff_relative_to_typical_position": float(np.nanmax(np.abs(a - b)) / typical),
        }

    result = {
        "shorter_history": compare(full[: len(truncated)], truncated),
        "altered_future": compare(full[: cut + 1], altered[: cut + 1]),
        # The altered test only means something if the positions after the cut really moved.
        "altered_future_after_cut": compare(full[cut + 1 :], altered[cut + 1 :]),
    }
    (RESULTS / "leakage.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def placebo(args):
    """Shuffle days to remove all return structure. Sharpe should then be near zero."""
    config = json.loads((RESULTS / "best_config.json").read_text())
    prices = lib.load_prices()
    matrix = lib.price_matrix(prices)
    returns = lib.daily_returns(matrix)
    rows = []
    for seed in range(args.placebo_runs):
        rng = np.random.default_rng(100 + seed)
        shuffled = returns[1:][rng.permutation(len(returns) - 1)]  # Same permutation for all assets.
        new = np.vstack([matrix[0], matrix[0] + np.cumsum(shuffled, axis=0)])
        frame = pl.DataFrame(new, schema=prices.drop(lib.DATE_COL).columns).with_columns(prices[lib.DATE_COL])
        positions = lstm_positions(frame, config, args.workers)
        figures = lib.slice_figures(positions, frame, lib.WF_FIRST_TRAIN_DAY, len(matrix) - 1)
        rows.append({"seed": seed, "sharpe": float(figures["sharpe"]), "adjusted_sharpe": float(figures["adjusted_sharpe"])})
        print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(RESULTS / "placebo.csv")


def seeds(args):
    """Refit the frozen config with one network per run, to measure how much the seed matters."""
    config = json.loads((RESULTS / "best_config.json").read_text())
    prices = lib.load_prices()
    first_holdout = lib.holdout_first_position_day(prices.height)
    rows = []
    for seed in range(args.seed_runs):
        positions = lstm_positions(prices, config, args.workers, seeds=[seed])
        dev = lib.slice_figures(positions, prices, lib.WF_FIRST_TRAIN_DAY, first_holdout - 1)
        holdout = lib.slice_figures(positions, prices, first_holdout, prices.height - 1)
        rows.append({"seed": seed, "dev_sharpe": float(dev["sharpe"]), "holdout_sharpe": float(holdout["sharpe"])})
        print(rows[-1], flush=True)
    pl.DataFrame(rows).write_csv(RESULTS / "seed_stability.csv")


def report(args):
    prices = lib.load_prices()
    matrix = lib.price_matrix(prices)
    returns = lib.daily_returns(matrix)
    _, _, sigma = lib.build_features(returns)
    n_days = len(matrix)
    instruments = [c for c in prices.columns if c != lib.DATE_COL]
    lstm = pl.read_csv(RESULTS / "positions_lstm.csv", try_parse_dates=True).select(instruments).to_numpy().astype(float)
    ridge = ridge_positions(prices, json.loads((RESULTS / "best_ridge.json").read_text()))
    lib.to_frame(ridge, prices).write_csv(RESULTS / "positions_ridge.csv")
    strategies = {
        "LSTM": lstm,
        "Ridge on the same features": ridge,
        "Trend follower M=50": lib.size_positions(lib.trend_signal(matrix, 50), returns, sigma),
        "EWMA M=16": lib.size_positions(lib.ewma_signal(prices, 16), returns, sigma),
    }
    first_holdout = lib.holdout_first_position_day(n_days)
    periods = {"dev": (lib.WF_FIRST_TRAIN_DAY, first_holdout - 1), "holdout": (first_holdout, n_days - 1)}
    names = ["sharpe", "adjusted_sharpe", "sharpe_ex_slippage", "sharpe_1d_lag", "av_holding", "position_bias", "skewness", "kurtosis"]
    rows, daily = [], {}
    for name, positions in strategies.items():
        for period, (first, last) in periods.items():
            figures = lib.slice_figures(positions, prices, first, last, key_figures=[*names, "returns"])
            rows.append({"strategy": name, "period": period, **{k: float(figures[k]) for k in names}})
            if period == "holdout":
                daily[name] = figures["returns"].drop_nulls()["returns"].to_numpy()
    table = pl.DataFrame(rows)
    table.write_csv(RESULTS / "summary.csv")

    n = min(len(values) for values in daily.values())
    daily = {name: values[-n:] for name, values in daily.items()}
    bootstrap = {
        "sharpe": {name: lib.block_bootstrap_sharpe(values) for name, values in daily.items()},
        "lstm_minus": {
            name: lib.block_bootstrap_sharpe(daily["LSTM"], values) for name, values in daily.items() if name != "LSTM"
        },
    }
    (RESULTS / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2))
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, tbl_rows=-1, float_precision=2):
        print(table)
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["search", "walkforward", "leakage", "placebo", "seeds", "report"])
    parser.add_argument("--trials", type=int, default=96)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--placebo-runs", type=int, default=5)
    parser.add_argument("--seed-runs", type=int, default=10)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    globals()[args.command](args)
