"""Experiment drivers: sanity checks, dev-set grid search, multi-seed retraining, the dev results table.
Everything here uses train + dev only."""
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.metrics import compute_metrics
from src.train import initial_loss, predict_proba, train_classifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEV_RESULTS_CSV = RESULTS / "dev_results.csv"
METRICS = ["accuracy", "precision", "recall", "f1", "auroc"]

GRID = dict(hidden_dim=[64, 256, 512], lr=[1e-3, 3e-4, 1e-4, 3e-5], dropout=[0.0, 0.3],
            weight_decay=[0.0, 1e-4], pos_weight=[None, "balanced"])
MAX_EPOCHS = 100
EARLY_STOP = "auroc"  # keep the epoch with the highest dev AUROC (same metric as model selection)
SEEDS = (0, 1, 2)


def balanced_pos_weight(ytr) -> float:
    """n_neg / n_pos on the training labels."""
    ytr = torch.as_tensor(ytr)
    return float((ytr == 0).sum() / (ytr == 1).sum())


# ---------------------------------------------------------------- sanity checks
def sanity_checks(Xtr, ytr, ydev, hidden_dim=256, save_path=None, tiny_n=20, tiny_epochs=300) -> dict:
    """(a) always-"not hateful" dev baseline, (b) initial loss vs ln 2, (c) overfit tiny_n train examples."""
    ydev_np = np.asarray(ydev).astype(int)
    baseline = compute_metrics(ydev_np, np.zeros_like(ydev_np), np.zeros(len(ydev_np)))

    init = initial_loss(Xtr, ytr, hidden_dim)

    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(Xtr), generator=gen)[:tiny_n]
    Xs, ys = Xtr[idx], ytr[idx]
    res = train_classifier(Xs, ys, Xs, ys, hidden_dim=hidden_dim, lr=1e-3, epochs=tiny_epochs, seed=0,
                           early_stop_metric="last")  # report the final epoch
    p = predict_proba(res.model, Xs)
    out = {
        "a_dumb_baseline_dev": baseline,
        "b_initial_loss": {"loss": init, "ln2": math.log(2), "abs_diff": abs(init - math.log(2))},
        "c_tiny_overfit": {"n_examples": tiny_n, "n_hateful": int(ys.sum()), "epochs": tiny_epochs,
                           "hidden_dim": hidden_dim, "lr": 1e-3,
                           "final_train_loss": res.curves["train_loss"][-1],
                           "train_accuracy": float(((p >= 0.5).float() == ys).float().mean())},
    }
    if save_path:
        Path(save_path).write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- grid search
def _configs(grid: dict):
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))


def run_grid(name, Xtr, ytr, Xdev, ydev, grid=GRID, epochs=MAX_EPOCHS, seed=0, early_stop_metric=EARLY_STOP,
             force=False, verbose=True):
    """Train every config (seed 0), record dev metrics at each run's selected epoch (early_stop_metric).
    Cached to results/grid_<name>.csv; returns it sorted by dev AUROC (ties: lower dev loss).
    A cached grid made with different settings raises instead of being silently reused."""
    path = RESULTS / f"grid_{name}.csv"
    configs = list(_configs(grid))
    if path.exists() and not force:
        df = pd.read_csv(path)
        same = (len(df) == len(configs) and "early_stop_metric" in df and "max_epochs" in df
                and (df["early_stop_metric"] == early_stop_metric).all() and (df["max_epochs"] == epochs).all())
        if not same:
            raise RuntimeError(f"{path} was made with different grid settings; rerun with force=True")
        return df
    pw_bal = balanced_pos_weight(ytr)
    rows = []
    t0 = time.perf_counter()
    for i, cfg in enumerate(configs, 1):
        pw = pw_bal if cfg["pos_weight"] == "balanced" else None
        t = time.perf_counter()
        res = train_classifier(Xtr, ytr, Xdev, ydev, epochs=epochs, seed=seed, early_stop_metric=early_stop_metric,
                               **{k: v for k, v in cfg.items() if k != "pos_weight"}, pos_weight=pw)
        rows.append({**cfg, "pos_weight": "none" if pw is None else f"{pw:.4f}",
                     "early_stop_metric": early_stop_metric, "max_epochs": epochs, "best_epoch": res.best_epoch,
                     "min_dev_loss_epoch": int(np.argmin(res.curves["dev_loss"])),
                     **{f"dev_{k}": v for k, v in res.dev_metrics.items()}, "seconds": time.perf_counter() - t})
        if verbose:
            print(f"[{i:2d}/{len(configs)}] {cfg}  best_epoch={res.best_epoch:2d}  "
                  f"dev_auroc={res.dev_metrics['auroc']:.4f}  dev_loss={res.dev_metrics['dev_loss']:.4f}")
    df = (pd.DataFrame(rows).sort_values(["dev_auroc", "dev_dev_loss"], ascending=[False, True])
          .rename(columns={"dev_dev_loss": "dev_loss"}).reset_index(drop=True))
    df.to_csv(path, index=False)
    if verbose:
        print(f"grid done in {time.perf_counter() - t0:.0f}s -> {path}")
    return df


def config_from_row(row) -> dict:
    """Turn a grid row back into train_classifier kwargs."""
    pw = str(row["pos_weight"])
    return dict(hidden_dim=int(row["hidden_dim"]), lr=float(row["lr"]), dropout=float(row["dropout"]),
                weight_decay=float(row["weight_decay"]), pos_weight=None if pw == "none" else float(pw))


def run_seeds(Xtr, ytr, Xdev, ydev, config: dict, epochs=MAX_EPOCHS, seeds=SEEDS, early_stop_metric=EARLY_STOP):
    """Retrain one config with several seeds. Returns the list of TrainResults (seed order)."""
    return [train_classifier(Xtr, ytr, Xdev, ydev, epochs=epochs, seed=s, early_stop_metric=early_stop_metric,
                             **config) for s in seeds]


# ---------------------------------------------------------------- dev results table
def summarize(metric_dicts) -> dict:
    """mean and (sample) std over seeds for each metric; std = 0 for a single run."""
    out = {}
    for m in METRICS:
        vals = np.array([d[m] for d in metric_dicts], dtype=float)
        out[f"{m}_mean"] = vals.mean()
        out[f"{m}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
    out["n_runs"] = len(metric_dicts)
    return out


def record_dev_result(model_name: str, metric_dicts, notes: str = "") -> pd.DataFrame:
    """Insert/replace one row of results/dev_results.csv (re-running a step doesn't duplicate rows)."""
    row = {"model": model_name, **summarize(metric_dicts), "notes": notes}
    new = pd.DataFrame([row])
    if DEV_RESULTS_CSV.exists():
        df = pd.read_csv(DEV_RESULTS_CSV)
        if (df["model"] == model_name).any():  # replace in place, keeping the row order
            i = df.index[df["model"] == model_name][0]
            df = pd.concat([df.iloc[:i], new, df.iloc[i + 1:]], ignore_index=True)
        else:
            df = pd.concat([df, new], ignore_index=True)
    else:
        df = new
    df.to_csv(DEV_RESULTS_CSV, index=False)
    return df


def dev_table() -> pd.DataFrame:
    """results/dev_results.csv formatted as 'mean ± std'."""
    df = pd.read_csv(DEV_RESULTS_CSV)
    out = pd.DataFrame({"model": df["model"]})
    for m in METRICS:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
    out["runs"] = df["n_runs"]
    return out.set_index("model")
