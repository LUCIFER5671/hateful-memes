"""Recreate the seed-0/1/2 model files needed by src.evaluate_test (train + dev only; never touches test).

Steps 4-6 trained seeds 0, 1, 2 for every model but saved only the seed-0 model to disk. Training is deterministic,
so this script retrains each model with its FROZEN config (the top row of its saved grid; nothing is re-selected)
and the same seeds, then refuses to save unless the result is identical to what Steps 4-6 produced:
  - seed 0: weights equal to the saved results/models/<name>.pt
  - every seed: dev probabilities equal (max abs diff <= 1e-6) to results/preds_dev_<name>.csv
Only then it writes results/models/<name>_seed<k>.pt.

    python -m src.restore_seed_models
"""
import sys

import numpy as np
import torch

from src.evaluate_test import TRAINED, seed_model_name
from src.experiments import RESULTS, SEEDS, config_from_row, load_dev_preds, run_seeds
from src.features import load_concat
from src.train import MODEL_DIR, predict_proba, save_model
import pandas as pd

TOL = 1e-6


def restore(model: str, encoders) -> dict:
    grid = pd.read_csv(RESULTS / f"grid_{model}.csv")
    cfg = config_from_row(grid.iloc[0])
    saved0 = torch.load(MODEL_DIR / f"{model}.pt")
    for k, v in cfg.items():
        assert saved0["config"][k] == v, f"{model}: grid config {k}={v} != saved model {saved0['config'][k]}"

    Xtr, ytr, _ = load_concat("train", encoders)
    Xdev, ydev, meta_dev = load_concat("dev", encoders)
    runs = run_seeds(Xtr, ytr, Xdev, ydev, cfg)

    ref = load_dev_preds(model)
    assert ref["id"].tolist() == meta_dev["id"].astype(str).tolist()
    report = {"model": model, "best_epochs": [r.best_epoch for r in runs]}
    ok = True
    sd0 = runs[0].model.state_dict()
    report["seed0_weights_equal"] = all(torch.equal(sd0[k].cpu(), saved0["state_dict"][k].cpu()) for k in sd0)
    ok &= report["seed0_weights_equal"]
    for r in runs:
        s = r.config["seed"]
        diff = float(np.abs(predict_proba(r.model, Xdev).numpy() - ref[f"prob_seed{s}"].to_numpy()).max())
        report[f"dev_maxdiff_seed{s}"] = diff
        ok &= diff <= TOL
    report["ok"] = bool(ok)
    if ok:
        for r in runs:
            save_model(r, seed_model_name(model, r.config["seed"]))
    return report


def main():
    reports = []
    for model, encoders in TRAINED.items():
        rep = restore(model, encoders)
        print(rep, flush=True)
        reports.append(rep)
    bad = [r["model"] for r in reports if not r["ok"]]
    if bad:
        sys.exit(f"NOT identical to Steps 4-6, nothing saved for: {bad}")
    print("all seed models restored and verified")


if __name__ == "__main__":
    main()
