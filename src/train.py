"""One training function for every model in the lab (and reusable for project Phase 2b).

    res = train_classifier(Xtr, ytr, Xdev, ydev, hidden_dim=256, lr=1e-3, epochs=50)
    res.model, res.curves, res.best_epoch, res.dev_metrics
"""
import copy
import json
import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.metrics import compute_metrics
from src.models import StandardizedClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results" / "models"


class TrainResult(NamedTuple):
    model: StandardizedClassifier  # weights from best_epoch, on CPU, eval mode
    curves: dict                   # index = epoch (0 = untrained): train_loss, dev_loss (eval mode, unweighted),
                                   # dev_auroc, train_objective (NaN at epoch 0)
    best_epoch: int                # selected epoch (0 = untrained model) per early_stop_metric
    dev_metrics: dict              # compute_metrics at best_epoch (threshold 0.5 on sigmoid, + AUROC) + dev_loss
    config: dict


EARLY_STOP_METRICS = ("auroc", "loss", "last")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def standardization(Xtr: torch.Tensor):
    """Mean/std per feature from the TRAINING set only (std clamped to avoid division by 0)."""
    Xtr = Xtr.float()
    return Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)


@torch.no_grad()
def predict_proba(model: nn.Module, X: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """Sigmoid probabilities for raw (unstandardized) features; the model standardizes internally."""
    model.eval()
    dev = next(model.parameters()).device
    out = [torch.sigmoid(model(X[i:i + batch_size].float().to(dev))).squeeze(1).cpu()
           for i in range(0, len(X), batch_size)]
    return torch.cat(out)


def initial_loss(Xtr, ytr, hidden_dim: int, batch_size: int = 64, seed: int = 0) -> float:
    """Sanity check: loss of a freshly initialized model on one shuffled training batch (expect ~ln 2)."""
    set_seed(seed)
    mean, std = standardization(Xtr)
    model = StandardizedClassifier(Xtr.shape[1], hidden_dim, 0.0, mean, std).eval()
    idx = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(seed))[:batch_size]
    with torch.no_grad():
        return nn.BCEWithLogitsLoss()(model(Xtr[idx].float()), ytr[idx].float().unsqueeze(1)).item()


def train_classifier(Xtr, ytr, Xdev, ydev, hidden_dim, lr, epochs, batch_size=64, dropout=0.0,
                     weight_decay=0.0, pos_weight=None, seed=0, early_stop_metric="auroc") -> TrainResult:
    """Train a StandardizedClassifier with BCEWithLogitsLoss + Adam on shuffled mini-batches.

    - Standardization stats come from Xtr only and are stored inside the model.
    - pos_weight (if given) only affects the training objective; the recorded train/dev losses are
      always the plain (unweighted) BCE in eval mode, so curves are comparable across configs.
    - Curves include epoch 0 (the untrained model).
    - early_stop_metric selects which epoch's weights are returned:
        "auroc" (default): highest dev AUROC;  "loss": lowest dev loss;  "last": final epoch
      (ties -> earliest epoch). "last" is for the tiny-batch overfit sanity check.
    """
    if early_stop_metric not in EARLY_STOP_METRICS:
        raise ValueError(f"early_stop_metric must be one of {EARLY_STOP_METRICS}")
    config = dict(hidden_dim=int(hidden_dim), lr=float(lr), epochs=int(epochs), batch_size=int(batch_size),
                  dropout=float(dropout), weight_decay=float(weight_decay),
                  pos_weight=None if pos_weight is None else float(pos_weight), seed=int(seed),
                  early_stop_metric=early_stop_metric, input_dim=int(Xtr.shape[1]))
    set_seed(seed)
    dev = device()
    Xtr, Xdev = Xtr.float().to(dev), Xdev.float().to(dev)
    ytr_c, ydev_c = ytr.float().view(-1, 1).to(dev), ydev.float().view(-1, 1).to(dev)
    mean, std = standardization(Xtr)
    model = StandardizedClassifier(Xtr.shape[1], hidden_dim, dropout, mean, std).to(dev)

    pw = None if pos_weight is None else torch.tensor([float(pos_weight)], device=dev)
    objective = nn.BCEWithLogitsLoss(pos_weight=pw)  # what we optimize
    plain = nn.BCEWithLogitsLoss()                    # what we record (comparable across configs)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)

    ydev_np = ydev.cpu().numpy().astype(int)
    curves = {"train_loss": [], "dev_loss": [], "dev_auroc": [], "train_objective": []}
    best = (None, 0, None)  # (score where higher is better, epoch, state_dict)

    def evaluate(epoch, objective_value):
        nonlocal best
        model.eval()
        with torch.no_grad():
            tr_loss = plain(model(Xtr), ytr_c).item()
            dev_logits = model(Xdev)
            dev_loss = plain(dev_logits, ydev_c).item()
        dev_auc = roc_auc_score(ydev_np, dev_logits.squeeze(1).cpu().numpy())
        curves["train_loss"].append(tr_loss)
        curves["dev_loss"].append(dev_loss)
        curves["dev_auroc"].append(dev_auc)
        curves["train_objective"].append(objective_value)
        score = {"auroc": dev_auc, "loss": -dev_loss, "last": epoch}[early_stop_metric]
        if best[0] is None or score > best[0]:
            best = (score, epoch, copy.deepcopy(model.state_dict()))

    evaluate(0, float("nan"))  # the untrained model
    n = len(Xtr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        total = torch.zeros((), device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = objective(model(Xtr[idx]), ytr_c[idx])
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(idx)
        evaluate(epoch, (total / n).item())

    best_epoch = best[1]
    model.load_state_dict(best[2])
    model = model.cpu().eval()
    probs = predict_proba(model, Xdev.cpu())
    metrics = compute_metrics(ydev_np, (probs >= 0.5).int().numpy(), probs.numpy())
    metrics["dev_loss"] = curves["dev_loss"][best_epoch]
    return TrainResult(model, curves, best_epoch, metrics, config)


def save_model(result: TrainResult, name: str) -> Path:
    """results/models/<name>.pt (weights incl. standardization buffers + config) and <name>.json (config)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(result.config, best_epoch=result.best_epoch)
    path = MODEL_DIR / f"{name}.pt"
    torch.save({"state_dict": result.model.state_dict(), "config": cfg,
                "mean": result.model.mean, "std": result.model.std}, path)
    (MODEL_DIR / f"{name}.json").write_text(json.dumps(cfg, indent=2))
    return path


def load_model(name: str) -> StandardizedClassifier:
    ckpt = torch.load(MODEL_DIR / f"{name}.pt")
    cfg = ckpt["config"]
    model = StandardizedClassifier(cfg["input_dim"], cfg["hidden_dim"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.config = cfg
    return model.eval()
