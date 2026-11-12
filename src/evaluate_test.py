"""Step 7: the single, final test-set evaluation (run ONCE, after every decision was made on dev).

    python -m src.evaluate_test            # refuses to run if results/test_results.csv already exists

Loads the saved seed-0/1/2 models (results/models/<name>_seed<k>.pt, never retrained here), evaluates them on
test_seen at the frozen threshold 0.5, and writes results/test_results.csv (same layout as dev_results.csv) and
results/preds_test_<model>.csv (id, label, prob_seed0..2). Late fusion = the same seed-paired plain average as on
dev. Each model applies its own stored train-set standardization (mean/std buffers inside StandardizedClassifier).
"""
import argparse
import sys

import numpy as np
import pandas as pd

from src.experiments import RESULTS, SEEDS, seed_metrics_from_preds, summarize
from src.metrics import compute_metrics

# model name -> encoders whose raw embeddings are concatenated (in this order) as the model input
TRAINED = {
    "text_bert": ("text_bert",),
    "image_resnet": ("image_resnet",),
    "early_concat": ("text_bert", "image_resnet"),
    "clip_text": ("clip_text",),
    "clip_image": ("clip_image",),
    "clip_early_concat": ("clip_text", "clip_image"),
}
LATE = {"late_avg": ("text_bert", "image_resnet"), "clip_late_avg": ("clip_text", "clip_image")}
ORDER = ["always_not_hateful", "text_bert", "image_resnet", "early_concat", "late_avg",
         "clip_text", "clip_image", "clip_early_concat", "clip_late_avg"]
DESIGN_CHANGE_510 = "clip_early_concat"
THRESHOLD = 0.5  # frozen

TEST_RESULTS_CSV = RESULTS / "test_results.csv"


def seed_model_name(model: str, seed: int) -> str:
    return f"{model}_seed{seed}"


def test_preds_path(model: str):
    return RESULTS / f"preds_test_{model}.csv"


def evaluate() -> pd.DataFrame:
    import src.features as F
    from src.train import load_model, predict_proba

    F.unlock_test()  # the one and only place the test split is opened
    rows, preds = [], {}

    # always-"not hateful" baseline (same definition as on dev: predict 0, constant probability 0)
    _, y, meta = F.load_concat("test", ("text_bert",))
    y_np = y.numpy().astype(int)
    base = compute_metrics(y_np, np.zeros_like(y_np), np.zeros(len(y_np)))
    rows.append({"model": "always_not_hateful", **summarize([base]), "notes": "constant prediction 0"})

    for model, encoders in TRAINED.items():
        X, y_m, meta_m = F.load_concat("test", encoders)
        assert meta_m["id"].tolist() == meta["id"].tolist()
        df = pd.DataFrame({"id": meta_m["id"].astype(str).values, "label": meta_m["label"].astype(int).values})
        for s in SEEDS:
            m = load_model(seed_model_name(model, s))
            assert m.config["seed"] == s and m.config["input_dim"] == X.shape[1]
            df[f"prob_seed{s}"] = predict_proba(m, X).numpy()
        df.to_csv(test_preds_path(model), index=False)
        preds[model] = df

    for model, (a, b) in LATE.items():
        pa, pb = preds[a], preds[b]
        assert pa["id"].tolist() == pb["id"].tolist()
        df = pa[["id", "label"]].copy()
        for s in SEEDS:
            df[f"prob_seed{s}"] = (pa[f"prob_seed{s}"] + pb[f"prob_seed{s}"]) / 2
        df.to_csv(test_preds_path(model), index=False)
        preds[model] = df

    for model in ORDER[1:]:
        note = "510 design change" if model == DESIGN_CHANGE_510 else ""
        rows.append({"model": model, **summarize(seed_metrics_from_preds(preds[model], THRESHOLD)),
                     "notes": f"seeds {','.join(map(str, SEEDS))}; threshold {THRESHOLD}" + (f"; {note}" if note else "")})
    out = pd.DataFrame(rows)
    out.to_csv(TEST_RESULTS_CSV, index=False)
    return out


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    from src.experiments import METRICS
    out = pd.DataFrame({"model": df["model"]})
    for m in METRICS:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
    out["runs"] = df["n_runs"]
    return out.set_index("model").loc[ORDER]


def test_table(mark_design_change: bool = True) -> pd.DataFrame:
    """results/test_results.csv as 'mean ± std' (reads the saved file; never re-evaluates)."""
    t = _fmt(pd.read_csv(TEST_RESULTS_CSV))
    if mark_design_change:
        t = t.rename(index={DESIGN_CHANGE_510: f"{DESIGN_CHANGE_510} (510 design change)"})
    return t


def dev_vs_test_auroc() -> pd.DataFrame:
    """Dev and test AUROC (3-seed mean ± std) side by side, plus test - dev."""
    from src.experiments import DEV_RESULTS_CSV
    d = pd.read_csv(DEV_RESULTS_CSV).set_index("model").loc[ORDER]
    t = pd.read_csv(TEST_RESULTS_CSV).set_index("model").loc[ORDER]
    return pd.DataFrame({
        "dev AUROC": [f"{a:.3f} ± {b:.3f}" for a, b in zip(d["auroc_mean"], d["auroc_std"])],
        "test AUROC": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t["auroc_mean"], t["auroc_std"])],
        "test - dev": (t["auroc_mean"] - d["auroc_mean"]).round(3).values,
    }, index=ORDER)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing test evaluation (do NOT use)")
    args = ap.parse_args(argv)
    if TEST_RESULTS_CSV.exists() and not args.force:
        sys.exit(f"{TEST_RESULTS_CSV} already exists: the test set is evaluated exactly once. Refusing to run.")
    out = evaluate()
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
