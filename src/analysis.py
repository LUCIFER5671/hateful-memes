"""ARI 410 analysis: dev memes where fusion and single-modality models disagree (seed-0 predictions).
Uses only the saved dev predictions (results/preds_dev_<model>.csv)."""
from pathlib import Path

import pandas as pd

from src.experiments import RESULTS, load_dev_preds

MODELS = ("text_bert", "image_resnet", "early_concat", "late_avg")
DISAGREE_CSV = RESULTS / "disagreements_dev.csv"


def seed_predictions(models=MODELS, seed: int = 0, threshold: float = 0.5) -> pd.DataFrame:
    """One row per dev meme: id, label, and per model prob_<m>, pred_<m> (prob >= threshold), correct_<m>."""
    base = None
    for m in models:
        p = load_dev_preds(m)[["id", "label", f"prob_seed{seed}"]].rename(columns={f"prob_seed{seed}": f"prob_{m}"})
        if base is None:
            base = p
        else:
            assert p["id"].tolist() == base["id"].tolist() and p["label"].tolist() == base["label"].tolist()
            base[f"prob_{m}"] = p[f"prob_{m}"].values
    for m in models:
        base[f"pred_{m}"] = (base[f"prob_{m}"] >= threshold).astype(int)
        base[f"correct_{m}"] = (base[f"pred_{m}"] == base["label"]).astype(int)
    return base


def disagreements(seed: int = 0, threshold: float = 0.5, save: bool = True) -> pd.DataFrame:
    """Dev memes where (early_concat or late_avg right) and (text_bert or image_resnet wrong),
    plus every meme where early_concat was wrong. Flag columns say which case(s) apply."""
    d = seed_predictions(MODELS, seed, threshold)
    c = {m: d[f"correct_{m}"] == 1 for m in MODELS}
    d["fusion_right_single_wrong"] = ((c["early_concat"] | c["late_avg"]) & (~c["text_bert"] | ~c["image_resnet"])).astype(int)
    d["early_wrong"] = (~c["early_concat"]).astype(int)
    d["early_wrong_both_single_right"] = (~c["early_concat"] & c["text_bert"] & c["image_resnet"]).astype(int)
    out = d[(d["fusion_right_single_wrong"] == 1) | (d["early_wrong"] == 1)].reset_index(drop=True)
    cols = ["id", "label"] + [f"{k}_{m}" for m in MODELS for k in ("pred", "prob")] + \
           ["fusion_right_single_wrong", "early_wrong", "early_wrong_both_single_right"]
    out = out[cols]
    if save:
        out.to_csv(DISAGREE_CSV, index=False)
    return out


def pick_examples(dis: pd.DataFrame, seed: int = 510) -> dict:
    """Mechanical pick (fixed seed, no look at content):
    - 2 where fusion was right and a single model was wrong, preferring memes where text_bert was wrong
      (and, among those, where both fusion models were right);
    - 1 where early_concat was wrong while text_bert and image_resnet were both right (if any)."""
    fr = dis[dis["fusion_right_single_wrong"] == 1].copy()
    fr["text_wrong"] = (fr["pred_text_bert"] != fr["label"]).astype(int)
    fr["both_fusion_right"] = ((fr["pred_early_concat"] == fr["label"]) & (fr["pred_late_avg"] == fr["label"])).astype(int)
    picks = []
    for tiers in [(1, 1), (1, 0), (0, 1), (0, 0)]:  # (text_wrong, both_fusion_right), most preferred first
        pool = fr[(fr["text_wrong"] == tiers[0]) & (fr["both_fusion_right"] == tiers[1]) & ~fr["id"].isin(picks)]
        picks += pool.sample(min(2 - len(picks), len(pool)), random_state=seed)["id"].tolist()
        if len(picks) == 2:
            break
    ew = dis[dis["early_wrong_both_single_right"] == 1]
    fusion_wrong = ew.sample(1, random_state=seed)["id"].tolist() if len(ew) else []
    return {"fusion_right": picks, "fusion_wrong": fusion_wrong}


def summary(dis: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "fusion right, a single model wrong": int(dis["fusion_right_single_wrong"].sum()),
        "  ...of which text_bert wrong": int(((dis["fusion_right_single_wrong"] == 1) & (dis["pred_text_bert"] != dis["label"])).sum()),
        "  ...of which image_resnet wrong": int(((dis["fusion_right_single_wrong"] == 1) & (dis["pred_image_resnet"] != dis["label"])).sum()),
        "early_concat wrong": int(dis["early_wrong"].sum()),
        "  ...of which both single models right": int(dis["early_wrong_both_single_right"].sum()),
        "rows saved": len(dis),
    })
