"""Part 0.3 "be the model": blind human labeling of 20 dev memes.

The id list and the labeling sheet never contain the true labels; labels are joined only in score().
"""
from pathlib import Path

import pandas as pd

from src.data import load_split
from src.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[1]
IDS_CSV = ROOT / "results" / "human_eval_ids.csv"
LABELS_CSV = ROOT / "results" / "human_labels.csv"
N, SEED = 20, 510


def prepare(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Pick n dev memes (plain random sample, fixed seed), write the id list and an empty
    labeling sheet if they don't exist yet (never overwrites your filled-in sheet).
    Returns the chosen rows WITHOUT the label column, in order."""
    dev = load_split("dev")
    if IDS_CSV.exists():
        ids = pd.read_csv(IDS_CSV, dtype={"id": str})["id"].tolist()
    else:
        ids = dev.sample(n, random_state=seed)["id"].tolist()
        IDS_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ids}).to_csv(IDS_CSV, index=False)
    if not LABELS_CSV.exists():
        pd.DataFrame({"id": ids, "my_guess": "", "needed_both": "", "notes": ""}).to_csv(LABELS_CSV, index=False)
    blind = dev.set_index("id").loc[ids].reset_index().drop(columns="label")
    return blind


def score():
    """Return (metrics dict, DataFrame of the memes guessed wrong incl. true labels),
    or None if the sheet isn't completely filled in."""
    sheet = pd.read_csv(LABELS_CSV, dtype={"id": str})
    guess = pd.to_numeric(sheet["my_guess"], errors="coerce")
    if guess.isna().any() or not guess.isin([0, 1]).all():
        print("fill in results/human_labels.csv first")
        return None
    sheet["my_guess"] = guess.astype(int)
    dev = load_split("dev")
    df = sheet.merge(dev[["id", "label", "img", "text", "path"]], on="id", how="left", validate="1:1")
    m = compute_metrics(df["label"], df["my_guess"])
    wrong = df[df["my_guess"] != df["label"]].reset_index(drop=True)
    return m, wrong
