"""Load cached embeddings (written by `python -m src.embed`) as model-ready tensors."""
import pandas as pd
import torch

from src.embed import DIMS, emb_path, meta_path, subset

ENCODER_NAMES = list(DIMS)  # text_bert, image_resnet, clip_image, clip_text

# The test set is evaluated exactly once, at the very end. Until then it stays locked:
# load_features("test", ...) raises. The final step calls unlock_test() explicitly.
TEST_LOCKED = True


def unlock_test() -> None:
    """Only for the final, one-time test evaluation (after every decision has been made on dev)."""
    global TEST_LOCKED
    TEST_LOCKED = False


def feature_shapes(tag: str = "all") -> pd.DataFrame:
    """Shapes of all cached embedding files (no labels loaded; safe for the locked test split)."""
    rows = []
    for split in ["train", "dev", "test"]:
        for enc in ENCODER_NAMES:
            X = torch.load(emb_path(split, enc, tag), map_location="cpu")
            rows.append({"split": split, "encoder": enc, "shape": tuple(X.shape)})
    return pd.DataFrame(rows)


def load_features(split: str, encoder: str, tag: str = "all"):
    """Return (X float32 [n, d], y float32 [n], meta DataFrame[id, text, label]) for one split/encoder.

    Rows are in the split's jsonl order; checked against the meta file and the jsonl itself.
    """
    assert split != "test" or not TEST_LOCKED, (
        "test split is locked: every decision is made on dev and test is evaluated once at the end "
        "(call src.features.unlock_test() only in the final evaluation step)")
    if encoder not in DIMS:
        raise ValueError(f"unknown encoder {encoder!r}; use one of {ENCODER_NAMES}")
    p = emb_path(split, encoder, tag)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run: python -m src.embed --subset {tag}")
    X = torch.load(p).float()
    meta = pd.read_csv(meta_path(split, tag), dtype={"id": str}, keep_default_na=False)
    assert len(X) == len(meta), f"{split}/{encoder}: {len(X)} rows vs {len(meta)} meta rows"
    assert X.shape[1] == DIMS[encoder], f"{split}/{encoder}: dim {X.shape[1]} != {DIMS[encoder]}"
    expected_ids = subset(split, tag)["id"].tolist()
    assert meta["id"].tolist() == expected_ids, f"{split}/{encoder}: meta ids don't match jsonl order"
    y = torch.tensor(meta["label"].to_numpy(), dtype=torch.float32)
    return X, y, meta
