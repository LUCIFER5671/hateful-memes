"""Load cached embeddings (written by `python -m src.embed`) as model-ready tensors."""
import pandas as pd
import torch

from src.embed import DIMS, emb_path, meta_path, subset

ENCODER_NAMES = list(DIMS)  # text_bert, image_resnet, clip_image, clip_text


def load_features(split: str, encoder: str, tag: str = "all"):
    """Return (X float32 [n, d], y float32 [n], meta DataFrame[id, text, label]) for one split/encoder.

    Rows are in the split's jsonl order; checked against the meta file and the jsonl itself.
    """
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
