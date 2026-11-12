"""Frozen pretrained encoders -> cached embeddings.

Encoders (all eval mode, torch.no_grad(), never fine-tuned):
  text_bert     bert-base-uncased, last-layer [CLS] (last_hidden_state[:, 0, :])   -> [n, 768]
  image_resnet  torchvision resnet18 (ResNet18_Weights.DEFAULT), fc = Identity     -> [n, 512]
  clip          open_clip ViT-B-32 / "openai": raw (unnormalized) encode_image and
                encode_text, saved separately as clip_image and clip_text          -> [n, 512] each

Cache layout (embeddings/ is git-ignored):
  embeddings/{split}_{name}_{tag}.pt   name in text_bert, image_resnet, clip_image, clip_text
  embeddings/{split}_meta_{tag}.csv    id, text, label, in jsonl row order
  tag = "200" (first 200 rows of the split's jsonl) or "all".
Existing files are loaded instead of recomputed.

CLI:
  python -m src.embed --subset 200 [--encoders text_bert image_resnet clip] [--splits train dev test]
  python -m src.embed --subset all
  python -m src.embed --subset 200 --check     # shapes / finite / id order / determinism
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data import load_split

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "embeddings"
SPLITS = ["train", "dev", "test"]
ENCODERS = ["text_bert", "image_resnet", "clip"]
OUTPUTS = {"text_bert": ["text_bert"], "image_resnet": ["image_resnet"], "clip": ["clip_image", "clip_text"]}
DIMS = {"text_bert": 768, "image_resnet": 512, "clip_image": 512, "clip_text": 512}
BATCH_SIZE = 64
BERT_MAX_LEN = 128
SEED = 510


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _deterministic() -> None:
    # TF32 off + deterministic cuDNN so the same input gives the same vector regardless of batch.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def emb_path(split: str, name: str, tag: str) -> Path:
    return EMB_DIR / f"{split}_{name}_{tag}.pt"


def meta_path(split: str, tag: str) -> Path:
    return EMB_DIR / f"{split}_meta_{tag}.csv"


def subset(split: str, tag: str) -> pd.DataFrame:
    """Rows to embed, in jsonl order: the first int(tag) rows, or all rows for tag == 'all'."""
    df = load_split(split)
    if tag != "all":
        df = df.iloc[: int(tag)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- encoders
class _Images(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


def _image_loader(paths, transform, workers: int):
    return DataLoader(_Images(paths, transform), batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


class TextBert:
    name = "text_bert"

    def __init__(self, dev):
        from transformers import AutoModel, AutoTokenizer
        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(dev).eval()

    @torch.no_grad()
    def __call__(self, df, workers=0, progress=True):
        texts = df["text"].astype(str).tolist()
        out = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="text_bert", disable=not progress):
            # pad to a fixed length so a text's vector doesn't depend on what else is in its batch
            enc = self.tok(texts[i:i + BATCH_SIZE], padding="max_length", truncation=True,
                           max_length=BERT_MAX_LEN, return_tensors="pt").to(self.dev)
            out.append(self.model(**enc).last_hidden_state[:, 0, :].float().cpu())
        return {"text_bert": torch.cat(out)}


class ImageResnet:
    name = "image_resnet"

    def __init__(self, dev):
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.DEFAULT
        self.dev = dev
        self.transform = weights.transforms()
        model = resnet18(weights=weights)
        model.fc = nn.Identity()
        self.model = model.to(dev).eval()

    @torch.no_grad()
    def __call__(self, df, workers=0, progress=True):
        out = []
        for x in tqdm(_image_loader(df["path"], self.transform, workers), desc="image_resnet", disable=not progress):
            out.append(self.model(x.to(self.dev, non_blocking=True)).float().cpu())
        return {"image_resnet": torch.cat(out)}


class Clip:
    name = "clip"

    def __init__(self, dev):
        import open_clip
        self.dev = dev
        # OpenAI's CLIP weights were trained with QuickGELU; the plain "ViT-B-32" config uses GELU,
        # so force QuickGELU or the activations silently don't match the weights.
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", force_quick_gelu=True)
        self.model = model.to(dev).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")  # context 77, truncates

    @torch.no_grad()
    def __call__(self, df, workers=0, progress=True):
        imgs, txts = [], []
        for x in tqdm(_image_loader(df["path"], self.preprocess, workers), desc="clip_image", disable=not progress):
            imgs.append(self.model.encode_image(x.to(self.dev, non_blocking=True)).float().cpu())
        texts = df["text"].astype(str).tolist()
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="clip_text", disable=not progress):
            tokens = self.tokenizer(texts[i:i + BATCH_SIZE]).to(self.dev)
            txts.append(self.model.encode_text(tokens).float().cpu())
        return {"clip_image": torch.cat(imgs), "clip_text": torch.cat(txts)}  # raw, not normalized


ENCODER_CLASSES = {"text_bert": TextBert, "image_resnet": ImageResnet, "clip": Clip}


# ---------------------------------------------------------------- caching
def embed(tag: str, encoders=ENCODERS, splits=SPLITS, workers: int = 0) -> dict:
    """Compute (or load from cache) embeddings for every split x encoder. Returns {(split, name): tensor}."""
    set_seed()
    _deterministic()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    results = {}
    for split in splits:
        df = subset(split, tag)
        mp = meta_path(split, tag)
        if not mp.exists():
            df[["id", "text", "label"]].to_csv(mp, index=False)

    for enc in encoders:
        missing = [s for s in splits if not all(emb_path(s, n, tag).exists() for n in OUTPUTS[enc])]
        for s in splits:
            if s not in missing:
                for n in OUTPUTS[enc]:
                    results[(s, n)] = torch.load(emb_path(s, n, tag))
                print(f"[cache] {enc:12s} {s:5s} loaded")
        if not missing:
            continue
        t0 = time.perf_counter()
        model = ENCODER_CLASSES[enc](dev)
        print(f"[load ] {enc:12s} model ready on {dev} in {time.perf_counter() - t0:.1f}s")
        for s in missing:
            df = subset(s, tag)
            t1 = time.perf_counter()
            out = model(df, workers=workers)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            for n, X in out.items():
                assert X.shape == (len(df), DIMS[n]), (n, X.shape)
                torch.save(X.contiguous(), emb_path(s, n, tag))
                results[(s, n)] = X
            print(f"[embed] {enc:12s} {s:5s} n={len(df):5d}  {dt:7.1f}s  ({len(df) / dt:6.1f} items/s)")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------- checks
def check(tag: str, encoders=ENCODERS, splits=SPLITS, n_redo: int = 4, atol: float = 1e-4) -> bool:
    """Shapes, finiteness, row order vs jsonl, and determinism (re-embed first n_redo rows)."""
    _deterministic()
    dev = device()
    ok = True
    names = [n for e in encoders for n in OUTPUTS[e]]
    for split in splits:
        df = subset(split, tag)
        meta = pd.read_csv(meta_path(split, tag), dtype={"id": str})
        order_ok = meta["id"].tolist() == df["id"].tolist() and meta["label"].tolist() == df["label"].tolist()
        ok &= order_ok
        print(f"{split:5s} meta rows={len(meta)}  ids/labels match jsonl order: {order_ok}")
        for n in names:
            X = torch.load(emb_path(split, n, tag))
            shape_ok = tuple(X.shape) == (len(df), DIMS[n])
            finite = bool(torch.isfinite(X).all())
            ok &= shape_ok and finite
            print(f"   {n:12s} shape={tuple(X.shape)}  ok={shape_ok}  finite={finite}  "
                  f"mean={X.mean():+.3f} std={X.std():.3f}")
    # determinism: fresh models, first n_redo rows of each split, compared to the cached rows
    head = {s: subset(s, tag).iloc[:n_redo] for s in splits}
    for enc in encoders:
        model = ENCODER_CLASSES[enc](dev)
        for s in splits:
            out = model(head[s], progress=False)
            for n, X in out.items():
                cached = torch.load(emb_path(s, n, tag))[:n_redo]
                same = torch.allclose(X, cached, atol=atol)
                ok &= same
                print(f"determinism {n:12s} {s:5s} allclose(atol={atol}): {same}  "
                      f"max|diff|={(X - cached).abs().max():.2e}")
        del model
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", required=True, help='"200" (first 200 rows per split) or "all"')
    ap.add_argument("--encoders", nargs="+", default=ENCODERS, choices=ENCODERS)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers for images (0 is safest on Windows)")
    ap.add_argument("--check", action="store_true", help="run checks on the cached files instead of embedding")
    a = ap.parse_args()
    tag = a.subset if a.subset == "all" else str(int(a.subset))
    if a.check:
        raise SystemExit(0 if check(tag, a.encoders, a.splits) else 1)
    t0 = time.perf_counter()
    embed(tag, a.encoders, a.splits, a.workers)
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
