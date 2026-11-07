"""Dataset loading + verification for the Hateful Memes lab.

    from src.data import load_split
    df = load_split("train")      # or "dev" / "test" (aliases for dev_seen / test_seen)

`python -m src.data` re-runs the download checks (counts, missing images, PIL opens, disk size).
Never prints meme texts.
"""
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "hateful_memes"

SPLITS = {"train": "train.jsonl", "dev": "dev_seen.jsonl", "test": "test_seen.jsonl"}
ALIASES = {"dev_seen": "dev", "test_seen": "test"}
EXPECTED_COLUMNS = ["id", "img", "label", "text"]


def load_split(name: str) -> pd.DataFrame:
    """Load a split as a DataFrame with columns id, img, label, text, path (absolute image path)."""
    name = ALIASES.get(name, name)
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; use one of {list(SPLITS)}")
    df = pd.read_json(DATA_DIR / SPLITS[name], lines=True, dtype={"id": str})
    df["path"] = [str((DATA_DIR / p).resolve()) for p in df["img"]]
    return df


LABEL_NAMES = {0: "not hateful", 1: "hateful"}


def show_meme(row, show_label: bool = True, ax=None, wrap: int = 45):
    """Show one meme (image + id + text, plus the label if show_label). `row` is a load_split() row."""
    import textwrap

    import matplotlib.pyplot as plt
    from PIL import Image

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4.6))
    with Image.open(row["path"]) as im:
        ax.imshow(im.convert("RGB"))
    ax.set_xticks([]), ax.set_yticks([])
    title = f"id {row['id']}"
    if show_label:
        title += f"  |  {LABEL_NAMES[int(row['label'])]}"
    ax.set_title(title, fontsize=10, fontweight="bold",
                 color=("crimson" if show_label and int(row["label"]) == 1 else "black"))
    ax.set_xlabel(textwrap.fill(str(row["text"]), wrap), fontsize=8)
    return ax


def show_memes(df: pd.DataFrame, show_label: bool = True, ncols: int = 4, save_path=None):
    """Grid of memes, in the order of df's rows."""
    import matplotlib.pyplot as plt

    nrows = -(-len(df) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.8 * nrows), squeeze=False)
    for ax in axes.flat[len(df):]:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, df.iterrows()):
        show_meme(row, show_label=show_label, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def sample_balanced(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """n_per_class rows from each label, shuffled together (fixed seed)."""
    parts = [df[df.label == c].sample(n_per_class, random_state=seed) for c in (0, 1)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def verify(n_open: int = 5, seed: int = 0) -> bool:
    from PIL import Image

    ok = True
    frames = {}
    print(f"{'split':6s} {'rows':>6s} {'hateful':>8s} {'%hateful':>9s}  columns")
    for name in SPLITS:
        df = load_split(name)
        frames[name] = df
        cols = [c for c in df.columns if c != "path"]
        if sorted(cols) != sorted(EXPECTED_COLUMNS):
            ok = False
        print(f"{name:6s} {len(df):6d} {int(df.label.sum()):8d} {100 * df.label.mean():8.1f}%  {cols}")

    all_df = pd.concat(frames.values(), keys=frames.keys(), names=["split"]).reset_index(level=0)
    dup_ids = all_df.id.duplicated().sum()
    print(f"\nduplicate ids across splits: {dup_ids}")
    print(f"label values: {sorted(all_df.label.unique().tolist())}")
    print(f"empty texts: {int((all_df.text.str.strip() == '').sum())}")

    missing = all_df[~all_df.path.map(lambda p: Path(p).is_file())]
    print(f"\nimages referenced: {len(all_df)}; missing on disk: {len(missing)}")
    for _, r in missing.head(50).iterrows():
        print(f"  MISSING [{r.split}] {r.img}")
    ok &= missing.empty

    print(f"\nopening {n_open} random images with PIL:")
    rng = random.Random(seed)
    present = all_df[all_df.path.map(lambda p: Path(p).is_file())]
    for i in rng.sample(range(len(present)), min(n_open, len(present))):
        r = present.iloc[i]
        try:
            with Image.open(r.path) as im:
                im = im.convert("RGB")
                print(f"  [{r.split}] {r.img}: {im.size[0]}x{im.size[1]} RGB")
        except Exception as e:
            ok = False
            print(f"  [{r.split}] {r.img}: FAILED ({e})")

    print(f"\ndisk size of data/: {_dir_size(ROOT / 'data') / 1e9:.2f} GB")
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
