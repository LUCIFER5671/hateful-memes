"""Reusable figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#4C72B0", "#DD5555"]  # not hateful, hateful


def plot_loss_curves(curves: dict, best_epoch: int, title: str, save_path=None):
    """Two panels over epochs (0 = untrained model), selected epoch marked in both:
    (a) train and dev loss (plain BCE, eval mode), (b) dev AUROC."""
    ep = np.arange(len(curves["train_loss"]))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax.plot(ep, curves["train_loss"], label="train loss", color="#4C72B0")
    ax.plot(ep, curves["dev_loss"], label="dev loss", color="#DD5555")
    ax.axhline(np.log(2), color="lightgray", lw=0.8, ls=":", label="ln 2 (predict 0.5)")
    dmin = int(np.argmin(curves["dev_loss"]))
    ax.scatter([dmin], [curves["dev_loss"][dmin]], facecolors="none", edgecolors="#DD5555", zorder=3,
               label=f"min dev loss (epoch {dmin})")
    ax.set_ylabel("BCE loss (unweighted)")
    ax.set_title("(a) train and dev loss")

    bx.plot(ep, curves["dev_auroc"], color="#55A868", label="dev AUROC")
    bx.axhline(0.5, color="lightgray", lw=0.8, ls=":", label="chance (0.5)")
    bx.set_ylabel("dev AUROC")
    bx.set_title("(b) dev AUROC")

    for a, key in ((ax, "dev_loss"), (bx, "dev_auroc")):
        y = curves[key][best_epoch]
        a.axvline(best_epoch, color="gray", ls="--", lw=1)
        a.scatter([best_epoch], [y], color="black", zorder=4)
        a.annotate(f"best_epoch {best_epoch}\n{key.replace('_', ' ')} {y:.3f}", (best_epoch, y),
                   textcoords="offset points", xytext=(8, -28 if key == "dev_loss" else 8), fontsize=9)
        a.set_xlabel("epoch (0 = untrained)")
        a.legend(frameon=False, fontsize=8, loc="best")
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_label_distribution(splits: dict[str, pd.DataFrame], save_path=None):
    """One panel per split (own y-scale, since split sizes differ ~17x): count of
    not-hateful vs hateful, with % hateful annotated."""
    fig, axes = plt.subplots(1, len(splits), figsize=(3.2 * len(splits), 3.8))
    for ax, (name, df) in zip(axes, splits.items()):
        counts = [int((df.label == 0).sum()), int((df.label == 1).sum())]
        bars = ax.bar(["not hateful\n(0)", "hateful\n(1)"], counts, color=COLORS, width=0.65)
        ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=2, fontsize=9)
        ax.set_ylim(0, max(counts) * 1.18)
        ax.set_title(f"{name} (n={len(df):,})\n{100 * counts[1] / len(df):.1f}% hateful",
                     fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("number of memes")
    fig.suptitle("Hateful Memes: label distribution per split (note different y-scales)", fontsize=11)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
