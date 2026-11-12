"""Reusable figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#4C72B0", "#DD5555"]  # not hateful, hateful


def plot_loss_curves(curves: dict, best_epoch: int, title: str, save_path=None):
    """Train and dev loss per epoch (plain BCE, eval mode), with the selected (lowest dev loss) epoch marked."""
    ep = np.arange(1, len(curves["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ep, curves["train_loss"], label="train loss", color="#4C72B0")
    ax.plot(ep, curves["dev_loss"], label="dev loss", color="#DD5555")
    ax.axvline(best_epoch, color="gray", ls="--", lw=1)
    ax.scatter([best_epoch], [curves["dev_loss"][best_epoch - 1]], color="#DD5555", zorder=3)
    ax.annotate(f"best epoch {best_epoch}\ndev loss {curves['dev_loss'][best_epoch - 1]:.3f}",
                (best_epoch, curves["dev_loss"][best_epoch - 1]), textcoords="offset points",
                xytext=(8, 10), fontsize=9)
    ax.axhline(np.log(2), color="lightgray", lw=0.8, ls=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss (unweighted)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
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
