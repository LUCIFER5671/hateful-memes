"""Reusable figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["#4C72B0", "#DD5555"]  # not hateful, hateful


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
