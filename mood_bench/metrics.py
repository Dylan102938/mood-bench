from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

matplotlib.use("Agg")


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float = 0.01) -> float | np.ndarray:
    safe = scores[..., ~labels]
    unsafe = scores[..., labels]

    thresh = np.quantile(safe, 1.0 - fpr, axis=-1)
    result = np.asarray((unsafe > thresh[..., np.newaxis]).mean(axis=-1))

    return float(result) if result.ndim == 0 else result


def plot_score_hist(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
    bins: int = 30,
) -> None:
    """Save a safe-vs-unsafe score distribution histogram."""
    fig, ax = plt.subplots(figsize=(6, 4))
    safe_scores = scores[~labels]
    unsafe_scores = scores[labels]

    edges = np.histogram_bin_edges(scores, bins=bins) if scores.size > 0 else bins
    if safe_scores.size > 0:
        ax.hist(safe_scores, bins=edges, alpha=0.55, color="tab:blue", label="safe", density=True)
    if unsafe_scores.size > 0:
        ax.hist(
            unsafe_scores, bins=edges, alpha=0.55, color="tab:red", label="unsafe", density=True
        )

    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
    path: Path,
    auroc_value: float,
) -> None:
    """Save an ROC curve. No-ops when only one class is present."""
    if labels.sum() == 0 or (~labels).sum() == 0:
        return

    fpr, tpr, _ = roc_curve(labels.astype(int), scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUROC = {auroc_value:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
