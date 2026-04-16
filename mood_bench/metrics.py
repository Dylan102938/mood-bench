import numpy as np


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float = 0.01) -> float | np.ndarray:
    safe = scores[..., ~labels]
    unsafe = scores[..., labels]

    thresh = np.quantile(safe, fpr, axis=-1)
    result = np.asarray((unsafe < thresh[..., np.newaxis]).mean(axis=-1))

    return float(result) if result.ndim == 0 else result
