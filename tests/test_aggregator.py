from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from mood_bench.aggregator import (
    _merge_meta,
    lambda_aggregate,
    mean_aggregate,
    min_aggregate,
)


def test_min_aggregate_elementwise() -> None:
    r1 = (np.array([1.0, 5.0, 3.0]), {"a": 1})
    r2 = (np.array([2.0, 4.0, 6.0]), {"b": 2})

    scores, meta = min_aggregate([r1, r2])

    assert np.allclose(scores, [1.0, 4.0, 3.0])
    assert meta == {"a": 1, "b": 2}


def test_mean_aggregate_elementwise() -> None:
    r1 = (np.array([1.0, 3.0, 5.0]), {})
    r2 = (np.array([3.0, 5.0, 7.0]), {})

    scores, _ = mean_aggregate([r1, r2])

    assert np.allclose(scores, [2.0, 4.0, 6.0])


def test_merge_meta_keeps_first_seen_key() -> None:
    merged = _merge_meta([{"x": 1}, {"x": 2, "y": 3}, {"y": 99}])

    assert merged == {"x": 1, "y": 3}


_FPR = 0.1  # lifted above the default 0.01 so the safe quantile is stable at n=1000.


def _binary_signal(
    n: int,
    *,
    unsafe_shift: float,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unsafe = np.zeros(n, dtype=bool)
    unsafe[: n // 2] = True
    rng.shuffle(unsafe)
    scores = np.where(unsafe, unsafe_shift, 0.0) + rng.normal(0.0, noise, n)
    return scores, unsafe


def test_lambda_aggregate_recovers_zero_when_aux_is_anti_correlated() -> None:
    anchor, unsafe = _binary_signal(1000, unsafe_shift=-2.0, noise=1.0, seed=0)
    rng = np.random.default_rng(1)
    # Aux flips the direction: unsafe shifted positively, swamping the anchor signal.
    aux = np.where(unsafe, 20.0, 0.0) + rng.normal(0.0, 1.0, unsafe.size)
    id_labels = np.ones(unsafe.size, dtype=bool)

    _, meta = lambda_aggregate(
        [(anchor, {}), (aux, {})],
        id_labels=id_labels,
        unsafe_labels=unsafe,
        fpr_threshold=_FPR,
    )

    assert meta["lambda_0"] == 0.0


def test_lambda_aggregate_recovers_grid_max_when_aux_equals_anchor() -> None:
    anchor, unsafe = _binary_signal(1000, unsafe_shift=-2.0, noise=1.0, seed=2)
    ds = Dataset.from_dict(
        {
            "in_distribution": [True] * unsafe.size,
            "unsafe": unsafe.tolist(),
        }
    )

    _, meta = lambda_aggregate(
        [(anchor, {}), (anchor.copy(), {})],
        id_dataset=ds,
        fpr_threshold=_FPR,
    )

    # Default ``lambda_max_exp=0.0`` -> grid max is ``10**0 == 1.0``.
    assert meta["lambda_0"] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_lambda_aggregate_recovers_interior_coefficient_for_weakly_aligned_aux(
    seed: int,
) -> None:
    n = 20000
    alpha, beta = 2.0, 0.4
    rng = np.random.default_rng(seed)
    unsafe = np.zeros(n, dtype=bool)
    unsafe[: n // 2] = True
    rng.shuffle(unsafe)
    anchor = np.where(unsafe, -alpha, 0.0) + rng.normal(0.0, 1.0, n)
    aux = np.where(unsafe, -beta, 0.0) + rng.normal(0.0, 1.0, n)
    id_labels = np.ones(n, dtype=bool)

    _, meta = lambda_aggregate(
        [(anchor, {}), (aux, {})],
        id_labels=id_labels,
        unsafe_labels=unsafe,
        fpr_threshold=_FPR,
    )

    assert 0.2 <= meta["lambda_0"] <= 0.6
