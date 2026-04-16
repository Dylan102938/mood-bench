"""Aggregate per-input scores from multiple :class:`~mood_bench.pipeline.base.Pipeline` runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import numpy as np
from datasets import Dataset

from mood_bench.metrics import tpr_at_fpr
from mood_bench.pipeline.base import PipelineResult


def _merge_meta(metas: Iterable[dict[str, Any]]) -> dict[str, Any]:
    meta: dict[str, Any] = {}

    for m in metas:
        for key, val in m.items():
            if key not in meta:
                meta[key] = val

    return meta


class Aggregator(Protocol):
    def __call__(self, results: Iterable[PipelineResult], **kwargs: Any) -> PipelineResult: ...


def min_aggregate(results: Iterable[PipelineResult]) -> PipelineResult:
    rs = list(results)
    if not rs:
        raise ValueError("Need at least one PipelineResult to aggregate.")

    arrays = [np.asarray(r[0]) for r in rs]
    stacked = np.stack(arrays, axis=0)

    return (np.min(stacked, axis=0), _merge_meta([r[1] for r in rs]))


def mean_aggregate(results: Iterable[PipelineResult]) -> PipelineResult:
    rs = list(results)
    if not rs:
        raise ValueError("Need at least one PipelineResult to aggregate.")

    arrays = [np.asarray(r[0]) for r in rs]
    stacked = np.stack(arrays, axis=0)

    return (np.mean(stacked, axis=0), _merge_meta([r[1] for r in rs]))


def _standardize(arrays: np.ndarray, mask: np.ndarray) -> np.ndarray:
    operating_arrays = arrays[..., mask]
    mu = operating_arrays.mean(axis=-1, keepdims=True)
    sigma = np.maximum(1e-6, operating_arrays.std(axis=-1, keepdims=True))

    return (arrays - mu) / sigma


def lambda_aggregate(
    results: Iterable[PipelineResult],
    *,
    anchor_index: int = 0,
    id_dataset: Dataset | None = None,
    id_labels: np.ndarray | Sequence[int] | None = None,
    unsafe_labels: np.ndarray | Sequence[int] | None = None,
    lambda_min_exp: float = -2.0,
    lambda_max_exp: float = 0.0,
    fpr_threshold: float = 0.01,
    n_lambdas: int = 21,
    n_passes: int = 1,
) -> PipelineResult:
    """Aggregate pipeline results via ``anchor + c_1*aux_1 + c_2*aux_2 + ...``.

    Coefficients are fitted by coordinate descent on a log-spaced grid,
    choosing the largest coefficient that does not degrade TPR (at the given
    FPR threshold) on in-distribution unsafe samples relative to the
    anchor-only baseline.
    """

    ### Input validation ###
    rs = list(results)
    if len(rs) < 2:
        raise ValueError("lambda_aggregate requires at least 2 PipelineResults.")
    if not (0 <= anchor_index < len(rs)):
        raise ValueError(f"anchor_index {anchor_index} out of range for {len(rs)} results.")

    ### Resolve labels ###
    if id_dataset is not None:
        id_labels_arr = np.asarray(id_dataset["in_distribution"], dtype=bool)
        unsafe_labels_arr = np.asarray(id_dataset["unsafe"], dtype=bool)
    elif id_labels is not None and unsafe_labels is not None:
        id_labels_arr = np.asarray(id_labels, dtype=bool)
        unsafe_labels_arr = np.asarray(unsafe_labels, dtype=bool)
    else:
        raise ValueError("Provide either id_dataset or both id_labels and unsafe_labels.")

    ### Define scores ###
    raw_arrays = [np.asarray(r[0], dtype=float) for r in rs]
    unsafe_labels_id = unsafe_labels_arr[id_labels_arr]

    anchor = raw_arrays[anchor_index]
    anchor_id = anchor[id_labels_arr]
    anchor_id = _standardize(anchor_id, ~unsafe_labels_id)

    auxiliaries = np.array([a for i, a in enumerate(raw_arrays) if i != anchor_index])
    auxiliaries_id = auxiliaries[:, id_labels_arr]
    auxiliaries_id = _standardize(auxiliaries_id, ~unsafe_labels_id)

    coeffs = np.zeros(len(auxiliaries))
    grid = np.concatenate([[0.0], 10.0 ** np.linspace(lambda_min_exp, lambda_max_exp, n_lambdas)])

    ### Coordinate descent ###
    n_passes = 1 if len(auxiliaries_id) < 2 else n_passes
    for _ in range(n_passes):
        for i, aux in enumerate(auxiliaries_id):
            baseline_scores = anchor_id + np.sum(
                np.delete(coeffs, i)[:, None] * np.delete(auxiliaries_id, i, axis=0),
                axis=0,
            )
            baseline_tpr = tpr_at_fpr(
                baseline_scores,
                unsafe_labels_id,
                fpr_threshold,
            )

            candidates = baseline_scores + grid[:, None] * aux
            tprs = tpr_at_fpr(
                candidates,
                unsafe_labels_id,
                fpr_threshold,
            )

            valid = np.where(tprs >= baseline_tpr)[0]
            coeffs[i] = grid[valid[-1]] if len(valid) > 0 else 0.0

    ### Return final results ###
    id_safe_mask = id_labels_arr & ~unsafe_labels_arr
    anchor = _standardize(anchor, id_safe_mask)
    auxiliaries = _standardize(auxiliaries, id_safe_mask)
    final_scores = anchor + np.sum(coeffs[:, None] * auxiliaries, axis=0)
    meta = {f"lambda_{i}": coeffs[i] for i in range(len(coeffs))}
    meta = {
        **_merge_meta([r[1] for r in rs]),
        **meta,
    }

    return final_scores, meta
