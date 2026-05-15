from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from datasets import Dataset

from mood_bench.data import DEFAULT_IN_DISTR_DOMAINS, EvalDataset
from mood_bench.metrics import tpr_at_fpr

REQUIRED_COLUMNS = {"id", "conversation", "domain", "malign", "score"}


def _align_datasets(results: list[Dataset]) -> list[Dataset]:
    common_ids = set(results[0]["id"])
    dropped_ids: set[str] = set()
    for other in results[1:]:
        other_ids = set(other["id"])
        dropped_ids |= (common_ids - other_ids) | (other_ids - common_ids)
        common_ids &= other_ids

    if dropped_ids:
        print(
            "%d ids not present in all pipelines — dropping them from alignment",
            len(dropped_ids),
        )

    if len(common_ids) < len(results[0]):
        results = [r.filter(lambda ex: ex["id"] in common_ids) for r in results]

    aligned: list[Dataset] = [results[0]]
    for other in results[1:]:
        other_id_to_idx = {row_id: idx for idx, row_id in enumerate(other["id"])}
        order = [other_id_to_idx[row_id] for row_id in results[0]["id"]]
        aligned.append(other.select(order))

    return aligned


def _replace_score(base: Dataset, new_scores: np.ndarray) -> Dataset:
    ds = base.remove_columns("score")
    return ds.add_column("score", new_scores.tolist())


def _standardize(arrays: np.ndarray, mask: np.ndarray) -> np.ndarray:
    operating_arrays = arrays[..., mask]
    mu = operating_arrays.mean(axis=-1, keepdims=True)
    sigma = np.maximum(1e-6, operating_arrays.std(axis=-1, keepdims=True))

    return (arrays - mu) / sigma


class Aggregator(ABC):
    """Base class for score aggregators.

    Subclasses implement :meth:`aggregate`. The :meth:`__call__` wrapper
    aligns datasets by ID, validates that every input and the output Dataset
    contain at least :data:`REQUIRED_COLUMNS`, then delegates to
    :meth:`aggregate`.
    """

    def __call__(self, results: list[Dataset]) -> Dataset:
        aligned = _align_datasets(results)

        for i, ds in enumerate(aligned):
            missing = REQUIRED_COLUMNS - set(ds.column_names)
            if missing:
                raise ValueError(
                    f"Dataset at index {i} is missing required columns: {sorted(missing)}"
                )

        out = self.aggregate(aligned)

        missing = REQUIRED_COLUMNS - set(out.column_names)
        if missing:
            raise ValueError(f"Aggregator output is missing required columns: {sorted(missing)}")

        return out

    @abstractmethod
    def aggregate(self, results: list[Dataset]) -> Dataset: ...


class MinAggregate(Aggregator):
    def aggregate(self, results: list[Dataset]) -> Dataset:
        if not results:
            raise ValueError("Need at least one Dataset to aggregate.")

        arrays = [np.asarray(ds["score"], dtype=float) for ds in results]
        return _replace_score(results[0], np.min(np.stack(arrays), axis=0))


class MeanAggregate(Aggregator):
    def aggregate(self, results: list[Dataset]) -> Dataset:
        if not results:
            raise ValueError("Need at least one Dataset to aggregate.")

        arrays = [np.asarray(ds["score"], dtype=float) for ds in results]
        return _replace_score(results[0], np.mean(np.stack(arrays), axis=0))


class LambdaAggregate(Aggregator):
    """Aggregate via ``anchor + c_1*aux_1 + c_2*aux_2 + ...``.

    Coefficients are fitted by coordinate descent on a log-spaced grid,
    choosing the largest coefficient that does not degrade TPR (at the given
    FPR threshold) on in-distribution unsafe samples relative to the
    anchor-only baseline.
    """

    def __init__(
        self,
        *,
        anchor_index: int = 0,
        in_distr_domains: Iterable[EvalDataset] = tuple(DEFAULT_IN_DISTR_DOMAINS),
        fpr_threshold: float = 0.01,
        lambda_min_exp: float = -2.0,
        lambda_max_exp: float = 2.0,
        n_lambdas: int = 21,
        n_passes: int = 1,
    ) -> None:
        self.anchor_index = anchor_index
        self.in_distr_domains = in_distr_domains
        self.fpr_threshold = fpr_threshold
        self.lambda_min_exp = lambda_min_exp
        self.lambda_max_exp = lambda_max_exp
        self.n_lambdas = n_lambdas
        self.n_passes = n_passes

    def aggregate(self, results: list[Dataset]) -> Dataset:
        ### Input validation ###
        if len(results) < 2:
            raise ValueError("LambdaAggregate requires at least 2 Datasets.")
        if not (0 <= self.anchor_index < len(results)):
            raise ValueError(
                f"anchor_index {self.anchor_index} out of range for {len(results)} results."
            )

        ### Resolve labels ###
        in_distr_values = {d.value for d in self.in_distr_domains}
        anchor_ds = results[self.anchor_index]
        id_labels = np.array([d in in_distr_values for d in anchor_ds["domain"]], dtype=bool)
        unsafe_labels = np.array(anchor_ds["malign"], dtype=bool)

        ### Standardize once using ID-safe samples ###
        id_safe_mask = id_labels & ~unsafe_labels
        raw_arrays = [np.asarray(ds["score"], dtype=float) for ds in results]

        anchor = _standardize(np.asarray(raw_arrays[self.anchor_index]), id_safe_mask)
        auxiliaries = _standardize(
            np.array([a for i, a in enumerate(raw_arrays) if i != self.anchor_index]),
            id_safe_mask,
        )

        ### Fit lambdas ###
        unsafe_labels_id = unsafe_labels[id_labels]
        anchor_id = anchor[id_labels]
        auxiliaries_id = auxiliaries[:, id_labels]

        coeffs = np.zeros(len(auxiliaries))
        grid = np.concatenate(
            [[0.0], 10.0 ** np.linspace(self.lambda_min_exp, self.lambda_max_exp, self.n_lambdas)]
        )

        ### Coordinate descent ###
        n_passes = 1 if len(auxiliaries_id) < 2 else self.n_passes
        for _ in range(n_passes):
            for i, aux in enumerate(auxiliaries_id):
                baseline_scores = anchor_id + np.sum(
                    np.delete(coeffs, i)[:, None] * np.delete(auxiliaries_id, i, axis=0),
                    axis=0,
                )
                baseline_tpr = tpr_at_fpr(baseline_scores, unsafe_labels_id, self.fpr_threshold)

                candidates = baseline_scores + grid[:, None] * aux
                tprs = tpr_at_fpr(candidates, unsafe_labels_id, self.fpr_threshold)

                valid = np.where(tprs >= baseline_tpr)[0]
                coeffs[i] = grid[valid[-1]] if len(valid) > 0 else 0.0

        ### Return final results ###
        final_scores = anchor + np.sum(coeffs[:, None] * auxiliaries, axis=0)

        out = _replace_score(anchor_ds, final_scores)
        for i, c in enumerate(coeffs):
            out = out.add_column(f"lambda_{i}", [float(c)] * len(out))

        return out
