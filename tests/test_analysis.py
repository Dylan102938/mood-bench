"""Analysis-only tests that consume frozen ``results.jsonl`` fixtures.

These tests re-run mood_bench_analysis on the saved pipeline outputs to validate
metrics for various pipeline combinations (guard, guard+perplexity, guard+mahalanobis,
guard+perplexity+mahalanobis, IT alignment, IT alignment+IT uncertainty,
guard+IT uncertainty). No GPU required.
"""

from __future__ import annotations

from conftest import assert_tpr_metrics, get_metric
from datasets import Dataset

from mood_bench.aggregator import LambdaAggregate
from mood_bench.core import mood_bench_analysis

TOLERANCE = 2.0


def _run_analysis(
    results: Dataset | list[Dataset],
    aggregator=None,
    predict_safe: bool | list[bool] = False,
) -> dict:
    _, report = mood_bench_analysis(
        results=results,
        aggregator=aggregator,
        output_path=None,
        include_figures=False,
        predict_safe=predict_safe,
    )
    return report


class TestAnalysisGuardOnly:
    def test_guard_analysis(self, guard_dataset: Dataset) -> None:
        analysis = _run_analysis(guard_dataset)

        assert_tpr_metrics(
            analysis,
            {
                "id": 90.8,
                "controlling": 82.7,
                "function-calling-inappropriate": 2.8,
                "function-calling-missing": 0.6,
                "insecure-code": 0.0,
                "jailbroken": 46.3,
                "scheming": 37.1,
                "sycophantic": 49.9,
                "overall": 38.8,
            },
            tolerance=TOLERANCE,
        )


class TestAnalysisGuardPerplexity:
    def test_guard_perplexity_analysis(
        self,
        guard_dataset: Dataset,
        perplexity_dataset: Dataset,
    ) -> None:
        analysis = _run_analysis(
            [guard_dataset, perplexity_dataset],
            aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
        )

        assert_tpr_metrics(
            analysis,
            {
                "id": 91.1,
                "controlling": 88.1,
                "function-calling-inappropriate": 0.1,
                "function-calling-missing": 0.1,
                "insecure-code": 0.1,
                "jailbroken": 59.9,
                "scheming": 45.0,
                "sycophantic": 57.5,
                "overall": 43.1,
            },
            tolerance=TOLERANCE,
        )


class TestAnalysisGuardMahalanobis:
    def test_guard_mahalanobis_analysis(
        self,
        guard_dataset: Dataset,
        mahalanobis_dataset: Dataset,
    ) -> None:
        analysis = _run_analysis(
            [guard_dataset, mahalanobis_dataset],
            aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
        )

        assert_tpr_metrics(
            analysis,
            {
                "id": 91.3,
                "controlling": 87.9,
                "function-calling-inappropriate": 8.6,
                "function-calling-missing": 2.6,
                "insecure-code": 1.8,
                "jailbroken": 64.7,
                "scheming": 48.0,
                "sycophantic": 53.8,
                "overall": 44.8,
            },
            tolerance=TOLERANCE,
        )


class TestAnalysisGuardPerplexityMahalanobis:
    def test_guard_perplexity_mahalanobis_analysis(
        self,
        guard_dataset: Dataset,
        perplexity_dataset: Dataset,
        mahalanobis_dataset: Dataset,
    ) -> None:
        analysis = _run_analysis(
            [guard_dataset, perplexity_dataset, mahalanobis_dataset],
            aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
        )

        assert_tpr_metrics(
            analysis,
            {
                "id": 91.2,
                "controlling": 89.9,
                "function-calling-inappropriate": 2.3,
                "function-calling-missing": 0.6,
                "insecure-code": 0.6,
                "jailbroken": 70.1,
                "scheming": 51.5,
                "sycophantic": 60.0,
                "overall": 46.5,
            },
            tolerance=TOLERANCE,
        )


class TestAnalysisITAlignment:
    def test_it_analysis(self, it_vllm_dataset: Dataset) -> None:
        analysis = _run_analysis(it_vllm_dataset, predict_safe=True)

        # Per-domain checks are omitted; the IT fixture is known to be stale and
        # only id/overall expected values are tracked for now.
        assert_tpr_metrics(
            analysis,
            {
                "id": 50.1,
                "overall": 18.2,
            },
            tolerance=TOLERANCE,
        )


class TestAnalysisITAlignmentUncertainty:
    def test_it_alignment_uncertainty_analysis(
        self,
        it_vllm_dataset: Dataset,
        it_uncertainty_dataset: Dataset,
    ) -> None:
        analysis = _run_analysis(
            [it_vllm_dataset, it_uncertainty_dataset],
            aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
            predict_safe=True,
        )

        # TODO: fill in expected per-domain values once a baseline is recorded.
        id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
        overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100
        assert id_tpr > 0, f"ID tpr@fpr0.01 should be > 0, got {id_tpr:.2f}%"
        assert overall_tpr > 0, f"Overall tpr@fpr0.01 should be > 0, got {overall_tpr:.2f}%"


class TestAnalysisGuardITUncertainty:
    def test_guard_it_uncertainty_analysis(
        self,
        guard_dataset: Dataset,
        it_uncertainty_dataset: Dataset,
    ) -> None:
        analysis = _run_analysis(
            [guard_dataset, it_uncertainty_dataset],
            aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
            predict_safe=[False, True],
        )

        # TODO: fill in expected per-domain values once a baseline is recorded.
        id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
        overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100
        assert id_tpr > 0, f"ID tpr@fpr0.01 should be > 0, got {id_tpr:.2f}%"
        assert overall_tpr > 0, f"Overall tpr@fpr0.01 should be > 0, got {overall_tpr:.2f}%"
