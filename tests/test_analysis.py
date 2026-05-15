"""Analysis-only tests that consume results.jsonl outputs from pipeline tests.

These tests re-run mood_bench_analysis on the saved pipeline outputs to validate
metrics for various pipeline combinations (guard, guard+perplexity, guard+mahalanobis,
guard+perplexity+mahalanobis, IT alignment). No GPU required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RESULTS_DIR, get_metric, load_analysis
from datasets import Dataset, load_dataset

from mood_bench.aggregator import LambdaAggregate
from mood_bench.core import mood_bench_analysis

TOLERANCE = 0.5


def _find_results_jsonl(pipeline_dir: Path) -> Path:
    """Find the results.jsonl from the latest run directory under a pipeline output dir."""
    if not pipeline_dir.exists():
        pytest.skip(f"Pipeline output not found: {pipeline_dir}")
    run_dirs = sorted(pipeline_dir.iterdir())
    if not run_dirs:
        pytest.skip(f"No run directories in {pipeline_dir}")
    results_path = run_dirs[-1] / "results.jsonl"
    if not results_path.exists():
        pytest.skip(f"results.jsonl not found in {run_dirs[-1]}")
    return results_path


def _load_scored_dataset(pipeline_dir: Path) -> Dataset:
    results_path = _find_results_jsonl(pipeline_dir)
    ds = load_dataset("json", data_files=str(results_path), split="train")
    if "malign" not in ds.column_names and "safe" in ds.column_names:
        ds = ds.map(lambda ex: {"malign": int(not bool(ex["safe"]))})
    return ds


def _run_analysis_and_save(
    results: Dataset | list[Dataset],
    output_name: str,
    aggregator=None,
    predict_safe: bool = False,
) -> dict:
    output_path = RESULTS_DIR / output_name
    mood_bench_analysis(
        results=results,
        aggregator=aggregator,
        output_path=output_path,
        include_figures=False,
        predict_safe=predict_safe,
    )
    return load_analysis(output_path)


class TestAnalysisGuardOnly:
    def test_guard_analysis(self, results_dir: Path) -> None:
        ds = _load_scored_dataset(results_dir / "guard")
        analysis = _run_analysis_and_save(ds, "analysis_guard")

        id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
        overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100

        assert id_tpr == pytest.approx(90.8, abs=TOLERANCE)
        assert overall_tpr == pytest.approx(38.8, abs=TOLERANCE)


class TestAnalysisITAlignment:
    def test_it_analysis(self, results_dir: Path) -> None:
        ds = _load_scored_dataset(results_dir / "it_vllm")
        analysis = _run_analysis_and_save(ds, "analysis_it")

        id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
        overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100

        assert id_tpr == pytest.approx(50.1, abs=TOLERANCE)
        assert overall_tpr == pytest.approx(18.2, abs=TOLERANCE)


class TestAnalysisGuardPerplexity:
    def test_guard_perplexity_analysis(self, results_dir: Path) -> None:
        guard_ds = _load_scored_dataset(results_dir / "guard")
        ppl_ds = _load_scored_dataset(results_dir / "perplexity")

        analysis = _run_analysis_and_save(
            [guard_ds, ppl_ds],
            "analysis_guard_perplexity",
            aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
        )

        id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
        overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100

        assert id_tpr == pytest.approx(90.9, abs=TOLERANCE)
        assert overall_tpr == pytest.approx(41.9, abs=TOLERANCE)


class TestAnalysisGuardMahalanobis:
    def test_guard_mahalanobis_analysis(self, results_dir: Path) -> None:
        guard_ds = _load_scored_dataset(results_dir / "guard")
        mahal_ds = _load_scored_dataset(results_dir / "mahalanobis")

        analysis = _run_analysis_and_save(
            [guard_ds, mahal_ds],
            "analysis_guard_mahalanobis",
            aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
        )

        id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
        overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100

        assert id_tpr == pytest.approx(91.2, abs=TOLERANCE)
        assert overall_tpr == pytest.approx(44.8, abs=TOLERANCE)


class TestAnalysisGuardPerplexityMahalanobis:
    def test_guard_perplexity_mahalanobis_analysis(self, results_dir: Path) -> None:
        guard_ds = _load_scored_dataset(results_dir / "guard")
        ppl_ds = _load_scored_dataset(results_dir / "perplexity")
        mahal_ds = _load_scored_dataset(results_dir / "mahalanobis")

        analysis = _run_analysis_and_save(
            [guard_ds, ppl_ds, mahal_ds],
            "analysis_guard_perplexity_mahalanobis",
            aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
        )

        id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
        overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100

        assert id_tpr == pytest.approx(91.1, abs=TOLERANCE)
        assert overall_tpr == pytest.approx(42.7, abs=TOLERANCE)
