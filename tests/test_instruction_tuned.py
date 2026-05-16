from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch as t
from conftest import get_metric, load_analysis

from mood_bench.core import mood_bench
from mood_bench.data import DEFAULT_IN_DISTR_DOMAINS, EvalDataset
from mood_bench.pipeline.instruction_tuned import InstructionTunedPipeline

ADAPTER_ID = "shizwick/google-gemma-2-9b_ultrachat-it"
EXPECTED_ID_TPR = 50.1
EXPECTED_OVERALL_TPR = 18.2
TOLERANCE = 0.5
HF_DOMAINS = list(DEFAULT_IN_DISTR_DOMAINS) + [EvalDataset.FUNCTION_CALLING_MISSING]


def _make_pipeline(
    *,
    force_hf: bool = False,
    grading_type: Literal["alignment", "uncertainty"] = "alignment",
) -> InstructionTunedPipeline:
    pipeline = InstructionTunedPipeline(
        ADAPTER_ID,
        is_lora_adapter=True,
        grading_type=grading_type,
        max_new_tokens=256,
        temperature=1.0,
        max_retries=3,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        torch_dtype=t.bfloat16,
        enforce_eager=True,
    )

    if force_hf:
        pipeline._load_hf()
        pipeline._loaded = True

    return pipeline


@pytest.mark.gpu
def test_instruction_tuned_vllm(gpu: list[int], results_dir: Path) -> None:
    output_path = results_dir / "it_vllm"
    pipeline = _make_pipeline(force_hf=False)

    mood_bench(
        pipelines=pipeline,
        eval_batch_size=4,
        max_length=2048,
        output_dir=str(output_path),
        include_figures=False,
        predict_safe=True,
    )

    run_dirs = sorted(output_path.iterdir())
    assert run_dirs, "No output directory created"
    analysis = load_analysis(run_dirs[-1])

    id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
    overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100

    assert id_tpr == pytest.approx(
        EXPECTED_ID_TPR, abs=TOLERANCE
    ), f"ID tpr@fpr0.01: {id_tpr:.1f}% (expected {EXPECTED_ID_TPR}%)"
    assert overall_tpr == pytest.approx(
        EXPECTED_OVERALL_TPR, abs=TOLERANCE
    ), f"Overall tpr@fpr0.01: {overall_tpr:.1f}% (expected {EXPECTED_OVERALL_TPR}%)"


@pytest.mark.gpu
def test_instruction_tuned_hf(gpu: list[int], results_dir: Path) -> None:
    output_path = results_dir / "it_hf"
    pipeline = _make_pipeline(force_hf=True)

    mood_bench(
        pipelines=pipeline,
        domains=HF_DOMAINS,
        use_mini=True,
        eval_batch_size=4,
        max_length=2048,
        output_dir=str(output_path),
        include_figures=False,
        predict_safe=True,
    )

    run_dirs = sorted(output_path.iterdir())
    assert run_dirs, "No output directory created"
    analysis = load_analysis(run_dirs[-1])

    id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
    assert id_tpr > 0, f"ID tpr@fpr0.01 should be > 0, got {id_tpr:.1f}%"


@pytest.mark.gpu
def test_instruction_tuned_uncertainty_vllm(gpu: list[int], results_dir: Path) -> None:
    output_path = results_dir / "it_uncertainty"
    pipeline = _make_pipeline(force_hf=False, grading_type="uncertainty")

    mood_bench(
        pipelines=pipeline,
        eval_batch_size=4,
        max_length=2048,
        output_dir=str(output_path),
        include_figures=False,
        predict_safe=True,
    )

    run_dirs = sorted(output_path.iterdir())
    assert run_dirs, "No output directory created"
    analysis = load_analysis(run_dirs[-1])

    id_tpr = get_metric(analysis, "id", "tpr@fpr0.01") * 100
    overall_tpr = get_metric(analysis, "overall", "tpr@fpr0.01") * 100
    print(
        f"IT uncertainty: id tpr@fpr0.01 = {id_tpr:.1f}%, overall tpr@fpr0.01 = {overall_tpr:.1f}%"
    )
    assert id_tpr > 0, f"ID tpr@fpr0.01 should be > 0, got {id_tpr:.1f}%"
    assert overall_tpr > 0, f"Overall tpr@fpr0.01 should be > 0, got {overall_tpr:.1f}%"
