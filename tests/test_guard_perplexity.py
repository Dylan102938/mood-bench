from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pytest
import torch as t
from conftest import get_metric, load_analysis
from huggingface_hub import hf_hub_download
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from mood_bench.aggregator import LambdaAggregate
from mood_bench.core import mood_bench
from mood_bench.pipeline.base import Pipeline, PipelineResult
from mood_bench.pipeline.guard import GuardModelPipeline
from mood_bench.pipeline.perplexity import PerplexityPipeline
from mood_bench.tokenize import load_tokenizer

GUARD_ADAPTER_ID = "shizwick/google-gemma-2-9b_guard"
PPL_ADAPTER_ID = "shizwick/google-gemma-2-9b_causal-lm"

EXPECTED_ID_TPR = 91.1
EXPECTED_OVERALL_TPR = 43.1
TOLERANCE = 0.5


def _infer_num_labels(adapter_id: str) -> int | None:
    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])
    return None


def _free_cuda() -> None:
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def _make_guard_pipeline() -> Pipeline:
    """Lazy-loading guard pipeline that frees weights after inference."""
    num_labels = _infer_num_labels(GUARD_ADAPTER_ID)

    def run(samples: list[str], **kwargs: Any) -> PipelineResult:
        tokenizer = load_tokenizer(GUARD_ADAPTER_ID)
        load_kwargs: dict[str, object] = {"dtype": t.bfloat16}
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            tokenizer.name_or_path, **load_kwargs
        )
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, GUARD_ADAPTER_ID).merge_and_unload()
        model = model.to("cuda").eval()
        try:
            return GuardModelPipeline(model, tokenizer, unsafe_label_index=1)(samples, **kwargs)
        finally:
            del model, tokenizer
            _free_cuda()

    run.__name__ = "GuardModelPipeline"
    return run


def _make_ppl_pipeline() -> Pipeline:
    """Lazy-loading perplexity pipeline that frees weights after inference."""

    def run(samples: list[str], **kwargs: Any) -> PipelineResult:
        peft_config = PeftConfig.from_pretrained(PPL_ADAPTER_ID)
        base_model_id = peft_config.base_model_name_or_path
        tokenizer = load_tokenizer(PPL_ADAPTER_ID)
        model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=t.bfloat16)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, PPL_ADAPTER_ID).merge_and_unload()
        model = model.to("cuda").eval()
        try:
            return PerplexityPipeline(model, tokenizer, outlier_z_threshold=3.0)(samples, **kwargs)
        finally:
            del model, tokenizer
            _free_cuda()

    run.__name__ = "PerplexityPipeline"
    return run


@pytest.mark.gpu
def test_guard_perplexity_lambda(gpu: list[int], results_dir: Path) -> None:
    output_path = results_dir / "guard_perplexity"

    guard_fn = _make_guard_pipeline()
    ppl_fn = _make_ppl_pipeline()

    mood_bench(
        pipelines=[guard_fn, ppl_fn],
        aggregator=LambdaAggregate(anchor_index=0, fpr_threshold=0.01),
        eval_batch_size=4,
        max_length=2048,
        output_dir=str(output_path),
        include_figures=False,
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
