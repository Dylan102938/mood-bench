from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
from conftest import get_metric, load_analysis
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification

from mood_bench.core import mood_bench
from mood_bench.pipeline.guard import GuardModelPipeline
from mood_bench.tokenize import load_tokenizer

MODEL_ID = "shizwick/google-gemma-2-9b_guard"
EXPECTED_ID_TPR = 90.8
EXPECTED_OVERALL_TPR = 38.8
TOLERANCE = 0.5


def _infer_num_labels(adapter_id: str) -> int | None:
    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])
    return None


@pytest.mark.gpu
def test_guard_pipeline(gpu: list[int], results_dir: Path) -> None:
    output_path = results_dir / "guard"

    num_labels = _infer_num_labels(MODEL_ID)
    tokenizer = load_tokenizer(MODEL_ID)

    load_kwargs: dict[str, object] = {"dtype": t.bfloat16}
    if num_labels is not None:
        load_kwargs["num_labels"] = num_labels

    model = AutoModelForSequenceClassification.from_pretrained(
        tokenizer.name_or_path, **load_kwargs
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, MODEL_ID).merge_and_unload()
    model = model.to("cuda").eval()

    pipeline = GuardModelPipeline(model, tokenizer, unsafe_label_index=1)
    mood_bench(
        pipelines=pipeline,
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
