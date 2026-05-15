from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM

from mood_bench.core import mood_bench
from mood_bench.pipeline.perplexity import PerplexityPipeline
from mood_bench.tokenize import load_tokenizer

ADAPTER_ID = "shizwick/google-gemma-2-9b_causal-lm"


@pytest.mark.gpu
def test_perplexity_pipeline(gpu: list[int], results_dir: Path) -> None:
    output_path = results_dir / "perplexity"

    peft_config = PeftConfig.from_pretrained(ADAPTER_ID)
    base_model_id = peft_config.base_model_name_or_path

    tokenizer = load_tokenizer(ADAPTER_ID)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=t.bfloat16)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, ADAPTER_ID).merge_and_unload()
    model = model.to("cuda").eval()

    pipeline = PerplexityPipeline(model, tokenizer, outlier_z_threshold=3.0)
    mood_bench(
        pipelines=pipeline,
        eval_batch_size=4,
        max_length=2048,
        output_dir=str(output_path),
        include_figures=False,
    )

    run_dirs = sorted(output_path.iterdir())
    assert run_dirs, "No output directory created"
    assert (run_dirs[-1] / "results.jsonl").exists()
    assert (run_dirs[-1] / "analysis.json").exists()
