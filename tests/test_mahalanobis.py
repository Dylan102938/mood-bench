from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import torch as t
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification

from mood_bench.core import mood_bench
from mood_bench.pipeline.mahalanobis import MahalanobisPipeline, get_stats_for_model
from mood_bench.tokenize import load_tokenizer

ADAPTER_ID = "shizwick/google-gemma-2-9b_guard"
POOLING = "cls"


def _infer_num_labels(adapter_id: str) -> int | None:
    path = hf_hub_download(repo_id=adapter_id, filename="adapter_model.safetensors")
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith("score.weight") or key.endswith("classifier.weight"):
                return int(f.get_tensor(key).shape[0])
    return None


@pytest.mark.gpu
def test_mahalanobis_pipeline(gpu: list[int], results_dir: Path) -> None:
    output_path = results_dir / "mahalanobis"
    stats_cache_dir = tempfile.mkdtemp(prefix="mood_mahal_stats_")

    try:
        num_labels = _infer_num_labels(ADAPTER_ID)
        tokenizer = load_tokenizer(ADAPTER_ID)

        load_kwargs: dict[str, object] = {"dtype": t.bfloat16}
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels

        model = AutoModelForSequenceClassification.from_pretrained(
            tokenizer.name_or_path, **load_kwargs
        )
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, ADAPTER_ID).merge_and_unload()
        model = model.to("cuda").eval()

        stats = get_stats_for_model(model, tokenizer, pooling_strategy=POOLING, batch_size=4)

        cache_path = Path(stats_cache_dir) / "stats.pt"
        t.save({k: v.detach().cpu() for k, v in stats.items()}, cache_path)

        mean = stats["mean"].to(device="cuda", dtype=t.float64)
        inv_cov = stats["inv_cov"].to(device="cuda", dtype=t.float64)

        pipeline = MahalanobisPipeline(
            model, tokenizer, mean=mean, inv_cov=inv_cov, pooling_strategy=POOLING
        )
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
    finally:
        shutil.rmtree(stats_cache_dir, ignore_errors=True)
