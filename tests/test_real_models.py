from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch as t

from mood_bench.pipeline.guard import GuardModelPipeline
from mood_bench.pipeline.mahalanobis import MahalanobisPipeline, _pool_hidden_states
from mood_bench.pipeline.perplexity import PerplexityPipeline
from mood_bench.tokenize import rendered

pytestmark = pytest.mark.real_model


def test_perplexity_pipeline_real_gpt2(real_gpt2_model: Any, real_gpt2_tokenizer: Any) -> None:
    pipeline = PerplexityPipeline(real_gpt2_model, real_gpt2_tokenizer)
    samples = ["Hello, world!", "The cat sat on the mat."]

    scores, meta = pipeline(samples, batch_size=2)

    assert scores.shape == (len(samples),)
    assert np.all(scores > 0)
    assert np.all(np.isfinite(scores))
    assert "high_perplexity" in meta
    assert len(meta["high_perplexity"]) == len(samples)


def test_mahalanobis_pipeline_real_gpt2(real_gpt2_encoder: Any, real_gpt2_tokenizer: Any) -> None:
    corpus = [
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Good morning, friend.",
        "Today is a sunny day.",
    ]

    feats: list[t.Tensor] = []
    with t.inference_mode():
        for batch in rendered(
            corpus,
            renderer=real_gpt2_tokenizer,
            device=real_gpt2_encoder.device,
            batch_size=2,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ):
            out = real_gpt2_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            pooled = _pool_hidden_states(out.hidden_states[-1], batch["attention_mask"], "cls")
            feats.append(pooled)

    features = t.cat(feats, dim=0).to(t.float64)
    mean = features.mean(dim=0)
    centered = features - mean
    cov = (centered.T @ centered) / max(1, centered.size(0) - 1)
    cov += t.eye(cov.size(0), dtype=t.float64) * 1e-6
    inv_cov = t.linalg.pinv(cov)

    pipeline = MahalanobisPipeline(
        real_gpt2_encoder,
        real_gpt2_tokenizer,
        mean=mean,
        inv_cov=inv_cov,
    )
    scores, _ = pipeline(["A completely different test string."], batch_size=1)

    assert scores.shape == (1,)
    assert scores[0] >= 0
    assert np.isfinite(scores[0])


def test_guard_pipeline_real_gpt2(real_gpt2_classifier: Any, real_gpt2_tokenizer: Any) -> None:
    pipeline = GuardModelPipeline(real_gpt2_classifier, real_gpt2_tokenizer)
    samples = ["hello there", "goodbye friend"]

    scores, _ = pipeline(samples, batch_size=2)

    assert scores.shape == (len(samples),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert np.all(np.isfinite(scores))
