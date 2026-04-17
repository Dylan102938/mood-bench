"""Pipeline tests that run on stubbed HF models and tokenizers."""

from __future__ import annotations

import numpy as np
import torch as t
from conftest import FakeClassifier, FakeEncoder, FakeLM, FakeTokenizer

from mood_bench.pipeline.guard import GuardModelPipeline
from mood_bench.pipeline.mahalanobis import MahalanobisPipeline, _pool_hidden_states
from mood_bench.pipeline.perplexity import PerplexityPipeline


def test_perplexity_output_shape_and_positive(
    fake_tokenizer: FakeTokenizer,
    fake_lm: FakeLM,
    stub_samples: list[str],
) -> None:
    pipeline = PerplexityPipeline(
        model=fake_lm,
        tokenizer=fake_tokenizer,
        outlier_z_threshold=3.0,
    )

    scores, meta = pipeline(stub_samples, batch_size=2)

    assert scores.shape == (len(stub_samples),)
    assert np.all(scores > 0)
    assert np.all(np.isfinite(scores))
    assert "high_perplexity" in meta
    assert len(meta["high_perplexity"]) == len(stub_samples)


def test_perplexity_none_threshold_returns_every_valid_token(
    fake_tokenizer: FakeTokenizer, stub_samples: list[str]
) -> None:
    pipeline = PerplexityPipeline(
        model=FakeLM(), tokenizer=fake_tokenizer, outlier_z_threshold=None
    )

    _, meta = pipeline(stub_samples, batch_size=2)

    expected_valid = fake_tokenizer.seq_len - 1
    for outlier_list in meta["high_perplexity"]:
        assert len(outlier_list) == expected_valid
        for entry in outlier_list:
            assert {"char", "token", "nll", "z_score"} <= entry.keys()


def test_pool_hidden_states_cls_takes_last_token() -> None:
    hidden = t.arange(2 * 3 * 4, dtype=t.float32).reshape(2, 3, 4)
    mask = t.ones(2, 3, dtype=t.long)

    pooled = _pool_hidden_states(hidden, mask, "cls")

    assert t.equal(pooled, hidden[:, -1, :])


def test_pool_hidden_states_mean_ignores_masked_positions() -> None:
    hidden = t.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0], [99.0, 99.0]],
            [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
        ]
    )
    mask = t.tensor([[1, 1, 0], [1, 1, 1]], dtype=t.long)

    pooled = _pool_hidden_states(hidden, mask, "mean")

    assert t.allclose(pooled, t.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_pool_hidden_states_max_picks_largest() -> None:
    hidden = t.tensor([[[1.0, 2.0], [3.0, 1.0], [0.0, 5.0]]])
    mask = t.ones(1, 3, dtype=t.long)

    pooled = _pool_hidden_states(hidden, mask, "max")

    assert t.equal(pooled, t.tensor([[3.0, 5.0]]))


def test_mahalanobis_identity_returns_pooled_norm(
    fake_tokenizer: FakeTokenizer, fake_encoder: FakeEncoder, stub_samples: list[str]
) -> None:
    hidden = fake_encoder.hidden_size
    mean = t.zeros(hidden, dtype=t.float64)
    inv_cov = t.eye(hidden, dtype=t.float64)
    pipeline = MahalanobisPipeline(
        fake_encoder,
        fake_tokenizer,
        mean=mean,
        inv_cov=inv_cov,
        pooling_strategy="mean",
    )

    scores, _ = pipeline(stub_samples, batch_size=2)

    assert scores.shape == (len(stub_samples),)
    assert np.all(scores >= 0)
    assert np.allclose(scores, np.sqrt(hidden))


def test_guard_two_logits_returns_softmax_probabilities(
    fake_tokenizer: FakeTokenizer, stub_samples: list[str]
) -> None:
    pipeline = GuardModelPipeline(
        FakeClassifier(num_labels=2), fake_tokenizer, unsafe_label_index=1
    )

    scores, _ = pipeline(stub_samples, batch_size=2)

    assert scores.shape == (len(stub_samples),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_guard_single_logit_passes_through(
    fake_tokenizer: FakeTokenizer, stub_samples: list[str]
) -> None:
    pipeline = GuardModelPipeline(FakeClassifier(num_labels=1), fake_tokenizer)

    scores, _ = pipeline(stub_samples, batch_size=2)

    assert scores.shape == (len(stub_samples),)
    assert np.all(np.isfinite(scores))


def test_guard_batching_preserves_sample_order_and_length(
    fake_tokenizer: FakeTokenizer,
) -> None:
    samples = [f"sample-{i}" for i in range(5)]
    pipeline = GuardModelPipeline(FakeClassifier(num_labels=2), fake_tokenizer)

    scores, _ = pipeline(samples, batch_size=2)

    assert scores.shape == (len(samples),)
