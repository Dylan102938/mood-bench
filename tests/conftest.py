from __future__ import annotations

from typing import Any

import pytest
import torch as t
from transformers import BatchEncoding


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--real-models",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.real_model (download and load real HF weights).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_model: tests that load real HF weights; skipped unless --real-models is passed",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--real-models"):
        return
    skip_marker = pytest.mark.skip(reason="real-model test skipped (pass --real-models to enable)")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip_marker)


class _FakeOutput:
    """Duck-typed stand-in for HF model outputs."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTokenizer:
    def __init__(self, vocab_size: int = 32, seq_len: int = 6) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __call__(self, samples: list[str], **kwargs: Any) -> BatchEncoding:
        batch = len(samples)
        seq_len = self.seq_len

        input_ids = (
            t.arange(batch * seq_len, dtype=t.long).reshape(batch, seq_len) % self.vocab_size
        )
        attention_mask = t.ones(batch, seq_len, dtype=t.long)

        data: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if kwargs.get("return_offsets_mapping", False):
            positions = t.arange(seq_len, dtype=t.long)
            per_row = t.stack([positions, positions + 1], dim=-1)
            data["offset_mapping"] = per_row.unsqueeze(0).expand(batch, -1, -1).contiguous()

        return BatchEncoding(data=data)


class FakeLM(t.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.device = t.device("cpu")

    def forward(
        self,
        input_ids: t.Tensor,
        attention_mask: t.Tensor | None = None,
        labels: t.Tensor | None = None,
        **_: Any,
    ) -> _FakeOutput:
        batch, seq_len = input_ids.shape
        logits = t.zeros(batch, seq_len, self.vocab_size)
        for j in range(seq_len):
            logits[:, j, j % self.vocab_size] = 5.0
        return _FakeOutput(logits=logits)


class FakeEncoder(t.nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = t.device("cpu")

    def forward(
        self,
        input_ids: t.Tensor,
        attention_mask: t.Tensor | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> _FakeOutput:
        batch, seq_len = input_ids.shape
        hidden = t.ones(batch, seq_len, self.hidden_size, dtype=t.float32)
        logits = t.zeros(batch, seq_len, self.vocab_size)
        return _FakeOutput(hidden_states=(hidden,), logits=logits)


class FakeClassifier(t.nn.Module):
    def __init__(self, num_labels: int = 2) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.device = t.device("cpu")

    def forward(
        self,
        input_ids: t.Tensor,
        attention_mask: t.Tensor | None = None,
        **_: Any,
    ) -> _FakeOutput:
        batch = input_ids.shape[0]
        if self.num_labels == 1:
            logits = t.arange(batch, dtype=t.float32).unsqueeze(-1)
        else:
            logits = t.zeros(batch, self.num_labels, dtype=t.float32)
            if batch > 1:
                logits[:, 1] = t.linspace(-2.0, 2.0, batch)
        return _FakeOutput(logits=logits)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_lm() -> FakeLM:
    return FakeLM()


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    return FakeClassifier()


@pytest.fixture
def stub_samples() -> list[str]:
    return ["hello world", "foo bar baz", "quick brown fox"]


@pytest.fixture(scope="session")
def real_gpt2_tokenizer() -> Any:
    from mood_bench.tokenize import load_tokenizer

    return load_tokenizer("gpt2")


@pytest.fixture(scope="session")
def real_gpt2_model() -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=t.float32)
    model.eval()
    return model


@pytest.fixture(scope="session")
def real_gpt2_encoder() -> Any:
    from transformers import AutoModel

    model = AutoModel.from_pretrained("gpt2", torch_dtype=t.float32)
    model.eval()
    return model


@pytest.fixture(scope="session")
def real_gpt2_classifier(real_gpt2_tokenizer: Any) -> Any:
    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained("gpt2", num_labels=2)
    config.pad_token_id = real_gpt2_tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        "gpt2", config=config, ignore_mismatched_sizes=True
    )
    model.eval()
    return model
