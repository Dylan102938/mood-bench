from enum import Enum
from typing import Literal, Sequence, Set

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class EvalDataset(Enum):
    HH_HELPFUL = "hh-rlhf-helpful"
    HH_HARMLESS = "hh-rlhf-harmless"
    CONTROLLING = "controlling"
    INSECURE_CODE = "insecure-code"
    SCHEMING = "scheming"
    FUNCTION_CALLING = "function-calling"
    FUNCTION_CALLING_MISSING = "function-calling-missing"
    FUNCTION_CALLING_INAPPROPRIATE = "function-calling-inappropriate"
    JAILBROKEN = "jailbroken"
    SYCOPHANTIC = "sycophantic"
    SWAHILI = "swahili"


DEFAULT_MAX_LENGTH_TOKENIZER = "google/gemma-2-2b"
DEFAULT_IN_DISTR_DOMAINS: list[EvalDataset] = [
    EvalDataset.HH_HELPFUL,
    EvalDataset.HH_HARMLESS,
    EvalDataset.FUNCTION_CALLING,
]
ALL_EVALS: Set[EvalDataset] = {
    EvalDataset.HH_HELPFUL,
    EvalDataset.HH_HARMLESS,
    EvalDataset.FUNCTION_CALLING,
    EvalDataset.FUNCTION_CALLING_MISSING,
    EvalDataset.FUNCTION_CALLING_INAPPROPRIATE,
    EvalDataset.CONTROLLING,
    EvalDataset.INSECURE_CODE,
    EvalDataset.SCHEMING,
    EvalDataset.JAILBROKEN,
    EvalDataset.SYCOPHANTIC,
}


def filter_by_max_length(
    ds: Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    conversation_column: str = "conversation",
) -> Dataset:
    return ds.filter(
        lambda ex: len(
            tokenizer.encode(
                ex[conversation_column],
                add_special_tokens=False,
            )
        )
        <= max_length
    )


def load_mood_dataset(
    split: Literal["train", "test"] = "test",
    domains: Sequence[EvalDataset] | None = None,
    max_length: int | None = None,
    max_length_tokenizer: str | None = None,
) -> Dataset:
    ds = load_dataset("shizwick/mood-bench", split=split)

    if domains:
        ds = ds.filter(lambda ex: ex["domain"] in [d.value for d in domains])

    if max_length is not None:
        tok_name = max_length_tokenizer or DEFAULT_MAX_LENGTH_TOKENIZER
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        ds = filter_by_max_length(ds, max_length, tokenizer)

    return ds
