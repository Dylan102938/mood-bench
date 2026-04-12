from typing import Any, Iterator, Literal, Protocol, overload

import torch as t
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase


class Renderer(Protocol):
    def __call__(self, samples: list[str], **kwargs: Any) -> BatchEncoding: ...


@overload
def rendered(
    samples: list[str],
    renderer: Renderer,
    device: t.device | str,
    *,
    batch_size: int = 1,
    desc: str | None = None,
    return_inputs: Literal[False] = False,
    **renderer_kwargs: Any,
) -> Iterator[BatchEncoding]: ...


@overload
def rendered(
    samples: list[str],
    renderer: Renderer,
    device: t.device | str,
    *,
    batch_size: int = 1,
    desc: str | None = None,
    return_inputs: Literal[True],
    **renderer_kwargs: Any,
) -> Iterator[tuple[list[str], BatchEncoding]]: ...


def rendered(
    samples: list[str],
    renderer: Renderer,
    device: t.device | str,
    *,
    batch_size: int = 1,
    desc: str | None = None,
    return_inputs: bool = False,
    **renderer_kwargs: Any,
) -> Iterator[BatchEncoding] | Iterator[tuple[list[str], BatchEncoding]]:
    dataloader = DataLoader(
        Dataset.from_dict({"conversation": samples}),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    for batch in tqdm(dataloader, desc=desc, leave=False):
        conversations: list[str] = batch["conversation"]
        enc = renderer(conversations, **renderer_kwargs).to(device)
        yield (conversations, enc) if return_inputs else enc


def load_tokenizer(
    tokenizer_name: str,
    *,
    hf_token: str | None = None,
    pad_token: str | None = None,
    pad_token_id: int | None = None,
    padding_side: Literal["left", "right"] = "left",
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)

    pad_token = pad_token or tokenizer.eos_token
    pad_token_id = pad_token_id or tokenizer.eos_token_id

    tokenizer.pad_token = pad_token
    tokenizer.pad_token_id = pad_token_id
    tokenizer.padding_side = padding_side

    return tokenizer
