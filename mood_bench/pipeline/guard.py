from typing import Any

import numpy as np
import torch as t
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

from mood_bench.pipeline.base import Pipeline, PipelineResult
from mood_bench.tokenize import rendered


class GuardModelPipeline(Pipeline):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        unsafe_label_index: int = 1,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.unsafe_label_index = unsafe_label_index

    def __call__(self, samples: list[str], **kwargs: Any) -> PipelineResult:
        scores_list: list[np.ndarray] = []
        batch_size = kwargs.get("batch_size", 1)
        for batch in rendered(
            samples,
            renderer=self.tokenizer,
            device=self.model.device,
            batch_size=batch_size,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ):
            scores_list.append(self._batch_inference(batch))

        scores = np.concatenate(scores_list)
        return scores, {}

    def _batch_inference(self, enc: BatchEncoding) -> np.ndarray:
        with t.inference_mode():
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )

        logits = outputs.logits.detach().float().cpu()
        if logits.shape[-1] > 1:
            return t.softmax(logits, dim=-1)[:, self.unsafe_label_index].numpy()
        else:
            return logits.squeeze(-1).numpy()
