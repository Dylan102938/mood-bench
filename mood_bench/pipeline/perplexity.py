from typing import Any, cast

import numpy as np
import torch as t
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

from mood_bench.pipeline.base import Pipeline
from mood_bench.tokenize import rendered


class PerplexityPipeline(Pipeline):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        outlier_z_threshold: float | None = 3,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.outlier_z_threshold = outlier_z_threshold

    def __call__(self, samples: list[str], **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        perplexities_list: list[np.ndarray] = []
        outliers_list: list[list[dict[str, Any]]] = []
        batch_size = kwargs.get("batch_size", 1)
        for conversations, batch in rendered(
            samples,
            renderer=self.tokenizer,
            device=self.model.device,
            batch_size=batch_size,
            return_inputs=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ):
            ppl, outliers = self._batch_inference(batch, conversations)
            perplexities_list.append(ppl)
            outliers_list.extend(outliers)

        perplexities = np.concatenate(perplexities_list)
        meta = {"high_perplexity": outliers_list}

        return -perplexities, meta

    def _batch_inference(
        self, enc: BatchEncoding, samples: list[str]
    ) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        input_ids = cast(t.Tensor, enc["input_ids"])
        attention_mask = cast(t.Tensor, enc["attention_mask"])
        offset_mapping = enc["offset_mapping"]  # list of (start, end) per token

        labels = input_ids.clone()
        labels = labels.masked_fill(attention_mask == 0, -100)

        with t.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            shift_logits, shift_labels = out.logits[..., :-1, :], labels[..., 1:]
            valid_mask = shift_labels != -100
            safe_labels = shift_labels.masked_fill(~valid_mask, 0)

            token_counts = valid_mask.sum(dim=-1).clamp_min(1)

            logprobs = t.nn.functional.log_softmax(shift_logits, dim=-1)
            label_logprobs = logprobs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
            label_logprobs = label_logprobs.masked_fill(~valid_mask, 0.0)

            nll = -label_logprobs.sum(dim=-1) / token_counts
            ppl = t.exp(nll)

            entropy = -(t.exp(logprobs) * logprobs).sum(dim=-1)  # (batch, seq_len-1)
            excess_surprise = (-label_logprobs - entropy).masked_fill(~valid_mask, 0.0)

            token_nll_np = -label_logprobs.detach().cpu().float().numpy()
            excess_surprise_np = excess_surprise.detach().cpu().float().numpy()
            valid_mask_np = valid_mask.detach().cpu().float().numpy()
            outliers = [
                self._find_outlier_tokens(
                    token_nll_np[i],
                    excess_surprise_np[i],
                    valid_mask_np[i],
                    offset_mapping[i][1:],
                    samples[i],
                )
                for i in range(len(samples))
            ]

            return ppl.detach().cpu().numpy(), outliers

    def _find_outlier_tokens(
        self,
        token_nll: np.ndarray,
        excess_surprise: np.ndarray,
        valid_mask: np.ndarray,
        offset_mapping: list[tuple[int, int]],
        sample: str,
    ) -> list[dict[str, Any]]:
        valid_idx = np.where(valid_mask)[0]
        valid_nll = token_nll[valid_idx]
        valid_excess = excess_surprise[valid_idx]
        if len(valid_excess) < 2:
            return []

        seq_outliers: list[dict[str, Any]] = []
        for i, (nll, excess) in enumerate(zip(valid_nll, valid_excess, strict=True)):
            if self.outlier_z_threshold is None or excess > self.outlier_z_threshold:
                start, end = offset_mapping[valid_idx[i]]
                seq_outliers.append(
                    {
                        "char": int(start),
                        "token": sample[start:end],
                        "nll": float(nll),
                        "z_score": float(excess),
                    }
                )

        return seq_outliers
