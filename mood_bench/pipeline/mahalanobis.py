from typing import Any, Literal, cast

import numpy as np
import torch as t
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

from mood_bench.data import load_mood_dataset
from mood_bench.pipeline.base import Pipeline, PipelineResult
from mood_bench.tokenize import rendered

PoolingStrategy = Literal["cls", "mean", "max"]


def _pool_hidden_states(
    last_hidden_state: t.Tensor,
    attention_mask: t.Tensor,
    strategy: PoolingStrategy,
) -> t.Tensor:
    if strategy == "cls":
        embedding_batch = last_hidden_state[:, -1, :]
    else:
        mask = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).type_as(last_hidden_state)
        )
        if strategy == "mean":
            sum_embeddings = (last_hidden_state * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            embedding_batch = sum_embeddings / sum_mask
        else:
            masked_hidden_state = last_hidden_state.clone()
            masked_hidden_state[mask == 0] = -1e9
            embedding_batch = masked_hidden_state.max(dim=1)[0]

    return embedding_batch.squeeze(1)


def get_stats_for_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pooling_strategy: PoolingStrategy = "cls",
    batch_size: int = 16,
    max_samples: int | None = None,
) -> dict[str, t.Tensor]:
    ds = load_mood_dataset(split="train")

    safe_ds = ds.filter(lambda x: x["unsafe"] == 0)
    if max_samples is not None and safe_ds.num_rows > max_samples:
        safe_ds = safe_ds.shuffle().select(range(max_samples))

    conversations = safe_ds["conversation"]
    features_list: list[t.Tensor] = []
    with t.inference_mode():
        for batch in rendered(
            conversations,
            renderer=tokenizer,
            device=model.device,
            batch_size=batch_size,
            max_length=4096,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ):
            outputs = model(
                input_ids=cast(t.Tensor, batch["input_ids"]),
                attention_mask=cast(t.Tensor, batch["attention_mask"]),
                output_hidden_states=True,
            )

            last_hidden_state = outputs.hidden_states[-1]
            pooled_features = (
                _pool_hidden_states(last_hidden_state, batch["attention_mask"], pooling_strategy)
                .contiguous()
                .clone()
            )

            features_list.append(pooled_features)

    features = t.cat(features_list, dim=0).to(t.float64)
    mean = features.mean(dim=0)
    centered = features - mean

    eps = 1e-6
    cov = (centered.T @ centered) / (centered.size(0) - 1)
    cov += t.eye(cov.size(0), device=cov.device) * eps
    inv_cov = t.linalg.pinv(cov)

    dists_sq = t.sum((centered @ inv_cov) * centered, dim=1)
    calibration_var = dists_sq.var(unbiased=False)

    return {
        "mean": mean,
        "cov": cov,
        "inv_cov": inv_cov,
        "var": calibration_var,
    }


class MahalanobisPipeline(Pipeline):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        mean: t.Tensor,
        inv_cov: t.Tensor,
        pooling_strategy: PoolingStrategy = "cls",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pooling_strategy = pooling_strategy
        self.mean = mean
        self.inv_cov = inv_cov

    def __call__(self, samples: list[str], **kwargs: Any) -> PipelineResult:
        batch_size = kwargs.get("batch_size", 16)
        anomaly_scores_list: list[np.ndarray] = []
        for batch in rendered(
            samples,
            renderer=self.tokenizer,
            device=self.model.device,
            batch_size=batch_size,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ):
            features, logits = self._batch_inference(batch)
            features = features.to(t.float64)

            centered = features - self.mean
            dists_squared = t.sum((centered @ self.inv_cov) * centered, dim=1)
            dists = t.sqrt(t.clamp(dists_squared, min=0))

            anomaly_scores_list.append(dists.detach().cpu().numpy())

        anomaly_scores = np.concatenate(anomaly_scores_list)
        return anomaly_scores, {}

    def _batch_inference(self, enc: BatchEncoding) -> tuple[t.Tensor, t.Tensor]:
        input_ids = cast(t.Tensor, enc["input_ids"])
        attention_mask = cast(t.Tensor, enc["attention_mask"])

        with t.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            last_hidden_state = outputs.hidden_states[-1]
            pooled_features = (
                _pool_hidden_states(last_hidden_state, attention_mask, self.pooling_strategy)
                .contiguous()
                .clone()
            )

            return pooled_features, outputs.logits
