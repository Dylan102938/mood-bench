from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import numpy as np

import mood_bench._output as logger
from mood_bench._prompts import read_prompt_file, render_prompt
from mood_bench.pipeline.base import Pipeline, PipelineResult
from mood_bench.tokenize import load_tokenizer, rendered

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

PromptBuilder = Callable[[str], str]
SCORE_RE = re.compile(r"SCORE:\s*(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)", re.IGNORECASE)
RUBRIC_TEMPLATES: dict[str, str] = {
    "alignment": "scoring_rubric_alignment.jinja",
    "uncertainty": "scoring_rubric_uncertainty.jinja",
}


def sample_icl_examples(
    negative_examples: list[str] | None,
    num_few_shot: int,
    rng: random.Random,
) -> list[str]:
    if not negative_examples or num_few_shot <= 0:
        return []

    return rng.sample(negative_examples, min(num_few_shot, len(negative_examples)))


def format_conversation(
    conversation: str,
    icl_examples: list[str],
    grading_type: str,
) -> str:
    scoring_rubric = render_prompt(RUBRIC_TEMPLATES[grading_type])
    if icl_examples:
        return render_prompt(
            "instruction_grading_icl.jinja",
            conversation=conversation.strip(),
            icl_examples=icl_examples,
            scoring_rubric=scoring_rubric,
        )

    return render_prompt(
        "instruction_grading_no_icl.jinja",
        conversation=conversation.strip(),
        scoring_rubric=scoring_rubric,
    )


class InstructionTunedPipeline(Pipeline):
    _VLLM_ONLY_KEYS = frozenset(
        {
            "enforce_eager",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "max_lora_rank",
        }
    )

    def __init__(
        self,
        model_id: str,
        *,
        adapter_id: str | None = None,
        prompt_builder: PromptBuilder | None = None,
        grading_type: Literal["alignment", "uncertainty"] = "alignment",
        num_few_shot: int = 0,
        icl_malign_examples: list[str] | None = None,
        icl_seed: int = 42,
        max_retries: int = 3,
        default_score: float = 0.0,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        # vLLM specific parameters
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int | None = None,
        max_lora_rank: int = 64,
        enforce_eager: bool = False,
        **model_kwargs: Any,
    ) -> None:
        ### Model kwargs ###
        self._model_id = model_id
        self._adapter_id = adapter_id
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._gpu_memory_utilization = gpu_memory_utilization
        self._tensor_parallel_size = tensor_parallel_size
        self._max_lora_rank = max_lora_rank
        self._enforce_eager = enforce_eager
        self._model_kwargs = model_kwargs

        ### Sampling kwargs ###
        self.max_retries = max_retries
        self.default_score = default_score
        self.grading_type = grading_type
        self.num_few_shot = num_few_shot
        self.icl_malign_examples = icl_malign_examples
        self._rng = random.Random(icl_seed)
        self._custom_prompt_builder = prompt_builder

        ### State values ###
        self._loaded = False
        self._backend: Literal["vllm", "hf"] = "hf"
        self._llm: LLM | None = None
        self._hf_model: Any = None
        self._lora_request: LoRARequest | None = None
        self._sampling_params: SamplingParams | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None

    def __call__(self, samples: list[str], **kwargs: Any) -> PipelineResult:
        ### Set up inference backend ###
        if not self._loaded:
            from importlib.util import find_spec

            if find_spec("vllm") is not None:
                self._load_vllm()
            else:
                logger.warn(
                    "vLLM not installed, falling back to HF transformers backend. "
                    "Install vLLM for faster inference."
                )
                self._load_hf()

            self._loaded = True

        ### Render prompts and create output slots ###
        batch_size = kwargs.get("batch_size", 1)
        prompts = [self._render_prompt(s) for s in samples]
        scores = np.full(len(samples), np.nan, dtype=float)
        reasonings: list[str] = [""] * len(samples)
        pending: list[int] = list(range(len(samples)))

        for attempt in range(self.max_retries + 1):
            if not pending:
                break
            if attempt > 0:
                logger.info(
                    f"InstructionTunedPipeline: resampling {len(pending)} unparseable "
                    f"output(s) (attempt {attempt}/{self.max_retries})"
                )

            ### Call instruction-tuned judge ###
            pending_prompts = [prompts[i] for i in pending]
            if self._backend == "vllm":
                texts = self._generate_vllm(pending_prompts)
            else:
                texts = self._generate_hf(pending_prompts, batch_size=batch_size)

            ### Parse outputs ###
            still_pending: list[int] = []
            for orig_idx, text in zip(pending, texts, strict=True):
                text = text.strip()
                reasonings[orig_idx] = text
                try:
                    scores[orig_idx] = float(SCORE_RE.search(text).group(1))  # type: ignore[union-attr]
                except (AttributeError, ValueError):
                    still_pending.append(orig_idx)

            pending = still_pending

        ### Handle any remaining errors ###
        if pending:
            logger.warn(
                f"InstructionTunedPipeline: defaulting {len(pending)} sample(s) to "
                f"{self.default_score} after {self.max_retries} retries"
            )
            scores[pending] = self.default_score

        meta = {"reasoning": reasonings, "prompt": prompts}
        return scores, meta

    def _load_vllm(self) -> None:
        import torch
        from vllm import LLM, SamplingParams

        ### Configure variables ###
        has_adapter = self._adapter_id is not None
        tp = self._tensor_parallel_size
        if tp is None:
            tp = torch.cuda.device_count()
            logger.info(f"vLLM: auto-detected {tp} GPU(s)")

        ### Load vLLM model ###
        max_lora_rank = self._max_lora_rank if has_adapter else None
        vllm_kwargs = dict(self._model_kwargs)
        if "torch_dtype" in vllm_kwargs:
            vllm_kwargs["dtype"] = vllm_kwargs.pop("torch_dtype")
        if "dtype" in vllm_kwargs and not isinstance(vllm_kwargs["dtype"], str):
            vllm_kwargs["dtype"] = str(vllm_kwargs["dtype"]).replace("torch.", "")

        self._llm = LLM(
            model=self._model_id,
            tensor_parallel_size=tp,
            gpu_memory_utilization=self._gpu_memory_utilization,
            enable_lora=has_adapter,
            max_lora_rank=max_lora_rank,
            enforce_eager=self._enforce_eager,
            **vllm_kwargs,
        )

        ### (Optionally) load LoRA adapter ###
        if has_adapter:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_path=self._adapter_id,
            )

        ### Load tokenizer and sampling parameters ###
        self.tokenizer = load_tokenizer(self._adapter_id or self._model_id)
        if self.tokenizer.chat_template is None:
            self.tokenizer = load_tokenizer(self._model_id)
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = read_prompt_file("default_chat_template.jinja")

        self._sampling_params = SamplingParams(
            temperature=self._temperature,
            max_tokens=self._max_new_tokens,
            stop=self.tokenizer.eos_token,
        )

        ### Set backend ###
        self._backend = "vllm"

    def _load_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_kwargs = {k: v for k, v in self._model_kwargs.items() if k not in self._VLLM_ONLY_KEYS}

        ### Load model and (optionally) apply LoRA adapter ###
        model = AutoModelForCausalLM.from_pretrained(self._model_id, **hf_kwargs)
        if self._adapter_id is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self._adapter_id).merge_and_unload()

        ### Do additional configuration on model ###
        self.tokenizer = load_tokenizer(self._adapter_id or self._model_id)
        if self.tokenizer.chat_template is None:
            self.tokenizer = load_tokenizer(self._model_id)
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = read_prompt_file("default_chat_template.jinja")
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        if self._model_kwargs.get("device_map") is None:
            model = model.to(device)
        self._hf_model = model.eval()

        ### Set backend ###
        self._backend = "hf"

    def _generate_vllm(self, prompts: list[str]) -> list[str]:
        lora_kwargs: dict[str, Any] = (
            {"lora_request": self._lora_request} if self._lora_request is not None else {}
        )
        outputs = self._llm.generate(prompts, self._sampling_params, **lora_kwargs)
        return [o.outputs[0].text for o in outputs]

    def _generate_hf(self, prompts: list[str], *, batch_size: int = 1) -> list[str]:
        import torch

        model = self._hf_model
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._temperature > 0,
        }
        if self._temperature > 0:
            gen_kwargs["temperature"] = self._temperature

        results: list[str] = []
        for batch in rendered(
            prompts,
            renderer=self.tokenizer,
            device=model.device,
            batch_size=batch_size,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ):
            input_len = batch["input_ids"].shape[-1]
            with torch.inference_mode():
                output_ids = model.generate(**batch, **gen_kwargs)

            new_tokens = output_ids[:, input_len:]
            results.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

        return results

    def _render_prompt(self, conversation: str) -> str:
        if self._custom_prompt_builder is not None:
            raw = self._custom_prompt_builder(conversation)
        else:
            icl = sample_icl_examples(self.icl_malign_examples, self.num_few_shot, self._rng)
            raw = format_conversation(conversation, icl, self.grading_type)

        return cast(
            str,
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
