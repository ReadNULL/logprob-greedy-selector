from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from selector.data.context import build_prompt
from selector.scoring.base import LogprobScorer


@dataclass
class HFLogprobScorer(LogprobScorer):
    model_name_or_path: str
    device: str = "auto"
    dtype: str = "auto"
    max_length: int | None = None

    def __post_init__(self) -> None:
        torch_dtype = self._resolve_dtype(self.dtype)
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        resolved_device = self._resolve_device(self.device)
        self.model.to(resolved_device)
        self.model.eval()
        self.resolved_device = resolved_device

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype | None:
        if dtype == "auto":
            return None
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"unsupported dtype: {dtype}")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @torch.inference_mode()
    def score(self, context: str, question: str, answer: str) -> float:
        prompt = build_prompt(context=context, question=question)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        if not answer_ids:
            raise ValueError("answer tokenization produced no tokens")

        input_ids = prompt_ids + answer_ids
        if self.max_length is not None and len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            if overflow >= len(prompt_ids):
                raise ValueError("max_length is too small to keep the answer tokens")
            prompt_ids = prompt_ids[overflow:]
            input_ids = prompt_ids + answer_ids

        x = torch.tensor([input_ids], dtype=torch.long, device=self.resolved_device)
        logits = self.model(input_ids=x).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        labels = x[:, 1:]

        answer_start = len(prompt_ids)
        shifted_answer_start = max(answer_start - 1, 0)
        shifted_answer_end = len(input_ids) - 1
        answer_labels = labels[:, shifted_answer_start:shifted_answer_end]
        answer_log_probs = log_probs[:, shifted_answer_start:shifted_answer_end, :]
        token_log_probs = answer_log_probs.gather(dim=-1, index=answer_labels.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.sum().item())

