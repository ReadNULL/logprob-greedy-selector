from __future__ import annotations

from dataclasses import dataclass

import torch

from selector.data.context import build_prompt_with_answer
from selector.model.delta_model import DeltaScoreModel, load_tokenizer
from selector.scoring.base import DeltaPredictor


@dataclass
class DeltaModelPredictor(DeltaPredictor):
    model_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "auto"
    max_length: int = 4096

    def __post_init__(self) -> None:
        self.tokenizer = load_tokenizer(self.model_dir)
        self.model = DeltaScoreModel.load(self.model_dir, dtype=self.dtype, device=self.device)
        self.resolved_device = torch.device(self.device)

    @torch.inference_mode()
    def score(self, context: str, question: str, answer: str) -> float:
        text = build_prompt_with_answer(context=context, question=question, answer=answer)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.resolved_device) for k, v in encoded.items()}
        return float(self.model(encoded["input_ids"], encoded["attention_mask"]).item())

    def predict_delta(self, minus_context: str, plus_context: str, question: str, answer: str) -> float:
        return self.score(plus_context, question, answer) - self.score(minus_context, question, answer)

