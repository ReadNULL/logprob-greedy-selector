from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from selector.data.context import build_prompt_with_answer
from selector.data.io import read_jsonl


class DeltaJsonlDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.rows = list(read_jsonl(path))
        if not self.rows:
            raise ValueError(f"empty delta dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "minus_text": build_prompt_with_answer(row["minus_context"], row["question"], row["answer"]),
            "plus_text": build_prompt_with_answer(row["plus_context"], row["question"], row["answer"]),
            "target_delta": float(row["target_delta"]),
            "action": row.get("action"),
        }


@dataclass
class DeltaBatchCollator:
    tokenizer: Any
    max_length: int = 4096

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        minus = self.tokenizer(
            [x["minus_text"] for x in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        plus = self.tokenizer(
            [x["plus_text"] for x in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "minus_input_ids": minus["input_ids"],
            "minus_attention_mask": minus["attention_mask"],
            "plus_input_ids": plus["input_ids"],
            "plus_attention_mask": plus["attention_mask"],
            "target_delta": torch.tensor([x["target_delta"] for x in rows], dtype=torch.float32),
        }

