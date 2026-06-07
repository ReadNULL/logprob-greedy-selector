from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class DeltaScoreModel(nn.Module):
    def __init__(self, base_model_name_or_path: str, dtype: str = "auto") -> None:
        super().__init__()
        self.base_model_name_or_path = base_model_name_or_path
        torch_dtype = self._resolve_dtype(dtype)
        config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        kwargs = {"config": config, "trust_remote_code": True}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.backbone = AutoModel.from_pretrained(base_model_name_or_path, **kwargs)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("base model config does not expose hidden_size")
        self.score_head = nn.Linear(hidden_size, 1)

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_idx, lengths]
        return self.score_head(pooled).squeeze(-1)

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output / "backbone")
        torch.save(self.score_head.state_dict(), output / "score_head.pt")
        (output / "base_model_name_or_path.txt").write_text(self.base_model_name_or_path, encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path, dtype: str = "auto", device: str = "cpu") -> "DeltaScoreModel":
        model_dir = Path(model_dir)
        model = cls(str(model_dir / "backbone"), dtype=dtype)
        state = torch.load(model_dir / "score_head.pt", map_location="cpu")
        model.score_head.load_state_dict(state)
        model.to(torch.device(device))
        model.eval()
        return model


def load_tokenizer(base_or_model_dir: str | Path):
    path = Path(base_or_model_dir)
    if (path / "backbone").exists():
        path = path / "backbone"
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

