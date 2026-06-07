from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selector.model.dataset import DeltaBatchCollator, DeltaJsonlDataset
from selector.model.delta_model import DeltaScoreModel, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scalar score-head model on delta JSONL samples.")
    parser.add_argument("--train", required=True, help="Delta train JSONL.")
    parser.add_argument("--base-model", required=True, help="HF base model path/name, e.g. Qwen/Qwen3-0.6B.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the scalar score head.")
    return parser.parse_args()


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_n = int(n * val_ratio)
    return indices[val_n:], indices[:val_n]


def run_eval(model: DeltaScoreModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    sign_correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            minus_score = model(batch["minus_input_ids"], batch["minus_attention_mask"])
            plus_score = model(batch["plus_input_ids"], batch["plus_attention_mask"])
            pred_delta = plus_score - minus_score
            target = batch["target_delta"]
            loss = F.huber_loss(pred_delta.float(), target.float(), reduction="mean")
            losses.append(float(loss.item()))
            sign_correct += int(((pred_delta >= 0) == (target >= 0)).sum().item())
            total += int(target.numel())
    model.train()
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "sign_acc": sign_correct / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = load_tokenizer(args.base_model)
    dataset = DeltaJsonlDataset(args.train)
    train_indices, val_indices = split_indices(len(dataset), args.val_ratio, args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    collator = DeltaBatchCollator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        if val_dataset is not None
        else None
    )

    model = DeltaScoreModel(args.base_model, dtype=args.dtype).to(device)
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            minus_score = model(batch["minus_input_ids"], batch["minus_attention_mask"])
            plus_score = model(batch["plus_input_ids"], batch["plus_attention_mask"])
            pred_delta = plus_score - minus_score
            loss = F.huber_loss(pred_delta.float(), batch["target_delta"].float(), reduction="mean")
            (loss / args.grad_accum).backward()

            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    sign_acc = ((pred_delta >= 0) == (batch["target_delta"] >= 0)).float().mean().item()
                    print(
                        f"epoch={epoch + 1} step={step} "
                        f"loss={loss.item():.6f} sign_acc={sign_acc:.3f}"
                    )

        if val_loader is not None:
            metrics = run_eval(model, val_loader, device=device)
            print(
                f"epoch={epoch + 1} val_loss={metrics['loss']:.6f} "
                f"val_sign_acc={metrics['sign_acc']:.3f}"
            )

    output = Path(args.output_dir)
    model.save(output)
    tokenizer.save_pretrained(output / "backbone")
    print(f"saved delta score model to {output}")


if __name__ == "__main__":
    main()

