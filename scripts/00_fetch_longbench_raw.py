from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from selector.data.io import write_jsonl


DEFAULT_COUNTS = {
    "2wikimqa": 80,
    "hotpotqa": 80,
    "hotpotqa_e": 80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch LongBench samples and normalize them to raw selector JSONL.")
    parser.add_argument("--repo", default="zai-org/LongBench", help="Hugging Face dataset repo.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="data/raw/samples.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chunk-words", type=int, default=220)
    parser.add_argument(
        "--dataset-count",
        action="append",
        default=[],
        help="Dataset count in NAME=N form. Can be repeated. Defaults to 80 each for 2wikimqa/hotpotqa/hotpotqa_e.",
    )
    return parser.parse_args()


def parse_counts(items: list[str]) -> dict[str, int]:
    if not items:
        return dict(DEFAULT_COUNTS)
    counts: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --dataset-count value: {item!r}; expected NAME=N")
        name, value = item.split("=", 1)
        counts[name.strip()] = int(value)
    return counts


def split_context_to_chunks(context: str, max_words: int) -> list[str]:
    paragraphs = [x.strip() for x in context.replace("\r\n", "\n").split("\n\n") if x.strip()]
    if not paragraphs:
        paragraphs = [context.strip()] if context.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_words:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0
            for start in range(0, len(words), max_words):
                chunks.append(" ".join(words[start : start + max_words]))
            continue

        if current and current_words + len(words) > max_words:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0

        current.append(paragraph)
        current_words += len(words)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def first_answer(answers: Any) -> str:
    if isinstance(answers, list) and answers:
        return str(answers[0])
    if isinstance(answers, str):
        return answers
    raise ValueError(f"cannot extract answer from: {answers!r}")


def normalize_row(row: dict[str, Any], dataset_name: str, max_chunk_words: int) -> dict[str, Any]:
    question = row.get("input")
    context = row.get("context")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("LongBench row missing non-empty input field")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("LongBench row missing non-empty context field")

    chunks = split_context_to_chunks(context, max_words=max_chunk_words)
    if not chunks:
        raise ValueError("context chunking produced no chunks")

    return {
        "sample_id": str(row.get("_id", "")),
        "dataset": dataset_name,
        "question": question,
        "answer": first_answer(row.get("answers")),
        "chunks": chunks,
        "metadata": {
            "source_repo": "LongBench",
            "source_dataset": dataset_name,
            "source_length": row.get("length"),
            "language": row.get("language"),
            "all_answers": row.get("answers"),
            "chunking": {
                "method": "paragraph_merge_by_word_count",
                "max_chunk_words": max_chunk_words,
            },
        },
    }


def shuffled_take(rows: Iterable[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < count:
        raise ValueError(f"requested {count} rows but only found {len(rows)}")
    return rows[:count]


def main() -> None:
    args = parse_args()
    counts = parse_counts(args.dataset_count)
    output_rows: list[dict[str, Any]] = []

    for dataset_name, count in counts.items():
        print(f"loading {dataset_name} from {args.repo} ({count} rows)")
        ds = load_dataset(args.repo, dataset_name, split=args.split, trust_remote_code=True)
        sampled = shuffled_take(ds, count=count, seed=args.seed + len(output_rows))
        for row in sampled:
            output_rows.append(normalize_row(row, dataset_name, max_chunk_words=args.max_chunk_words))

    written = write_jsonl(Path(args.output), output_rows)
    print(f"wrote {written} normalized raw samples to {args.output}")


if __name__ == "__main__":
    main()
