from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selector.data.build_delta_dataset import replay_greedy_path
from selector.data.io import read_jsonl, write_jsonl
from selector.data.schema import GreedyPathRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay teacher greedy paths into delta training samples.")
    parser.add_argument("--input", required=True, help="Input JSONL created by 01_build_teacher_paths.py.")
    parser.add_argument("--output", required=True, help="Output JSONL delta dataset.")
    parser.add_argument("--chunk-separator", default="\n\n")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of path records for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def rows():
        for idx, row in enumerate(read_jsonl(args.input)):
            if args.limit is not None and idx >= args.limit:
                break
            record = GreedyPathRecord.from_dict(row)
            for item in replay_greedy_path(record, chunk_separator=args.chunk_separator):
                yield item.to_dict()

    count = write_jsonl(Path(args.output), rows())
    print(f"wrote {count} delta training samples to {args.output}")


if __name__ == "__main__":
    main()
