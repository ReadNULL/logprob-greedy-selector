from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selector.data.io import read_jsonl, write_jsonl
from selector.data.schema import RawSample
from selector.scoring.delta_model_predictor import DeltaModelPredictor
from selector.select.greedy_selector import GreedySelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained delta model as a greedy chunk selector.")
    parser.add_argument("--input", required=True, help="Raw sample JSONL with question/answer/chunks.")
    parser.add_argument("--output", required=True, help="Output selected chunk JSONL.")
    parser.add_argument("--model-dir", required=True, help="Directory saved by 03_train_delta_model.py.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--chunk-separator", default="\n\n")
    parser.add_argument("--candidate-top-p", type=float, default=0.95)
    parser.add_argument("--solo-rank-cap", type=int, default=20)
    parser.add_argument("--delta-offset", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DeltaModelPredictor(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
    )
    selector = GreedySelector(
        predictor=predictor,
        chunk_separator=args.chunk_separator,
        candidate_top_p=args.candidate_top_p,
        solo_rank_cap=args.solo_rank_cap,
        delta_offset=args.delta_offset,
    )

    def rows():
        for idx, row in enumerate(read_jsonl(args.input)):
            if args.limit is not None and idx >= args.limit:
                break
            yield selector.select(RawSample.from_dict(row))

    count = write_jsonl(args.output, rows())
    print(f"wrote {count} greedy selector records to {args.output}")


if __name__ == "__main__":
    main()

