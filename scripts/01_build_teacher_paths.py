from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selector.data.greedy_teacher import TeacherGreedyBuilder
from selector.data.io import read_jsonl, write_jsonl
from selector.data.schema import RawSample
from selector.scoring.hf_logprob_scorer import HFLogprobScorer
from selector.scoring.openai_logprob_scorer import OpenAICompatibleCompletionsLogprobScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher greedy paths from raw QA chunk samples.")
    parser.add_argument("--input", required=True, help="Input JSONL with question/answer/chunks.")
    parser.add_argument("--output", required=True, help="Output JSONL with single_scores and greedy_path.")
    parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "openai-completions", "openai-compatible-completions"],
        help="Teacher scoring backend.",
    )
    parser.add_argument("--model", required=True, help="Model name or local path used as teacher scorer.")
    parser.add_argument("--device", default="auto", help="Device for HF model: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--openai-base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--openai-timeout", type=float, default=60.0)
    parser.add_argument("--openai-logprobs", type=int, default=1)
    parser.add_argument("--openai-max-tokens", type=int, default=0)
    parser.add_argument("--chunk-separator", default="\n\n")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "hf":
        scorer = HFLogprobScorer(
            model_name_or_path=args.model,
            device=args.device,
            dtype=args.dtype,
            max_length=args.max_length,
        )
    elif args.backend in {"openai-completions", "openai-compatible-completions"}:
        scorer = OpenAICompatibleCompletionsLogprobScorer(
            model=args.model,
            api_key_env=args.openai_api_key_env,
            base_url=args.openai_base_url,
            timeout=args.openai_timeout,
            logprobs=args.openai_logprobs,
            max_tokens=args.openai_max_tokens,
        )
    else:
        raise ValueError(f"unsupported backend: {args.backend}")
    builder = TeacherGreedyBuilder(
        scorer=scorer,
        chunk_separator=args.chunk_separator,
        max_candidates=args.max_candidates,
    )

    def rows():
        for idx, row in enumerate(read_jsonl(args.input)):
            if args.limit is not None and idx >= args.limit:
                break
            sample = RawSample.from_dict(row)
            yield builder.build(sample).to_dict()

    count = write_jsonl(Path(args.output), rows())
    print(f"wrote {count} teacher greedy path records to {args.output}")


if __name__ == "__main__":
    main()
