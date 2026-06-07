from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from selector.data.context import join_chunks
from selector.data.schema import RawSample
from selector.scoring.base import DeltaPredictor


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [exp(x - max_value) for x in values]
    total = sum(exps)
    return [x / total for x in exps]


@dataclass
class GreedySelector:
    predictor: DeltaPredictor
    chunk_separator: str = "\n\n"
    candidate_top_p: float = 0.95
    solo_rank_cap: int = 20
    delta_offset: float = 0.0

    def select(self, sample: RawSample) -> dict[str, Any]:
        single = self._single_chunk_scores(sample)
        ranked = sorted(single, key=lambda x: x["delta"], reverse=True)
        pool = self._top_p_pool(ranked)
        if not pool:
            return {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "selected_chunk_ids": [],
                "single_scores": single,
                "path": [],
            }

        selected = [pool[0]["chunk"]]
        path = [{"chunk": selected[0], "action": "seed", "delta": pool[0]["delta"]}]
        for item in pool[1:]:
            candidate = item["chunk"]
            minus_context = join_chunks(sample.chunks, selected, separator=self.chunk_separator)
            plus_context = join_chunks(sample.chunks, selected + [candidate], separator=self.chunk_separator)
            delta = self.predictor.predict_delta(minus_context, plus_context, sample.question, sample.answer)
            if delta >= self.delta_offset:
                selected.append(candidate)
                action = "accept"
            else:
                action = "reject"
            path.append({"chunk": candidate, "action": action, "delta": delta})

        selected_sorted = sorted(selected)
        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "question": sample.question,
            "answer": sample.answer,
            "selected_chunk_ids": selected_sorted,
            "selected_context": join_chunks(sample.chunks, selected_sorted, separator=self.chunk_separator),
            "single_scores": single,
            "candidate_pool": [x["chunk"] for x in pool],
            "path": path,
        }

    def _single_chunk_scores(self, sample: RawSample) -> list[dict[str, float | int]]:
        output = []
        for idx, chunk in enumerate(sample.chunks):
            delta = self.predictor.predict_delta("", chunk, sample.question, sample.answer)
            output.append({"chunk": idx, "delta": delta})
        return output

    def _top_p_pool(self, ranked: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
        capped = ranked[: self.solo_rank_cap]
        probs = softmax([float(x["delta"]) for x in capped])
        pool = []
        cumulative = 0.0
        for item, prob in zip(capped, probs):
            pool.append(item)
            cumulative += prob
            if cumulative >= self.candidate_top_p:
                break
        return pool

