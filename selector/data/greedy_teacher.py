from __future__ import annotations

from dataclasses import dataclass

from selector.data.context import join_chunks
from selector.data.schema import GreedyPathRecord, GreedyStep, RawSample
from selector.scoring.base import LogprobScorer


@dataclass
class TeacherGreedyBuilder:
    scorer: LogprobScorer
    chunk_separator: str = "\n\n"
    max_candidates: int | None = None
    require_strict_improvement: bool = True

    def build(self, sample: RawSample) -> GreedyPathRecord:
        single_scores = self._score_single_chunks(sample)
        ranked = sorted(single_scores, key=lambda x: float(x["score"]), reverse=True)
        if self.max_candidates is not None:
            ranked = ranked[: self.max_candidates]
        if not ranked:
            raise ValueError("no candidate chunks available")

        seed_id = int(ranked[0]["chunk"])
        selected_ids = [seed_id]
        current_score = self._score_context_ids(sample, selected_ids)
        path = [GreedyStep(chunk=seed_id, action="seed", score=current_score, iteration=0)]

        iteration = 1
        for item in ranked[1:]:
            candidate_id = int(item["chunk"])
            candidate_ids = selected_ids + [candidate_id]
            candidate_score = self._score_context_ids(sample, candidate_ids)
            improved = (
                candidate_score > current_score
                if self.require_strict_improvement
                else candidate_score >= current_score
            )
            if improved:
                selected_ids.append(candidate_id)
                current_score = candidate_score
                action = "accept"
            else:
                action = "reject"
            path.append(
                GreedyStep(
                    chunk=candidate_id,
                    action=action,
                    score=candidate_score,
                    iteration=iteration,
                )
            )
            iteration += 1

        return GreedyPathRecord(sample=sample, single_scores=single_scores, greedy_path=path)

    def _score_single_chunks(self, sample: RawSample) -> list[dict[str, float | int]]:
        scores = []
        for idx, chunk in enumerate(sample.chunks):
            score = self.scorer.score(context=chunk, question=sample.question, answer=sample.answer)
            scores.append({"chunk": idx, "score": score})
        return scores

    def _score_context_ids(self, sample: RawSample, indices: list[int]) -> float:
        context = join_chunks(sample.chunks, indices, separator=self.chunk_separator)
        return self.scorer.score(context=context, question=sample.question, answer=sample.answer)

