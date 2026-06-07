from __future__ import annotations

from selector.data.context import join_chunks
from selector.data.schema import DeltaTrainingSample, GreedyPathRecord


def replay_greedy_path(record: GreedyPathRecord, chunk_separator: str = "\n\n") -> list[DeltaTrainingSample]:
    sample = record.sample
    selected_ids: list[int] = []
    current_score: float | None = None
    output: list[DeltaTrainingSample] = []

    for step in record.greedy_path:
        if step.action == "seed":
            selected_ids = [step.chunk]
            current_score = step.score
            continue

        if current_score is None:
            raise ValueError("greedy path must start with a seed step")

        minus_ids = sorted(selected_ids)
        plus_ids = sorted(set(selected_ids + [step.chunk]))
        minus_context = join_chunks(sample.chunks, minus_ids, separator=chunk_separator)
        plus_context = join_chunks(sample.chunks, plus_ids, separator=chunk_separator)
        target_delta = step.score - current_score

        output.append(
            DeltaTrainingSample(
                question=sample.question,
                answer=sample.answer,
                minus_context=minus_context,
                plus_context=plus_context,
                minus_context_ids=minus_ids,
                plus_context_ids=plus_ids,
                target_delta=target_delta,
                action=step.action,
                sample_id=sample.sample_id,
                dataset=sample.dataset,
            )
        )

        if step.action == "accept":
            selected_ids.append(step.chunk)
            current_score = step.score

    return output

