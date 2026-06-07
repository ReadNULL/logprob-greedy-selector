from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Action = Literal["seed", "accept", "reject"]


@dataclass(frozen=True)
class RawSample:
    question: str
    answer: str
    chunks: list[str]
    sample_id: str | None = None
    dataset: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RawSample":
        question = row.get("question")
        answer = row.get("answer")
        chunks = row.get("chunks")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("raw sample requires non-empty string field: question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("raw sample requires non-empty string field: answer")
        if not isinstance(chunks, list) or not chunks or not all(isinstance(x, str) for x in chunks):
            raise ValueError("raw sample requires non-empty list[str] field: chunks")

        sample_id = row.get("sample_id") or row.get("id")
        dataset = row.get("dataset")
        metadata = row.get("metadata")
        return cls(
            question=question,
            answer=answer,
            chunks=chunks,
            sample_id=str(sample_id) if sample_id is not None else None,
            dataset=str(dataset) if dataset is not None else None,
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "chunks": self.chunks,
        }
        if self.sample_id is not None:
            out["sample_id"] = self.sample_id
        if self.dataset is not None:
            out["dataset"] = self.dataset
        if self.metadata is not None:
            out["metadata"] = self.metadata
        return out


@dataclass(frozen=True)
class GreedyStep:
    chunk: int
    action: Action
    score: float
    iteration: int

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GreedyStep":
        action = row.get("action")
        if action not in {"seed", "accept", "reject"}:
            raise ValueError(f"invalid greedy action: {action!r}")
        return cls(
            chunk=int(row["chunk"]),
            action=action,
            score=float(row["score"]),
            iteration=int(row.get("iteration", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk,
            "action": self.action,
            "score": self.score,
            "iteration": self.iteration,
        }


@dataclass(frozen=True)
class GreedyPathRecord:
    sample: RawSample
    single_scores: list[dict[str, float | int]]
    greedy_path: list[GreedyStep]

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GreedyPathRecord":
        sample = RawSample.from_dict(row)
        single_scores = row.get("single_scores")
        greedy_path = row.get("greedy_path")
        if not isinstance(single_scores, list):
            raise ValueError("greedy path record requires list field: single_scores")
        if not isinstance(greedy_path, list):
            raise ValueError("greedy path record requires list field: greedy_path")
        return cls(
            sample=sample,
            single_scores=single_scores,
            greedy_path=[GreedyStep.from_dict(x) for x in greedy_path],
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.sample.to_dict()
        out["single_scores"] = self.single_scores
        out["greedy_path"] = [x.to_dict() for x in self.greedy_path]
        return out


@dataclass(frozen=True)
class DeltaTrainingSample:
    question: str
    answer: str
    minus_context: str
    plus_context: str
    minus_context_ids: list[int]
    plus_context_ids: list[int]
    target_delta: float
    action: Literal["accept", "reject"]
    sample_id: str | None = None
    dataset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "minus_context": self.minus_context,
            "plus_context": self.plus_context,
            "minus_context_ids": self.minus_context_ids,
            "plus_context_ids": self.plus_context_ids,
            "target_delta": self.target_delta,
            "action": self.action,
        }
        if self.sample_id is not None:
            out["sample_id"] = self.sample_id
        if self.dataset is not None:
            out["dataset"] = self.dataset
        return out

