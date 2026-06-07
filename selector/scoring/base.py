from __future__ import annotations

from abc import ABC, abstractmethod


class LogprobScorer(ABC):
    @abstractmethod
    def score(self, context: str, question: str, answer: str) -> float:
        """Return summed token log probability of answer conditioned on context/question."""


class DeltaPredictor(ABC):
    @abstractmethod
    def predict_delta(self, minus_context: str, plus_context: str, question: str, answer: str) -> float:
        """Return predicted score(plus) - score(minus)."""

