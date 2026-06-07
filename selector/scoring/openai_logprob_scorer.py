from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from selector.data.context import build_prompt
from selector.scoring.base import LogprobScorer


@dataclass
class OpenAICompatibleCompletionsLogprobScorer(LogprobScorer):
    """Score gold answers with an OpenAI-compatible Completions prompt-logprob path.

    This scorer uses `echo=True` so the API returns logprobs for the prompt text.
    The full prompt is `context/question` prefix plus the gold answer, and only
    tokens whose text offsets fall inside the answer span are summed.
    """

    model: str = "gpt-3.5-turbo-instruct"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    timeout: float = 60.0
    logprobs: int = 1
    max_tokens: int = 0

    def __post_init__(self) -> None:
        kwargs = {"timeout": self.timeout}
        api_key = self.api_key or os.getenv(self.api_key_env)
        if api_key is not None:
            kwargs["api_key"] = api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)

    def score(self, context: str, question: str, answer: str) -> float:
        prefix = build_prompt(context=context, question=question)
        full_prompt = prefix + answer
        answer_start = len(prefix)
        answer_end = len(full_prompt)

        response = self.client.completions.create(
            model=self.model,
            prompt=full_prompt,
            max_tokens=self.max_tokens,
            temperature=0,
            echo=True,
            logprobs=self.logprobs,
        )
        choice = response.choices[0]
        logprobs = choice.logprobs
        if logprobs is None:
            raise RuntimeError("OpenAI completion response did not include logprobs")

        token_logprobs = logprobs.token_logprobs
        text_offsets = logprobs.text_offset
        if token_logprobs is None or text_offsets is None:
            raise RuntimeError("OpenAI completion logprobs missing token_logprobs/text_offset")

        selected = [
            lp
            for lp, offset in zip(token_logprobs, text_offsets)
            if offset is not None and answer_start <= offset < answer_end and lp is not None
        ]
        if not selected:
            raise RuntimeError("no answer token logprobs found in OpenAI completion response")
        return float(sum(selected))


OpenAICompletionsLogprobScorer = OpenAICompatibleCompletionsLogprobScorer
