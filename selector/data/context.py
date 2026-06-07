from __future__ import annotations


def join_chunks(chunks: list[str], indices: list[int], separator: str = "\n\n") -> str:
    """Join selected chunks in original document order."""
    return separator.join(chunks[i] for i in sorted(indices))


def build_prompt(context: str, question: str) -> str:
    return (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
    )


def build_prompt_with_answer(context: str, question: str, answer: str) -> str:
    return build_prompt(context=context, question=question) + answer

