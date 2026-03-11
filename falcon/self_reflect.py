from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from falcon.llm import LLM


@dataclass
class SelfReflectConfig:
    enabled: bool = False
    max_output_tokens: int = 256
    temperature: float = 0.2
    instruction: str = (
        "Revise the answer to remove contradictions while preserving correctness."
    )


def build_self_reflect_prompt(
    question: str,
    answer: str,
    instruction: str,
) -> str:
    return (
        "You are revising a model answer.\n"
        "Your goal is to remove internal contradictions while preserving useful and correct content.\n"
        "Do not mention that you are revising the answer. Do not explain your process.\n\n"
        f"Instruction:\n{instruction.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Original answer:\n{answer.strip()}\n\n"
        "Revised answer:"
    )


def run_self_reflection(
    question: str,
    answer: str,
    llm: Optional[LLM],
    cfg: SelfReflectConfig,
) -> Optional[str]:
    if not cfg.enabled or llm is None:
        return None

    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        return None

    prompt = build_self_reflect_prompt(question, answer, cfg.instruction)
    out = llm.generate(prompt).text.strip()
    return out if out else None