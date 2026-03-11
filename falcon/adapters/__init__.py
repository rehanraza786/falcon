from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset, DownloadMode


@dataclass
class TruthfulQAAdapter:
    split: str = "validation"

    def load(self):
        try:
            return load_dataset(
                "truthful_qa",
                "generation",
                split=self.split,
            )
        except ValueError as e:
            msg = str(e)
            if "Feature type 'List' not found" in msg:
                return load_dataset(
                    "truthful_qa",
                    "generation",
                    split=self.split,
                    download_mode=DownloadMode.FORCE_REDOWNLOAD,
                )
            raise

    def get_question(self, ex: Dict[str, Any]) -> str:
        # truthful_qa/generation fields typically include 'question'
        q = ex.get("question", "")
        # Remove yes/no constraint to allow free-form answers that can contain contradictions
        return f"{q}"

    def get_gold(self, ex: Dict[str, Any]) -> str:
        return ex.get("best_answer", "")

    def get_baseline(self, ex: Dict[str, Any]) -> str:
        # If no LLM is enabled, you may treat the baseline as empty or use a
        # provided field. Here we return empty to avoid leaking dataset labels.
        return ""


@dataclass
class StrategyQAAdapter:
    split: str = "test"

    def load(self):
        return load_dataset(
            "wics/strategy-qa",
            split=self.split,
        )

    def get_question(self, ex: Dict[str, Any]) -> str:
        return ex.get("question", "")

    def get_gold(self, ex: Dict[str, Any]) -> str:
        # Convert boolean to "yes"/"no" style if needed; otherwise string
        ans = ex.get("answer", "")
        if isinstance(ans, bool):
            return "yes" if ans else "no"
        return str(ans)

    def get_baseline(self, ex: Dict[str, Any]) -> str:
        return ""


@dataclass
class JSONLAdapter:
    """Adapter for custom local JSONL datasets."""
    file_path: str

    def load(self, split=None):
        import json
        with open(self.file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def get_question(self, ex: Dict) -> str:
        return ex.get("question", "")

    def get_gold(self, ex: Dict) -> str:
        return ex.get("reference", "")

    def get_baseline(self, ex: Dict) -> str:
        return ""


__all__ = [
    "TruthfulQAAdapter",
    "StrategyQAAdapter",
    "JSONLAdapter",
]
