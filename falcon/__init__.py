"""
FALCON package public API.

This file re-exports the core interfaces and entrypoints so other modules
(and users) can import them consistently.
"""

from .llm import LLM, Generation, UnifiedScorer, NoLLM
from .models import NLIJudge, load_nli_judge, normalize_weight
from .solver import FalconSolver, SolveResult
from .pipeline import run_falcon_on_text, run_eval, FalconRunStats

# Adapters (explicit exports to avoid ambiguous imports)
from .adapters import TruthfulQAAdapter, StrategyQAAdapter, JSONLAdapter

__all__ = [
    # LLM
    "LLM",
    "Generation",
    "UnifiedScorer",
    "NoLLM",
    # NLI
    "NLIJudge",
    "load_nli_judge",
    "normalize_weight",
    # Solver
    "FalconSolver",
    "SolveResult",
    # Pipeline
    "run_falcon_on_text",
    "run_eval",
    "FalconRunStats",
    # Adapters
    "TruthfulQAAdapter",
    "StrategyQAAdapter",
    "JSONLAdapter",
]