"""Adapters are selected in YAML, leaving the runner independent of API families."""

from pathlib import Path
from typing import cast

from ..config import ModelConfig
from ..prompts import Prompt
from ..types import ScoreType
from .api import OpenAIDecisionsScorer, PerplexityLocalScorer, PromptedLLMScorer, SystemOneScorer
from .base import Scorer, TrainedClassifierScorer
from .local import FakeScorer, PrecomputedScorer

__all__ = [
    "Scorer",
    "FakeScorer",
    "PrecomputedScorer",
    "PromptedLLMScorer",
    "TrainedClassifierScorer",
    "make_scorer",
]


def make_scorer(config: ModelConfig, prompt: Prompt) -> Scorer:
    """Select an adapter without coupling evaluation code to an API family.

    Args:
        config: Arm definition and adapter-specific options.
        prompt: Shared fixed criteria for prompted/native decision adapters.

    Returns:
        The configured scorer; local models are loaded lazily.

    Raises:
        ValueError: The adapter name or imported score file is invalid.
    """
    if config.adapter == "fake":
        return FakeScorer(
            int(config.options.get("seed", 0)), bool(config.options.get("binary", False))
        )
    if config.adapter == "precomputed":
        return PrecomputedScorer(
            Path(config.options["path"]),
            cast(ScoreType, config.options.get("score_type", "native_prob")),
        )
    if config.adapter == "chat":
        return PromptedLLMScorer(config, prompt)
    if config.adapter == "openai_decisions":
        return OpenAIDecisionsScorer(config, prompt)
    if config.adapter == "system_one":
        return SystemOneScorer(config, prompt)
    if config.adapter == "perplexity_local":
        return PerplexityLocalScorer(config, prompt)
    if config.adapter == "trained_classifier":
        return TrainedClassifierScorer()
    raise ValueError(f"Unknown adapter {config.adapter}")
