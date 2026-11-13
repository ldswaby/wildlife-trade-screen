"""Configuration is evidence, including explicit unknown prices and deployment TODOs."""

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .utils import file_hash, hash_json


@dataclass(frozen=True)
class ModelConfig:
    """Describe an evaluation arm without storing authentication secrets.

    Prices are USD per million tokens; None means unknown. The billable flag
    tracks external API spend separately from unknown local infrastructure costs.
    Adapter-specific options remain flexible to support new API families.
    """

    arm: str
    model_id: str
    adapter: str
    endpoint: str | None = None
    api_key_env: str | None = None
    auth_header: str = "Authorization"
    price_input_per_million: float | None = None
    price_output_per_million: float | None = None
    currency: str = "USD"
    billable: bool = True
    decoding: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    verified_at: str | None = None
    max_input_bound_tokens: int = 16384

    def __post_init__(self) -> None:
        """Reject settings that would invalidate pricing or bounded Chat scoring.

        Raises:
            ValueError: Prices, currency or decoding settings violate the v1 contract.
        """
        if not self.arm or not self.model_id:
            raise ValueError("arm and model_id are required")
        for price in (self.price_input_per_million, self.price_output_per_million):
            if price is not None and (not isfinite(price) or price < 0):
                raise ValueError("Prices must be finite, nonnegative, or null")
        if self.currency != "USD":
            raise ValueError("v1 ceilings and list prices use USD; no implicit FX conversion")
        if (
            isinstance(self.max_input_bound_tokens, bool)
            or not isinstance(self.max_input_bound_tokens, int)
            or self.max_input_bound_tokens < 1
        ):
            raise ValueError("max_input_bound_tokens must be a positive integer")
        # Keep old logprob settings loadable for historical manifest validation.
        allowed = {
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
            "logprobs",
            "top_logprobs",
            "seed",
        }
        if self.adapter == "chat":
            limits = {"max_tokens", "max_completion_tokens"} & self.decoding.keys()
            if (
                not set(self.decoding) <= allowed
                or len(limits) > 1
                or any(
                    isinstance(self.decoding[key], bool)
                    or not isinstance(self.decoding[key], int)
                    or not 1 <= self.decoding[key] <= 32
                    for key in limits
                )
                or self.decoding.get("reasoning_effort", "none") != "none"
            ):
                raise ValueError("Chat permits bounded label decoding settings only")
        elif self.decoding:
            raise ValueError("Native/local adapters do not use generative decoding settings")

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        """Load an arm configuration from YAML, preserving explicit unknown values.

        Args:
            path: YAML file matching the ModelConfig fields.

        Returns:
            A validated arm configuration.
        """
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(**raw)

    def fingerprint(self, fields: tuple[str, ...]) -> str:
        """Include field consent and imported-score provenance in cache identity.

        Args:
            fields: Text fields explicitly enabled for this run.

        Returns:
            SHA-256 of the configuration, fields and precomputed file hash, if used.
        """
        from dataclasses import asdict

        value = asdict(self)
        value["fields"] = fields
        if self.adapter == "precomputed":
            value["precomputed_sha256"] = file_hash(Path(self.options["path"]))
        return hash_json(value)

    def validate_live(self) -> None:
        """Fail before dispatch when deployment or pricing information is unresolved.

        Raises:
            ValueError: Required endpoint, checkpoint or billable prices are unavailable.
        """
        if "TODO" in self.model_id:
            raise ValueError("Model ID is TODO; verify deployment before running")
        if self.adapter in {"chat", "openai_decisions", "system_one"}:
            if not self.endpoint or "TODO" in self.endpoint:
                raise ValueError("Endpoint is TODO; supply the verified endpoint")
            url = urlsplit(self.endpoint)
            if url.scheme not in {"http", "https"} or url.username or url.password or url.query:
                raise ValueError("Endpoint must be an HTTP(S) URL without credentials or query")
            if url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Remote APIs require HTTPS")
        if self.adapter == "perplexity_local":
            checkpoint = self.options.get("checkpoint", "TODO")
            if "TODO" in checkpoint or not Path(checkpoint).is_dir():
                raise ValueError("TODO: provision a local Perplexity checkpoint before running")
        if self.billable and (
            self.price_input_per_million is None or self.price_output_per_million is None
        ):
            raise ValueError("Unknown price: spend ceiling cannot be enforced; verify prices first")

    def cost(self, tokens_in: int | None, tokens_out: int | None) -> float | None:
        """Estimate list-price cost without treating unknown usage or prices as zero.

        Args:
            tokens_in: Reported input tokens, or None if unavailable.
            tokens_out: Reported output tokens, or None if unavailable.

        Returns:
            Cost in USD, or None if any required quantity is unknown.
        """
        if (
            tokens_in is None
            or tokens_out is None
            or self.price_input_per_million is None
            or self.price_output_per_million is None
        ):
            return None
        return (
            tokens_in * self.price_input_per_million + tokens_out * self.price_output_per_million
        ) / 1_000_000
