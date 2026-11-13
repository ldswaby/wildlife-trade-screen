"""Strict parsers preserve uncertainty instead of silently fabricating probabilities."""

import json
import math
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from ..config import ModelConfig
from ..prompts import Prompt
from ..types import Advert, ScoreResult, ScoreType
from .base import Scorer


class RequestNotSent(RuntimeError):
    """Signal that request assembly stopped before the HTTP dispatch boundary."""


def probability_distribution(values: dict[str, float]) -> float:
    """Read relevance from the full labelled distribution, never winning confidence.

    Args:
        values: Probabilities for exactly relevant and not relevant.

    Returns:
        P(relevant) without recalibration or renormalisation.

    Raises:
        ValueError: Keys, values or total probability mass are invalid.
    """
    if set(values) != {"relevant", "not relevant"}:
        raise ValueError("Expected exactly two labelled probabilities")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("Invalid probability distribution")
    if not math.isclose(sum(values.values()), 1, abs_tol=1e-4):
        raise ValueError("Probabilities do not sum to one")
    return values["relevant"]


def token_count(value: Any) -> int | None:
    """Preserve invalid or missing provider usage as unknown instead of coercing it."""
    return value if type(value) is int and value >= 0 else None


class APIScorer(Scorer):
    """Share HTTP accounting and explicit failures while keeping protocols in adapters."""

    error_score_type: ScoreType = "native_prob"

    def __init__(
        self, config: ModelConfig, prompt: Prompt, client: httpx.Client | None = None
    ) -> None:
        """Configure transport without issuing an API request.

        Args:
            config: Endpoint, credential environment variable and pricing settings.
            prompt: Shared fixed criteria.
            client: Optional injected HTTP client, used for offline protocol tests.
        """
        self.config = config
        self.prompt = prompt
        self.client = client or httpx.Client(
            timeout=float(config.options.get("timeout_s", 60)), follow_redirects=False
        )

    def payload(self, advert: Advert) -> dict[str, Any]:
        """Define the request-building hook for a specific API family.

        Args:
            advert: Only the ID and allowlisted text are available.

        Returns:
            A JSON request body containing approved text and fixed criteria.

        Raises:
            NotImplementedError: A concrete adapter must implement this hook.
        """
        raise NotImplementedError

    def parse(self, response: dict[str, Any]) -> tuple[float, ScoreType]:
        """Define the probability-parsing hook without trusting a confidence field.

        Args:
            response: Decoded provider JSON; malformed answers must fail explicitly.

        Returns:
            P(relevant) and the score provenance.

        Raises:
            NotImplementedError: A concrete adapter must implement this hook.
        """
        raise NotImplementedError

    def headers(self) -> dict[str, str]:
        """Resolve credentials at call time so keys never enter configs or cache hashes.

        Returns:
            Content-type and optional authentication headers.

        Raises:
            ValueError: A configured credential environment variable is unset.
        """
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if not key:
                raise ValueError(f"Missing {self.config.api_key_env}")
            headers[self.config.auth_header] = (
                f"Bearer {key}" if self.config.auth_header == "Authorization" else key
            )
        return headers

    def score(self, adverts: list[Advert]) -> list[ScoreResult]:
        """Retain raw responses and accounting for successes and per-advert failures.

        Args:
            adverts: Allowlisted adverts; calls are sequential within this invocation.

        Returns:
            Aligned results. Refusals and transport/parsing failures retain an error,
            available usage and private raw output, without automatic retries.

        Raises:
            ValueError: Deployment, pricing or credential preflight fails before dispatch.
        """
        self.config.validate_live()
        headers = self.headers()  # Fail before attempting a chargeable call.
        results: list[ScoreResult] = []
        for advert in adverts:
            started = time.perf_counter()
            raw = ""
            tokens_in = tokens_out = None
            cost = None
            try:
                payload = self.payload(advert)
            except BaseException as exc:
                # A debugger exit here cannot have reached the provider.
                raise RequestNotSent("Request assembly stopped before dispatch") from exc
            try:
                response = self.client.post(
                    self.config.endpoint or "",
                    headers=headers,
                    json=payload,
                )
                raw = response.text
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("Invalid response envelope")
                usage = body.get("usage", {})
                if not isinstance(usage, dict):
                    raise ValueError("Invalid usage envelope")
                tokens_in = token_count(usage.get("input_tokens", usage.get("prompt_tokens")))
                tokens_out = token_count(usage.get("output_tokens", usage.get("completion_tokens")))
                cost = self.config.cost(tokens_in, tokens_out)
                p, kind = self.parse(body)
                result = ScoreResult(
                    advert.advert_id,
                    p,
                    kind,
                    raw,
                    tokens_in,
                    tokens_out,
                    cost,
                    time.perf_counter() - started,
                )
            except (
                httpx.HTTPError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                AttributeError,
            ) as exc:
                # Error text must not echo headers, keys, URLs or advert content.
                message = (
                    f"{type(exc).__name__}: response unavailable or invalid; "
                    "inspect private raw_output"
                )
                result = ScoreResult(
                    advert.advert_id,
                    None,
                    self.error_score_type,
                    raw,
                    tokens_in,
                    tokens_out,
                    cost,
                    time.perf_counter() - started,
                    message,
                )
            results.append(result)
        return results


class PromptedLLMScorer(APIScorer):
    """Score exact relevant/not relevant Chat labels as binary results."""

    error_score_type: ScoreType = "binary"

    def payload(self, advert: Advert) -> dict[str, Any]:
        """Bound Chat output while sending only the allowlisted advert text.

        Args:
            advert: Allowlisted text sent separately from the fixed instructions.

        Returns:
            An OpenAI-compatible Chat Completions request.
        """
        settings = {"temperature": 0}
        settings.update(self.config.decoding)
        if "logprobs" in settings or "top_logprobs" in settings:
            raise ValueError("Binary Chat scoring does not request token logprobs")
        if "max_completion_tokens" not in settings:
            settings.setdefault("max_tokens", 32)
        limits = {"max_tokens", "max_completion_tokens"} & settings.keys()
        if len(limits) != 1 or any(
            isinstance(settings[key], bool)
            or not isinstance(settings[key], int)
            or not 1 <= settings[key] <= 32
            for key in limits
        ):
            raise ValueError("Binary Chat scoring requires a bounded completion")

        return {
            "model": self.config.model_id,
            **settings,
            "messages": self.prompt.chat_messages(advert),
        }

    def parse(self, response: dict[str, Any]) -> tuple[float, ScoreType]:
        """Map the exact chosen label to a binary decision, without confidence.

        Args:
            response: Chat completion containing an exact relevance label.

        Returns:
            One for relevant, zero for not relevant.

        Raises:
            ValueError: Output is not exactly one non-refused relevance label.
        """
        choices = response["choices"]
        if len(choices) != 1:
            raise ValueError("Expected one Chat choice")
        choice = choices[0]
        raw_label = choice["message"].get("content")
        if raw_label not in {"relevant", "not relevant"} or choice["message"].get("refusal"):
            raise ValueError("Response is not an exact relevant/not relevant label")
        return float(raw_label == "relevant"), "binary"


class OpenAIDecisionsScorer(APIScorer):
    """Adapt fixed choices to the Decisions API's native probability array."""

    def payload(self, advert: Advert) -> dict[str, Any]:
        """Express the shared criteria as a native named choice question.

        Args:
            advert: Allowlisted text used as the input evidence.

        Returns:
            A Decisions request with no generative decoding settings.
        """
        if self.config.decoding:
            raise ValueError("Decisions API has no generative decoding settings")
        question = self.prompt.question()
        return {
            "model": self.config.model_id,
            "input": advert.text(),
            "questions": [
                {
                    "name": "relevance",
                    "type": "choice",
                    "instructions": question["instructions"],
                    "choices": [
                        {"value": name, "description": description}
                        for name, description in question["criteria"].items()
                    ],
                }
            ],
        }

    def parse(self, response: dict[str, Any]) -> tuple[float, ScoreType]:
        """Require one named choice answer and the full two-label distribution.

        Args:
            response: Native Decisions response with an answers array.

        Returns:
            P(relevant) tagged as native_prob.

        Raises:
            ValueError: The answer is a refusal or its probabilities are invalid.
        """
        answers = response["answers"]
        if len(answers) != 1 or answers[0].get("name") != "relevance":
            raise ValueError("Expected one named answer")
        answer = answers[0]
        if answer.get("type") != "choice":
            raise ValueError("Refusal or unsupported answer")
        pairs = answer["probabilities"]
        if len(pairs) != 2:
            raise ValueError("Expected two probabilities")
        return (
            probability_distribution({p["value"]: float(p["probability"]) for p in pairs}),
            "native_prob",
        )


class SystemOneScorer(APIScorer):
    """Use the documented choice protocol shared by Jev, Microsoft Foundry and d3."""

    def payload(self, advert: Advert) -> dict[str, Any]:
        """Keep System One question IDs stable for unambiguous response parsing.

        Args:
            advert: Allowlisted text used as state evidence.

        Returns:
            A System One request with the fixed relevance question.
        """
        if self.config.decoding:
            raise ValueError("System One does not use generative decoding settings")
        return {
            "model": self.config.model_id,
            "state": advert.text(),
            "questions": {"relevance": self.prompt.question()},
        }

    def parse(self, response: dict[str, Any]) -> tuple[float, ScoreType]:
        """Extract the labelled distribution instead of the selected choice's confidence.

        Args:
            response: System One response with an answers mapping.

        Returns:
            P(relevant) tagged as native_prob.

        Raises:
            ValueError: The native answer is refused, erroneous or invalid.
        """
        answer = response["answers"]["relevance"]
        if answer.get("type") != "choice" or answer.get("error"):
            raise ValueError("Refusal or invalid native answer")
        return (
            probability_distribution({k: float(v) for k, v in answer["probabilities"].items()}),
            "native_prob",
        )


class PerplexityLocalScorer(Scorer):
    """Use the evaluated autojev readout; causal chat would alter this checkpoint."""

    def __init__(
        self,
        config: ModelConfig,
        prompt: Prompt,
        predict: Callable[[list[dict[str, Any]]], list[list[float]]] | None = None,
    ) -> None:
        """Defer GPU loading until scoring and allow a small offline predictor in tests.

        Args:
            config: Checkpoint location, device and model provenance.
            prompt: Fixed native choice criteria.
            predict: Optional predictor returning probability vectors in criteria order.
        """
        self.config = config
        self.prompt = prompt
        self.predict = predict

    def score(self, adverts: list[Advert]) -> list[ScoreResult]:
        """Use native probabilities without downloading weights or adjusting calibration.

        Args:
            adverts: Allowlisted evidence evaluated by the separately provisioned model.

        Returns:
            Aligned native results; invalid distributions retain raw output and an error.
            Usage and infrastructure cost remain unknown.

        Raises:
            ValueError: The local checkpoint has not been provisioned.
        """
        if self.predict is None:
            checkpoint = self.config.options.get("checkpoint", "TODO")
            if "TODO" in checkpoint:
                raise ValueError(
                    "TODO: provision the pinned Perplexity checkpoint and autojev package"
                )
            # Caller installs the model's own pinned environment; do not download weights here.
            from autojev.model import DecisionModel  # type: ignore[import-not-found]

            model = DecisionModel(checkpoint, device=self.config.options.get("device", "cuda"))
            self.predict = model.predict
        results: list[ScoreResult] = []
        for advert in adverts:
            started = time.perf_counter()
            question = self.prompt.question()
            values = self.predict([{"state": advert.text(), "question": question}])[0]
            raw = json.dumps(values)
            p = None
            error = None
            try:
                p = probability_distribution(dict(zip(question["criteria"], values, strict=True)))
            except (ValueError, TypeError) as exc:
                error = f"{type(exc).__name__}: invalid native distribution"
            results.append(
                ScoreResult(
                    advert.advert_id,
                    p,
                    "native_prob",
                    raw,
                    None,
                    None,
                    None,
                    time.perf_counter() - started,
                    error,
                )
            )
        return results
