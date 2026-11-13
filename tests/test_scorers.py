from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from wildlife_screening.config import ModelConfig
from wildlife_screening.prompts import Prompt
from wildlife_screening.scorers.api import (
    APIScorer,
    OpenAIDecisionsScorer,
    PerplexityLocalScorer,
    PromptedLLMScorer,
    SystemOneScorer,
)
from wildlife_screening.scorers.local import PrecomputedScorer
from wildlife_screening.types import Advert, EvaluationData, ScoreResult


def config(adapter: str = "chat") -> ModelConfig:
    return ModelConfig(
        "test",
        "test",
        adapter,
        endpoint="https://example.test/api",
        price_input_per_million=1,
        price_output_per_million=2,
    )


def chat_response(logs: bool = True) -> dict[str, Any]:
    choice: dict[str, Any] = {"message": {"content": "relevant"}}
    if logs:
        choice["logprobs"] = {
            "content": [
                {
                    "token": "relevant",
                    "logprob": math.log(0.6),
                    "top_logprobs": [
                        {"token": "relevant", "logprob": math.log(0.6)},
                        {"token": " not relevant", "logprob": math.log(0.2)},
                    ],
                }
            ]
        }
    return {"choices": [choice], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}


def test_chat_binary_and_privacy(prompt: Prompt, data: EvaluationData) -> None:
    prompt = Prompt.load(Path(__file__).resolve().parents[1] / "prompts/sharks/llm_zeroshot_v0.txt")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response(False))

    scorer = PromptedLLMScorer(
        config(), prompt, httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = scorer.score(data.adverts[:1])[0]
    assert result.p_relevant == 1.0
    assert result.score_type == "binary"
    assert result.cost == pytest.approx(12 / 1e6)
    assert "DO_NOT_SEND" not in json.dumps(requests)
    assert requests[0]["messages"] == prompt.chat_messages(data.adverts[0])
    assert [message["role"] for message in requests[0]["messages"]] == ["developer", "user"]
    assert "description:" not in requests[0]["messages"][1]["content"]


def test_luna_chat_anchor_request_and_exact_labels(
    monkeypatch: pytest.MonkeyPatch,
    data: EvaluationData,
) -> None:
    root = Path(__file__).resolve().parents[1]
    anchor = ModelConfig.load(root / "configs/models/gpt6_luna_chat_anchor.yaml")
    prompt = Prompt.load(root / "prompts/sharks/llm_zeroshot_v0.txt")
    advert = data.adverts[0]
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response(False))

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    scorer = PromptedLLMScorer(anchor, prompt, httpx.Client(transport=httpx.MockTransport(handler)))
    result = scorer.score([advert])[0]
    assert result.p_relevant == 1.0
    assert result.score_type == "binary"
    assert result.cost == pytest.approx(1.5 / 1e6)
    assert requests == [
        {
            "model": "gpt-6-luna",
            "reasoning_effort": "none",
            "temperature": 0,
            "max_completion_tokens": 32,
            "messages": prompt.chat_messages(advert),
        }
    ]
    assert "max_tokens" not in requests[0]
    assert "DO_NOT_SEND" not in json.dumps(requests)
    assert "Synthetic description" not in json.dumps(requests)
    assert advert.advert_id not in json.dumps(requests)

    assert scorer.parse({"choices": [{"message": {"content": "not relevant"}}]}) == (
        0.0,
        "binary",
    )
    for invalid in ("", "A", "B", " relevant", "relevant ", "Relevant", "not_relevant"):
        response = {"choices": [{"message": {"content": invalid}}]}
        with pytest.raises(ValueError, match="exact relevant/not relevant"):
            scorer.parse(response)
        failed_client = httpx.Client(
            transport=httpx.MockTransport(lambda _, body=response: httpx.Response(200, json=body))
        )
        failed = PromptedLLMScorer(anchor, prompt, failed_client).score([advert])[0]
        assert failed.error and failed.p_relevant is None
        assert failed.score_type == "binary"
        assert json.loads(failed.raw_output) == response


def test_binary_chat_ignores_returned_logprobs_but_does_not_request_them(prompt: Prompt) -> None:
    prompt = Prompt.load(Path(__file__).resolve().parents[1] / "prompts/sharks/llm_zeroshot_v0.txt")
    scorer = PromptedLLMScorer(config(), prompt)
    assert scorer.parse(chat_response(False)) == (1.0, "binary")
    assert scorer.parse(chat_response()) == (1.0, "binary")
    with pytest.raises(ValueError, match="does not request"):
        PromptedLLMScorer(replace(config(), decoding={"logprobs": True}), prompt).payload(
            Advert("x", "synthetic")
        )
    with pytest.raises(ValueError, match="exact relevant/not relevant"):
        scorer.parse({"choices": [{"message": {"content": "relevant", "refusal": "no"}}]})


def test_binary_anchor_uses_new_cache_identity() -> None:
    anchor = ModelConfig.load(
        Path(__file__).resolve().parents[1] / "configs/models/gpt6_luna_chat_anchor.yaml"
    )
    old = replace(
        anchor,
        decoding={**anchor.decoding, "logprobs": True, "top_logprobs": 5},
        options={**anchor.options, "missing_logprobs": "error"},
    )
    assert anchor.fingerprint(("title",)) != old.fingerprint(("title",))


@pytest.mark.parametrize(
    "adapter,cls", [("system_one", SystemOneScorer), ("openai_decisions", OpenAIDecisionsScorer)]
)
def test_native_choice(adapter: str, cls: type[APIScorer], prompt: Prompt) -> None:
    scorer = cls(config(adapter), prompt)
    p = scorer.payload(Advert("x", "synthetic"))
    response: dict[str, Any]
    if adapter == "system_one":
        assert set(p["questions"]["relevance"]["criteria"]) == {"relevant", "not relevant"}
        response = {
            "answers": {
                "relevance": {
                    "type": "choice",
                    "probabilities": {"relevant": 0.2, "not relevant": 0.8},
                }
            }
        }
    else:
        assert p["questions"][0]["name"] == "relevance"
        response = {
            "answers": [
                {
                    "name": "relevance",
                    "type": "choice",
                    "probabilities": [
                        {"value": "relevant", "probability": 0.2},
                        {"value": "not relevant", "probability": 0.8},
                    ],
                }
            ]
        }
    assert scorer.parse(response) == (0.2, "native_prob")


def test_invalid_native_probabilities(prompt: Prompt) -> None:
    scorer = SystemOneScorer(config("system_one"), prompt)
    for probabilities in (
        {"relevant": 0.1},
        {"relevant": 0.5, "not relevant": 0.7},
        {"relevant": math.nan, "not relevant": 0.5},
    ):
        with pytest.raises(ValueError):
            scorer.parse(
                {"answers": {"relevance": {"type": "choice", "probabilities": probabilities}}}
            )


def test_raw_error_usage_preserved(prompt: Prompt) -> None:
    response = {
        "answers": [{"type": "refusal", "name": "relevance"}],
        "usage": {"input_tokens": 9, "output_tokens": 0},
    }
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response))
    )
    result = OpenAIDecisionsScorer(config("openai_decisions"), prompt, client).score(
        [Advert("x", "synthetic")]
    )[0]
    assert result.error
    assert result.p_relevant is None
    assert json.loads(result.raw_output) == response
    assert result.tokens_in == 9
    assert result.cost == pytest.approx(9 / 1e6)


def test_precomputed_validation(tmp_path: Path) -> None:
    p = tmp_path / "scores.csv"
    p.write_text("advert_id,p_relevant\nx,0.4\ny,0.8\n")
    scorer = PrecomputedScorer(p)
    assert scorer.score([Advert("y", "z"), Advert("x", "a")])[0].p_relevant == 0.8
    assert scorer.score([Advert("x", "z")])[0].cost is None
    with pytest.raises(ValueError, match="Missing"):
        scorer.score([Advert("missing", "z")])
    for text in ("advert_id,p_relevant\nx,nan\n", "advert_id,p_relevant\nx,0.4\nx,0.5\n"):
        p.write_text(text)
        with pytest.raises(ValueError):
            PrecomputedScorer(p)


def test_perplexity_native_readout(prompt: Prompt) -> None:
    requests = []

    def predict(rows: list[dict[str, Any]]) -> list[list[float]]:
        requests.extend(rows)
        return [[0.7, 0.3]]

    scorer = PerplexityLocalScorer(
        replace(config("perplexity_local"), billable=False), prompt, predict
    )
    result = scorer.score([Advert("x", "synthetic")])[0]
    assert result.p_relevant == 0.7
    assert result.cost is None and result.tokens_in is None
    assert set(requests[0]) == {"state", "question"}


@pytest.mark.parametrize("p", [math.nan, math.inf, -0.1, 1.1])
def test_score_contract(p: float) -> None:
    with pytest.raises(ValueError):
        ScoreResult("x", p, "native_prob", "", 0, 0, 0, 0)


def test_all_candidate_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs/models").glob("*.yaml"):
        ModelConfig.load(path)


def test_decoding_cannot_override_model_or_output_bound() -> None:
    settings: dict[str, Any]
    for settings in (
        {"model": "other"},
        {"max_tokens": 100},
        {"max_completion_tokens": 100},
        {"max_tokens": 1, "max_completion_tokens": 1},
        {"reasoning_effort": "low"},
        {"tools": []},
    ):
        with pytest.raises(ValueError, match="bounded"):
            replace(config(), decoding=settings)


@pytest.mark.parametrize(
    "body", [[], {"usage": None}, {"usage": {"input_tokens": "bad"}, "answers": []}]
)
def test_malformed_envelopes_preserve_raw(prompt: Prompt, body: Any) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    result = OpenAIDecisionsScorer(config("openai_decisions"), prompt, client).score(
        [Advert("x", "synthetic")]
    )[0]
    assert result.error and json.loads(result.raw_output) == body
    assert result.cost is None


def test_prompt_criteria_retain_source_policy(prompt: Prompt) -> None:
    root = Path(__file__).resolve().parents[1]
    original = (root / "prompts/sharks/lts_figure3_full_source.txt").read_text()

    def normalise(s: str) -> str:
        return " ".join(s.split())

    positive = prompt.definition["criteria"]["relevant"]
    negative = prompt.definition["criteria"]["not relevant"]
    assert normalise(positive) in normalise(original)
    assert normalise(negative.split("For products such as teeth", 1)[0]) in normalise(original)
    assert "only one tooth" in negative and "more than one tooth" in negative
    assert "fossil" in negative
    assert "Examples:" not in prompt.question()["instructions"]


def test_native_confidence_is_not_probability(prompt: Prompt) -> None:
    scorer = SystemOneScorer(config("system_one"), prompt)
    answer = {
        "type": "choice",
        "confidence": 0.99,
        "choice": "not relevant",
        "probabilities": {"relevant": 0.1, "not relevant": 0.9},
    }
    assert scorer.parse({"answers": {"relevance": answer}}) == (0.1, "native_prob")
