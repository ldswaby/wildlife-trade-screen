"""Validate reusable responses and account for spending across the shared cache."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple, cast
from urllib.parse import urlsplit

from .config import ModelConfig
from .prompts import Prompt
from .types import Advert, EvaluationData, ScoreResult
from .utils import hash_json


class PendingCall(NamedTuple):
    """One advert and the durable reservation that must precede its request."""

    advert: Advert
    path: Path
    entry: dict[str, Any]

    @property
    def reservation(self) -> dict[str, Any]:
        """Return the request bounds prepared during cache inspection."""
        return cast(dict[str, Any], self.entry["reservation"])

    def prior_charge(self) -> float:
        """Recover an existing reservation when a local retry was explicitly authorized."""
        return cached_charge(json.loads(self.path.read_text())) if self.path.exists() else 0.0


def reservation(config: ModelConfig, prompt: Prompt, advert: Advert) -> dict[str, Any]:
    """Reserve conservative request costs before concurrent calls can spend money.

    Args:
        config: Arm pricing and API family; billable prices may be unknown in dry runs.
        prompt: Fixed criteria used to estimate input size.
        advert: Allowlisted text only.

    Returns:
        Upper token/cost estimates and external spend reserved. UTF-8 bytes plus
        framing are conservative estimates, not exact provider tokenisation.
        Unknown infrastructure cost is distinct from zero external API spend.
    """
    if config.adapter in {"fake", "precomputed"}:
        tokens_in = tokens_out = 0
    elif config.adapter == "chat":
        size = len(json.dumps(prompt.chat_messages(advert), ensure_ascii=False).encode())
        tokens_in = size + 2048
        tokens_out = int(
            config.decoding.get("max_completion_tokens", config.decoding.get("max_tokens", 1))
        )
    else:
        size = len((json.dumps(prompt.question()) + advert.text()).encode())
        tokens_in = size + 2048
        tokens_out = 0 if config.adapter == "openai_decisions" else 4096
    estimate = config.cost(tokens_in, tokens_out)
    # Local infrastructure cost remains unknown, while external API spend is absent.
    bound = estimate if config.billable else 0.0
    return {
        "tokens_in_upper_estimate": tokens_in,
        "tokens_out_upper_estimate": tokens_out,
        "cost_upper_estimate": estimate,
        "external_spend_reserved": bound,
    }


def cached_charge(entry: dict[str, Any]) -> float:
    """Retain a reservation when billed usage is unknown or a call is uncertain."""
    if entry["status"] == "cancelled_before_send":
        return 0.0
    result = entry.get("result", {})
    cost = result.get("cost")
    if entry.get("billable") and cost is not None:
        return float(cost)
    return float(entry["reservation"]["external_spend_reserved"] or 0)


def cache_spend(cache: Path) -> float:
    """Count historical charges and outstanding reservations across all cached arms."""
    return sum(
        cached_charge(json.loads(p.read_text(encoding="utf-8"))) for p in cache.glob("*/*/*.json")
    )


def prepare_cache(
    data: EvaluationData,
    config: ModelConfig,
    prompt: Prompt,
    cache: Path,
    config_hash: str,
) -> tuple[dict[str, ScoreResult], list[PendingCall]]:
    """Detect ambiguous cache reuse before starting any new calls.

    Args:
        data: Evaluation adverts; gold labels never enter cache input fingerprints.
        config: Arm definition used for pending-call reservations.
        prompt: Fixed prompt and its file hash.
        cache: Root shared by all arms under the lifetime spend ceiling.
        config_hash: Configuration/field/source fingerprint for this run.

    Returns:
        Cached results by advert ID and pending calls with reservation metadata.

    Raises:
        ValueError: Allowlisted text changed under an existing cache key.
        RuntimeError: A prior call was sent without a saved response.
    """
    results: dict[str, ScoreResult] = {}
    pending: list[PendingCall] = []
    for advert in data.adverts:
        path = cache / config_hash / prompt.sha256 / (hash_json(advert.advert_id) + ".json")
        cache_input = asdict(advert)
        # Historical inputs had exactly three fields; an unused new field must
        # not invalidate their saved responses or trigger a second paid request.
        if advert.title_and_desc is None:
            cache_input.pop("title_and_desc")
        input_hash = hash_json(cache_input)
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry["input_hash"] != input_hash:
                raise ValueError(
                    "Cached advert text changed under the same ID; use a new namespace"
                )
            if entry["status"] == "retry_authorized_local":
                authorization = entry.get("retry_authorization", {})
                previous = authorization.get("previous_entry", {})
                if (
                    config.billable
                    or entry.get("billable") is not False
                    or config.adapter != "system_one"
                    or urlsplit(config.endpoint or "").hostname
                    not in {"127.0.0.1", "::1", "localhost"}
                    or not authorization.get("authorized_at")
                    or not authorization.get("reason")
                    or previous.get("status") != "in_flight"
                    or previous.get("input_hash") != input_hash
                    or previous.get("model_config_hash") != config_hash
                    or previous.get("prompt_hash") != prompt.sha256
                ):
                    raise RuntimeError("Invalid local retry authorization; no calls sent")
            if entry["status"] in {"cancelled_before_send", "retry_authorized_local"}:
                entry["status"] = "in_flight"
                entry["reservation"] = reservation(config, prompt, advert)
                pending.append(PendingCall(advert, path, entry))
                continue
            if entry["status"] != "complete":
                raise RuntimeError(
                    "Uncertain prior call is cached; reconcile manually before proceeding"
                )
            results[advert.advert_id] = ScoreResult(**entry["result"])
        else:
            pending.append(
                PendingCall(
                    advert,
                    path,
                    {
                        "input_hash": input_hash,
                        "advert_id": advert.advert_id,
                        "model_config_hash": config_hash,
                        "prompt_hash": prompt.sha256,
                        "billable": config.billable,
                        "reservation": reservation(config, prompt, advert),
                        "status": "in_flight",
                    },
                )
            )
    return results, pending
