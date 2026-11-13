"""Keep gold labels outside the scorer contract to prevent accidental leakage."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

ScoreType = Literal["logprob", "native_prob", "verbalised", "binary"]


@dataclass(frozen=True)
class Advert:
    """Restrict scorer inputs to identity and consented text, excluding gold labels.

    Attributes:
        advert_id: Stable local identity; not sent as evidence to the API.
        title: Default allowlisted text.
        description: Optional text populated only after explicit opt-in.
        title_and_desc: Optional combined text, selected instead of separate fields.
    """

    advert_id: str
    title: str = ""
    description: str | None = None
    title_and_desc: str | None = None

    def __post_init__(self) -> None:
        """Prevent combined evidence from silently including separate text fields."""
        if self.title_and_desc is not None and (self.title or self.description is not None):
            raise ValueError("Combined text cannot be mixed with title or description")

    def evidence(self) -> str:
        """Return selected advert text for local demonstration-overlap checks."""
        return self.title_and_desc if self.title_and_desc is not None else self.title

    def text(self) -> str:
        """Serialise enabled text fields without including identifiers or gold labels."""
        if self.title_and_desc is not None:
            return f"advert text: {self.title_and_desc}"
        return f"title: {self.title}" + (
            f"\ndescription: {self.description}" if self.description is not None else ""
        )


@dataclass(frozen=True)
class EvaluationData:
    """Keep gold labels locally alongside their aligned, label-free scorer inputs.

    Attributes:
        adverts: Inputs in original evaluation order.
        labels: Aligned binary gold labels, with one meaning relevant.
        sha256: Hash of the original CSV bytes for provenance.
        positive_label: Resolved numeric polarity when loaded from a CSV.
    """

    adverts: list[Advert]
    labels: list[int]
    sha256: str
    positive_label: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Preserve one advert's score and resource evidence, including explicit uncertainty.

    Attributes:
        advert_id: Identity matching the corresponding scorer input.
        p_relevant: Probability or binary decision; None requires an error.
        score_type: logprob, native_prob, verbalised or binary provenance.
        raw_output: Private provider response, which may repeat advert text.
        tokens_in: Input tokens, or None if unknown.
        tokens_out: Output tokens, or None if unknown.
        cost: Configured list-price cost in USD, or None if unknown.
        latency_s: Recorded scoring latency in seconds; import/fake adapters use zero.
        error: Failure description, or None for a valid score.
    """

    advert_id: str
    p_relevant: float | None
    score_type: ScoreType
    raw_output: str
    tokens_in: int | None
    tokens_out: int | None
    cost: float | None
    latency_s: float
    error: str | None = None

    def __post_init__(self) -> None:
        """Prevent invalid probabilities or fabricated resource values entering reports.

        Raises:
            ValueError: Score provenance, probabilities, usage, cost or latency are invalid.
        """
        if self.score_type not in ("logprob", "native_prob", "verbalised", "binary"):
            raise ValueError("Invalid score_type")
        if self.p_relevant is not None and (
            not isfinite(self.p_relevant) or not 0 <= self.p_relevant <= 1
        ):
            raise ValueError("p_relevant must be finite and in [0, 1]")
        if self.error is None and self.p_relevant is None:
            raise ValueError("A missing probability requires an error")
        if self.score_type == "binary" and self.p_relevant not in (None, 0.0, 1.0):
            raise ValueError("Binary scores must be 0 or 1")
        for value in (self.tokens_in, self.tokens_out):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token counts must be nonnegative integers or unknown")
        for measured in (self.cost, self.latency_s):
            if measured is not None and (not isfinite(measured) or measured < 0):
                raise ValueError("Cost and latency must be nonnegative and finite")
