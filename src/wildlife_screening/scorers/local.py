"""Offline scores exercise the same pipeline without reading evaluation labels."""

import csv
import hashlib
from pathlib import Path

from ..types import Advert, ScoreResult, ScoreType
from .base import Scorer


class FakeScorer(Scorer):
    """Exercise the pipeline deterministically without reading gold labels or making calls."""

    def __init__(self, seed: int = 0, binary: bool = False) -> None:
        """Configure synthetic scores independently of test labels.

        Args:
            seed: Salt for deterministic advert-ID hashing.
            binary: Emit only zero/one scores to exercise point-only plotting.
        """
        self.seed = seed
        self.binary = binary
        self.calls = 0

    def score(self, adverts: list[Advert]) -> list[ScoreResult]:
        """Return deterministic synthetic probabilities for an offline smoke test.

        Args:
            adverts: IDs used as hash inputs; advert text and gold labels are ignored.

        Returns:
            Aligned fake results with zero API tokens, cost and latency.
        """
        results: list[ScoreResult] = []
        for advert in adverts:
            self.calls += 1
            value = int(
                hashlib.sha256(f"{self.seed}:{advert.advert_id}".encode()).hexdigest()[:8], 16
            )
            p = value / 0xFFFFFFFF
            if self.binary:
                p = float(p >= 0.5)
            results.append(
                ScoreResult(
                    advert.advert_id,
                    p,
                    "binary" if self.binary else "native_prob",
                    f"synthetic score={p:.6f}",
                    0,
                    0,
                    0.0,
                    0.0,
                )
            )
        return results


class PrecomputedScorer(Scorer):
    """Import authors' probabilities through the same interface as measured arms."""

    def __init__(self, path: Path, score_type: ScoreType = "native_prob") -> None:
        """Validate imported probabilities before any evaluation uses them.

        Args:
            path: CSV containing unique advert_id and p_relevant columns.
            score_type: Provenance of the imported score, normally native_prob.

        Raises:
            ValueError: Headers, IDs or probability values are invalid.
        """
        self.values: dict[str, float] = {}
        self.score_type = score_type
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"advert_id", "p_relevant"} <= set(reader.fieldnames or []):
                raise ValueError("Precomputed CSV needs advert_id,p_relevant")
            for row in reader:
                advert_id = row["advert_id"]
                if not advert_id or advert_id in self.values:
                    raise ValueError("Empty or duplicate precomputed advert_id")
                value = float(row["p_relevant"])
                ScoreResult(advert_id, value, score_type, "", None, None, None, 0)
                self.values[advert_id] = value

    def score(self, adverts: list[Advert]) -> list[ScoreResult]:
        """Align imported scores by ID rather than trusting CSV row order.

        Args:
            adverts: Evaluation adverts in the desired output order.

        Returns:
            Imported scores with unknown original tokens/cost; latency is zero for import.

        Raises:
            ValueError: An evaluation advert has no imported probability.
        """
        missing = [a.advert_id for a in adverts if a.advert_id not in self.values]
        if missing:
            raise ValueError(f"Missing {len(missing)} precomputed advert IDs")
        return [
            ScoreResult(
                a.advert_id,
                self.values[a.advert_id],
                self.score_type,
                "imported probability",
                None,
                None,
                None,
                0,
            )
            for a in adverts
        ]
