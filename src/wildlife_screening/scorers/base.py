"""The single contract future classifiers must also implement."""

from abc import ABC, abstractmethod

from ..types import Advert, ScoreResult


class Scorer(ABC):
    """Define the label-free scoring contract shared by all present and future arms."""

    @abstractmethod
    def score(self, adverts: list[Advert]) -> list[ScoreResult]:
        """Score adverts without access to their gold labels.

        Args:
            adverts: IDs and allowlisted text in evaluation order.

        Returns:
            One ScoreResult per input in the same order and with the same ID.
            Per-advert failures must be explicit rather than silently dropping rows.
        """


class TrainedClassifierScorer(Scorer):
    """Reserve a future classifier interface without adding training code."""

    def score(self, adverts: list[Advert]) -> list[ScoreResult]:
        """Keep future trained arms compatible with the shared scorer contract.

        Args:
            adverts: IDs and allowlisted text to score after training is implemented.

        Returns:
            Aligned scores when the future implementation exists.

        Raises:
            NotImplementedError: Trained classifiers are not implemented in this study.
        """
        raise NotImplementedError("Trained classifiers are future work")
