"""Reserve a place for pool decontamination in a future training study."""

from .types import Advert


def remove_test_and_near_duplicates(pool: list[Advert], test: list[Advert]) -> list[Advert]:
    """Reserve the future decontamination contract without implementing training.

    Args:
        pool: Candidate adverts for future pseudo-labelling or training.
        test: Held-out adverts whose identities and near-duplicates must be excluded.

    Returns:
        A decontaminated pool when the training study is implemented.

    Raises:
        NotImplementedError: Pool filtering is not implemented in this study.
    """
    raise NotImplementedError("Pool filtering and training are future work")
