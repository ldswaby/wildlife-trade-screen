"""Report both conventions with a cutoff fixed before examining test scores."""

import math
from collections.abc import Iterator, Sequence
from typing import Any

from .types import ScoreResult

EVALUATION_POLICY = "review_on_failure"
FINISHED_WITH_FAILURES = "finished with failures"


def balanced_accuracy(recall: float | None, fpr: float | None) -> float | None:
    """Average class rates only when both rates are available."""
    return (recall + 1 - fpr) / 2 if recall is not None and fpr is not None else None


def ratio(numerator: int, denominator: int) -> float:
    """Use zero for undefined precision/F1 ratios to match the sklearn convention."""
    return numerator / denominator if denominator else 0.0


def recall_ci(tp: int, positives: int) -> tuple[float | None, float | None]:
    """Use a Wilson interval that remains stable near zero and perfect recall.

    Args:
        tp: Correctly retrieved positives, between zero and positives.
        positives: Total gold positives; zero means recall is undefined.

    Returns:
        Lower and upper 95% limits, or (None, None) when there are no positives.
    """
    if positives == 0:
        return None, None
    z = 1.959963984540054
    p = tp / positives
    denominator = 1 + z * z / positives
    centre = (p + z * z / (2 * positives)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / positives + z * z / (4 * positives * positives)) / denominator
    )
    return max(0, centre - half), min(1, centre + half)


def evaluate(labels: Sequence[int], scores: Sequence[float], cutoff: float = 0.5) -> dict[str, Any]:
    """Report both averaging conventions at a predeclared cutoff.

    Args:
        labels: Nonempty binary gold labels, with one meaning relevant.
        scores: Aligned probabilities in [0, 1].
        cutoff: Flag scores greater than or equal to this value; the study uses 0.5.

    Returns:
        Positive/weighted metrics, confusion counts, missed positives, workload,
        false-positive rate, balanced accuracy and a Wilson recall interval. Recall is None when no
        positives exist; false-positive rate is None when no negatives exist.

    Raises:
        ValueError: Inputs are empty, misaligned, non-finite or out of range.
    """
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Cutoff must be finite and in [0, 1]")
    if not labels or len(labels) != len(scores) or any(y not in (0, 1) for y in labels):
        raise ValueError("Metrics require aligned, nonempty binary labels and scores")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Metrics require finite probabilities in [0,1]")
    predictions = [int(p >= cutoff) for p in scores]
    tp = sum(y == 1 and pred == 1 for y, pred in zip(labels, predictions, strict=True))
    fp = sum(y == 0 and pred == 1 for y, pred in zip(labels, predictions, strict=True))
    fn = sum(y == 1 and pred == 0 for y, pred in zip(labels, predictions, strict=True))
    tn = len(labels) - tp - fp - fn
    p, r, f = ratio(tp, tp + fp), ratio(tp, tp + fn), ratio(2 * tp, 2 * tp + fp + fn)
    pn, rn, fn_score = ratio(tn, tn + fn), ratio(tn, tn + fp), ratio(2 * tn, 2 * tn + fp + fn)
    positives, negatives = tp + fn, tn + fp
    low, high = recall_ci(tp, positives)
    n = len(labels)
    return {
        "positive_precision": p,
        "positive_recall": r if positives else None,
        "positive_f1": f,
        "weighted_precision": (p * positives + pn * negatives) / n,
        "weighted_recall": (r * positives + rn * negatives) / n,
        "weighted_f1": (f * positives + fn_score * negatives) / n,
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": balanced_accuracy(
            r if positives else None, fp / negatives if negatives else None
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positives": positives,
        "n": n,
        "missed": f"missed {fn} of {positives} positives",
        "recall_ci_low": low,
        "recall_ci_high": high,
        "false_positive_rate": fp / negatives if negatives else None,
        "workload": (tp + fp) / n,
        "cutoff": cutoff,
    }


def _review_counts(
    labels: Sequence[int], scores: Sequence[float | None]
) -> Iterator[tuple[int, int]]:
    """Yield cumulative (true positives, false positives) for validated inputs.

    A missing score here denotes an explicit failure already routed for review.
    Include that starting point even when no requests failed. Each score group
    enters together so a cutoff can never separate tied predictions.
    """
    tp = fp = 0
    groups: dict[float, list[int]] = {}
    for label, score in zip(labels, scores, strict=True):
        if score is None:
            tp += label
            fp += 1 - label
        else:
            groups.setdefault(score, [0, 0])[label] += 1
    yield tp, fp
    for score in sorted(groups, reverse=True):
        negatives, positives = groups[score]
        tp += positives
        fp += negatives
        yield tp, fp


def _curve(
    labels: Sequence[int], scores: Sequence[float | None], *, roc: bool
) -> list[tuple[float, float]]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives:
        raise ValueError(f"{'ROC' if roc else 'Recall'} curve undefined with no positives")
    if roc and not negatives:
        raise ValueError("ROC curve undefined with no negatives")
    return [
        (fp / negatives if roc else (tp + fp) / len(labels), tp / positives)
        for tp, fp in _review_counts(labels, scores)
    ]


def workload_curve(labels: Sequence[int], scores: Sequence[float]) -> list[tuple[float, float]]:
    """Sweep descriptive cutoffs without selecting an operating point.

    Args:
        labels: Binary gold labels containing at least one positive.
        scores: Aligned probabilities; tied scores are reviewed together.

    Returns:
        Workload/recall fractions at every unique cutoff, including no/all review.

    Raises:
        ValueError: Inputs are invalid or recall is undefined.
    """
    evaluate(labels, scores)
    return _curve(labels, scores, roc=False)


def roc_curve(labels: Sequence[int], scores: Sequence[float]) -> list[tuple[float, float]]:
    """Sweep descriptive cutoffs into false-positive-rate/recall coordinates.

    Tied scores enter together. Both gold classes are required because recall is
    undefined without positives and false-positive rate is undefined without negatives.
    """
    evaluate(labels, scores)
    return _curve(labels, scores, roc=True)


def policy_evaluate(
    labels: Sequence[int], results: Sequence[ScoreResult], cutoff: float = 0.5
) -> dict[str, Any]:
    """Evaluate review actions over all rows, keeping failed probabilities untouched.

    Args:
        labels: Full aligned binary evaluation labels.
        results: One saved result per advert; failures require an explicit error.
        cutoff: Fixed 0.5 in tables; descriptive cutoffs are used only for curves.

    Returns:
        Screening-policy metrics, valid-prediction coverage and failure counts.

    Raises:
        ValueError: Rows, identities, predictions or cutoff are invalid.
    """
    if not labels or len(labels) != len(results):
        raise ValueError("Policy evaluation requires full aligned result coverage")
    if len({r.advert_id for r in results}) != len(results):
        raise ValueError("Policy evaluation requires unique advert IDs")
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Cutoff must be finite and in [0, 1]")
    if any(r.error is None and r.p_relevant is None for r in results):
        raise ValueError("Missing prediction without an explicit inference failure")
    if any(
        r.error is not None and (not isinstance(r.error, str) or not r.error.strip())
        for r in results
    ):
        raise ValueError("Inference failures require nonempty error text")
    failures = sum(r.error is not None for r in results)
    # These are review actions, never imputed probabilities or modified ScoreResults.
    actions = [
        float(r.error is not None or (r.p_relevant is not None and r.p_relevant >= cutoff))
        for r in results
    ]
    metrics = evaluate(labels, actions)
    metrics.update(
        cutoff=cutoff,
        evaluation_policy=EVALUATION_POLICY,
        valid_predictions=len(results) - failures,
        failures=failures,
        coverage=(len(results) - failures) / len(results),
    )
    return metrics


def screening_metrics(
    labels: Sequence[int], results: Sequence[ScoreResult], cutoff: float = 0.5
) -> dict[str, Any]:
    """Add a retrospective high-recall summary to the fixed-rule policy metrics.

    A grouped sorted-score sweep keeps ties intact and failures always reviewed.
    It selects no operational threshold and uses no interpolated ROC coordinates.
    Binary, scoreless and single-class arms have no target-recall summary.
    """
    metrics = policy_evaluate(labels, results, cutoff)
    metrics["fpr_at_95_tpr"] = None
    positives = sum(labels)
    negatives = len(labels) - positives
    scores = [r.p_relevant if r.error is None else None for r in results]
    if (
        not positives
        or not negatives
        or metrics["valid_predictions"] == 0
        or any(r.score_type == "binary" for r in results)
    ):
        return metrics
    # Compare integer counts so rounding cannot admit a point just below 95%.
    for tp, fp in _review_counts(labels, scores):
        if 20 * tp >= 19 * positives:
            metrics["fpr_at_95_tpr"] = fp / negatives
            break
    return metrics


def policy_workload_curve(
    labels: Sequence[int], results: Sequence[ScoreResult]
) -> list[tuple[float, float]]:
    """Sweep genuine scores while recorded failures are always sent for review.

    The first point reviews failures only; ties are then added together. With no
    failures this is identical to workload_curve, including its original origin.
    """
    policy_evaluate(labels, results)
    scores = [r.p_relevant if r.error is None else None for r in results]
    return _curve(labels, scores, roc=False)


def policy_roc_curve(
    labels: Sequence[int], results: Sequence[ScoreResult]
) -> list[tuple[float, float]]:
    """Sweep genuine scores into policy ROC coordinates with failures reviewed.

    The first point contains reviewed failures only; ties are then added together.
    With no failures this is identical to :func:`roc_curve`.
    """
    policy_evaluate(labels, results)
    scores = [r.p_relevant if r.error is None else None for r in results]
    return _curve(labels, scores, roc=True)


def published_workload(precision: float, recall: float, positives: int, total: int) -> float:
    """Approximate workload only under the positive-class metric assumption.

    Args:
        precision: Published positive-class precision, assumed rather than verified.
        recall: Published positive-class recall under the same assumption.
        positives: Positive count in the comparison dataset.
        total: Total adverts in the comparison dataset.

    Returns:
        Fraction flagged, assuming compatible prevalence and rounded published rates.

    Raises:
        ValueError: Inputs are invalid or imply more flagged adverts than exist.
    """
    if not 0 < precision <= 1 or not 0 <= recall <= 1 or not 0 < positives <= total:
        raise ValueError("Invalid inputs for published workload approximation")
    workload = recall * positives / precision / total
    if workload > 1:
        raise ValueError("Published point implies more flagged adverts than exist")
    return workload


def published_false_positive_rate(
    precision: float, recall: float, positives: int, total: int
) -> float:
    """Derive a conditional FPR from published positive-class precision and recall.

    The calculation uses fractional counts without rounding. It is valid only if
    the published metrics use the assumed convention and compatible prevalence.
    """
    if not 0 < precision <= 1 or not 0 <= recall <= 1 or not 0 < positives < total:
        raise ValueError("Invalid inputs for published FPR approximation")
    estimated_false_positives = recall * positives * (1 / precision - 1)
    false_positive_rate = estimated_false_positives / (total - positives)
    if false_positive_rate > 1:
        raise ValueError("Published point implies more false positives than negatives")
    return false_positive_rate


def published_rates(
    precision: float | None,
    recall: float | None,
    positives: int,
    total: int,
    *,
    reported_fpr: float | None = None,
) -> tuple[float | None, float | None]:
    """Return conditional FPR and balanced accuracy at the dataset prevalence.

    A custom CLI baseline may supply its own FPR when reconstruction is unavailable.
    """
    fpr = reported_fpr
    if precision is not None and recall is not None and 0 < positives < total:
        fpr = published_false_positive_rate(precision, recall, positives, total)
    return fpr, balanced_accuracy(recall, fpr)
