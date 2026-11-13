from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve as sklearn_roc_curve

from wildlife_screening.metrics import (
    evaluate,
    policy_evaluate,
    policy_roc_curve,
    published_false_positive_rate,
    recall_ci,
    roc_curve,
    workload_curve,
)


def test_single_class_curve_errors_name_the_curve() -> None:
    from wildlife_screening.types import ScoreResult

    results = [
        ScoreResult(str(i), v, "native_prob", "{}", None, None, None, 0)
        for i, v in enumerate([0.2, 0.7])
    ]
    with pytest.raises(ValueError, match="^ROC curve undefined with no positives"):
        roc_curve([0, 0], [0.2, 0.7])
    with pytest.raises(ValueError, match="^ROC curve undefined with no positives"):
        policy_roc_curve([0, 0], results)
    with pytest.raises(ValueError, match="^Recall curve undefined with no positives"):
        workload_curve([0, 0], [0.2, 0.7])
    with pytest.raises(ValueError, match="^ROC curve undefined with no negatives"):
        roc_curve([1, 1], [0.2, 0.7])


def test_hand_computed() -> None:
    result = evaluate([1, 1, 1, 0, 0, 0], [0.9, 0.7, 0.1, 0.8, 0.2, 0.3])
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 2)
    for key in (
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "accuracy",
        "balanced_accuracy",
    ):
        assert result[key] == pytest.approx(2 / 3)
    assert result["missed"] == "missed 1 of 3 positives"
    assert result["false_positive_rate"] == pytest.approx(1 / 3)
    assert result["recall_ci_low"] == pytest.approx(0.2076596008)
    assert result["recall_ci_high"] == pytest.approx(0.9385080553)


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([1, 1, 0, 0, 0], [0.9, 0.4, 0.5, 0.1, 0.2]),
        ([1, 0, 0], [0, 0, 0]),
        ([1, 1, 0], [1, 1, 1]),
        ([0, 0], [0.2, 0.7]),
        ([1, 1], [0.9, 0.1]),
    ],
)
def test_sklearn(labels: list[int], scores: list[float]) -> None:
    result = evaluate(labels, scores)
    predictions = [int(p >= 0.5) for p in scores]
    for average, prefix in [("binary", "positive"), ("weighted", "weighted")]:
        p, r, f, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        for metric, value in [("precision", p), ("recall", r), ("f1", f)]:
            if prefix == "positive" and metric == "recall" and not sum(labels):
                assert result[f"{prefix}_{metric}"] is None
            else:
                assert result[f"{prefix}_{metric}"] == pytest.approx(value)
    assert result["accuracy"] == accuracy_score(labels, predictions)
    assert result["weighted_recall"] == pytest.approx(result["accuracy"])


def test_ties_and_endpoints() -> None:
    points = roc_curve([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.0])
    assert points == [(0, 0), (0.5, 0.5), (0.5, 1), (1, 1)]
    assert published_false_positive_rate(0.958, 0.892, 102, 499) == pytest.approx(
        0.010047506612748032
    )


@pytest.mark.parametrize(
    "baseline,positives,negatives",
    [
        ("published_sharks.yaml", 102, 397),
        ("published_animals.yaml", 101, 398),
        ("published_leather.yaml", 218, 282),
    ],
)
def test_each_published_fpr_uses_unrounded_conditional_counts(
    baseline: str, positives: int, negatives: int
) -> None:
    rows = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "baselines" / baseline).read_text()
    )["rows"]
    for row in rows:
        precision = row.get("published_precision")
        recall = row.get("published_recall")
        if precision is None or recall is None:
            continue
        estimated_false_positives = recall * positives * (1 / precision - 1)
        expected = estimated_false_positives / negatives
        assert published_false_positive_rate(
            precision, recall, positives, positives + negatives
        ) == pytest.approx(expected)


def test_roc_curve_matches_sklearn_without_dropping_thresholds() -> None:
    labels = [1, 0, 1, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.7, 0.1]
    expected_fpr, expected_recall, _ = sklearn_roc_curve(labels, scores, drop_intermediate=False)
    assert roc_curve(labels, scores) == pytest.approx(
        list(zip(expected_fpr, expected_recall, strict=True))
    )


def test_ci_edges() -> None:
    assert recall_ci(0, 0) == (None, None)
    assert recall_ci(0, 5)[0] == 0
    assert recall_ci(5, 5)[1] == 1


@pytest.mark.parametrize("scores", [[math.nan], [math.inf], [-0.1], [1.1]])
def test_reject_bad_scores(scores: list[float]) -> None:
    with pytest.raises(ValueError):
        evaluate([1], scores)


def test_review_on_failure_hand_calculated_and_raw_results_preserved() -> None:
    from dataclasses import asdict

    from wildlife_screening.types import ScoreResult

    results = [
        ScoreResult("a", None, "native_prob", "refusal", 1, 0, 0.1, 1, "refusal"),
        ScoreResult("b", None, "native_prob", "error", 1, 0, 0.1, 1, "error"),
        ScoreResult("c", 0.8, "native_prob", "ok", 1, 0, 0.1, 1),
        ScoreResult("d", 0.8, "native_prob", "ok", 1, 0, 0.1, 1),
        ScoreResult("e", 0.2, "native_prob", "ok", 1, 0, 0.1, 1),
        ScoreResult("f", 0.0, "native_prob", "ok", 1, 0, 0.1, 1),
    ]
    labels = [1, 0, 1, 0, 1, 0]
    before = [asdict(r) for r in results]
    metric = policy_evaluate(labels, results)
    assert (metric["tp"], metric["fp"], metric["fn"], metric["tn"]) == (2, 2, 1, 1)
    assert metric["positive_precision"] == 0.5
    assert metric["positive_recall"] == pytest.approx(2 / 3)
    assert metric["positive_f1"] == pytest.approx(4 / 7)
    assert metric["accuracy"] == 0.5
    assert metric["weighted_f1"] == pytest.approx((4 / 7 + 2 / 5) / 2)
    assert metric["workload"] == pytest.approx(4 / 6)
    assert metric["coverage"] == pytest.approx(4 / 6)
    assert metric["evaluation_policy"] == "review_on_failure"
    assert policy_roc_curve(labels, results) == [
        (1 / 3, 1 / 3),
        (2 / 3, 2 / 3),
        (2 / 3, 1),
        (1, 1),
    ]
    assert [asdict(r) for r in results] == before


def test_review_policy_edges_and_zero_failure_equivalence() -> None:
    from wildlife_screening.types import ScoreResult

    labels = [1, 0, 1]
    scores = [1.0, 0.5, 0.0]
    results = [ScoreResult(str(i), p, "native_prob", "", 0, 0, 0, 0) for i, p in enumerate(scores)]
    assert policy_roc_curve(labels, results) == roc_curve(labels, scores)
    assert {k: policy_evaluate(labels, results)[k] for k in evaluate(labels, scores)} == evaluate(
        labels, scores
    )
    failed = [ScoreResult(str(i), None, "native_prob", "", 0, 0, 0, 0, "refusal") for i in range(3)]
    assert policy_evaluate(labels, failed)["coverage"] == 0
    assert policy_roc_curve(labels, failed) == [(1, 1)]
    assert policy_evaluate([0, 0, 0], failed)["positive_recall"] is None
    assert policy_evaluate([1, 1, 1], failed)["false_positive_rate"] is None
    with pytest.raises(ValueError, match="no positives"):
        policy_roc_curve([0, 0, 0], failed)
    with pytest.raises(ValueError, match="no negatives"):
        policy_roc_curve([1, 1, 1], failed)
    with pytest.raises(ValueError, match="coverage"):
        policy_evaluate(labels, failed[:2])
    with pytest.raises(ValueError, match="unique"):
        policy_evaluate(labels, [failed[0]] * 3)
    with pytest.raises(ValueError, match="Cutoff"):
        policy_evaluate(labels, results, math.nan)


@pytest.mark.parametrize("error", ["", "   "])
def test_empty_error_records_do_not_become_fallback_predictions(error: str) -> None:
    from wildlife_screening.metrics import policy_evaluate
    from wildlife_screening.types import ScoreResult

    invalid = ScoreResult("invented", None, "native_prob", "", 0, 0, 0, 0, error)
    with pytest.raises(ValueError, match="nonempty error"):
        policy_evaluate([1], [invalid])
