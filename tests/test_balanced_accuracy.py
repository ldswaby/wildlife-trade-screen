"""Check fixed-rule balanced accuracy using invented screening outcomes."""

import copy
import re
from typing import Any

import pytest
from report_support import _table

from wildlife_screening.metrics import policy_evaluate
from wildlife_screening.types import ScoreResult, ScoreType


def _cells(table: str) -> list[str]:
    """Extract the first row's cells while retaining leader markup."""
    body = table.split("<tbody>", 1)[1].split("</tr>", 1)[0]
    return re.findall(r"<td>(.*?)</td>", body)


@pytest.mark.parametrize(
    "labels,predictions,expected",
    [
        ([1, 1, 0, 0], [1, 1, 0, 0], "100.0%"),
        ([1, 1, 0, 0], [0, 0, 1, 1], "0.0%"),
        ([1, 0, 0, 0], [1, 1, 1, 1], "50.0%"),
        ([1, 0, 0, 0], [0, 0, 0, 0], "50.0%"),
        # TPR 1/2, specificity 3/4: BA 62.5%, ordinary accuracy 4/6.
        ([1, 1, 0, 0, 0, 0], [0.5, 0.1, 0.9, 0.2, 0.3, 0.4], "62.5%"),
        ([1, 1], [1, 0], "N/A"),
        ([0, 0], [1, 0], "N/A"),
    ],
)
def test_fixed_rule_balanced_accuracy_from_confusion_counts(
    labels: list[int], predictions: list[float], expected: str
) -> None:
    results = [
        ScoreResult(str(i), p, "native_prob", "{}", None, None, None, 0)
        for i, p in enumerate(predictions)
    ]
    metrics = policy_evaluate(labels, results)
    row: dict[str, Any] = {
        "kind": "measured",
        "status": "complete",
        "arm": "Invented",
        "input": "title",
        "prompt": "zero-shot",
        "metrics": metrics,
    }
    before = copy.deepcopy(row)
    table = _table([row])
    assert "<th>Balanced accuracy</th><th>F1</th><th>TPR [95% CI]</th>" in table
    assert re.sub(r"</?strong>", "", _cells(table)[3]) == expected
    assert row == before  # Presentation does not alter saved metrics or interfaces.


@pytest.mark.parametrize("score_type", ["native_prob", "binary"])
def test_balanced_accuracy_preserves_review_on_failure(score_type: ScoreType) -> None:
    # Positive failure is caught; negative failure adds an FP. TPR=1, TNR=1/2.
    scores = [
        ScoreResult("a", None, score_type, "{}", None, None, None, 0, error="Invented failure"),
        ScoreResult("b", None, score_type, "{}", None, None, None, 0, error="Invented failure"),
        ScoreResult("c", 0, score_type, "{}", None, None, None, 0),
    ]
    metrics = policy_evaluate([1, 0, 0], scores)
    table = _table(
        [
            {
                "kind": "measured",
                "status": "finished with failures",
                "arm": "Invented",
                "input": "title",
                "prompt": "zero-shot",
                "metrics": metrics,
            }
        ]
    )
    assert _cells(table)[3] == "<strong>75.0%</strong>"
    assert metrics["n"] == 3 and metrics["failures"] == 2


def test_published_balanced_accuracy_is_conditional_and_f1_only_is_unknown() -> None:
    rows: list[dict[str, Any]] = [
        {
            "kind": "published",
            "arm": "Invented published",
            "input_modality": "text",
            "recall": 0.8,
            "false_positive_rate": 0.1,
        }
    ]
    before = copy.deepcopy(rows)
    cell = _cells(_table(rows))[3]
    assert cell.startswith('<strong>85.0%</strong><sup class="published-origin"')
    assert cell.endswith(">‡</sup>")
    assert "(est.)" not in cell
    assert rows == before
    rows[0] = {
        **rows[0],
        "arm": "GPT-4 zero-shot",
        "recall": None,
        "false_positive_rate": None,
        "f1": 0.9,
    }
    cells = _cells(_table(rows))
    assert cells[3] == "N/A"
    assert cells[4].endswith(">†</sup>")
    assert sum(cell.count(">†</sup>") for cell in cells) == 1
    assert all(">‡</sup>" not in cell for cell in cells)


def test_balanced_accuracy_leaders_use_unrounded_values_and_retain_ties() -> None:
    rows: list[dict[str, Any]] = [
        {
            "kind": "measured",
            "status": status,
            "arm": name,
            "input": "title",
            "prompt": "zero-shot",
            "metrics": {"positive_recall": recall, "false_positive_rate": 0.1},
        }
        for name, recall, status in [
            ("A", 0.8001, "complete"),
            ("B", 0.8002, "complete"),
            ("C", 0.8002, "complete"),
            ("Unfinished", 1.0, "incomplete; metrics withheld"),
        ]
    ]
    table = _table(rows)
    assert table.count("<strong>85.0%</strong>") == 2
    assert "<strong>95.0%</strong>" not in table
