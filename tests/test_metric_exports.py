"""Verify additive screening summaries and automatic exports with invented evidence."""

import csv
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from report_support import _table
from site_builder.content import render_markdown

from wildlife_screening.cli import main
from wildlife_screening.config import ModelConfig
from wildlife_screening.metrics import policy_evaluate, policy_roc_curve, screening_metrics
from wildlife_screening.prompts import Prompt
from wildlife_screening.report import COLUMNS, PERFORMANCE_HEADERS, create_report
from wildlife_screening.runner import run_arm
from wildlife_screening.scorers.local import FakeScorer
from wildlife_screening.types import Advert, EvaluationData, ScoreResult, ScoreType

ROOT = Path(__file__).resolve().parents[1]


def scores(values: list[float | None], kind: ScoreType = "native_prob") -> list[ScoreResult]:
    """Build invented responses whose missing scores are explicit failures."""
    return [
        ScoreResult(
            str(i),
            value,
            kind,
            "{}",
            None,
            None,
            None,
            0,
            error="Invented failure" if value is None else None,
        )
        for i, value in enumerate(values)
    ]


@pytest.mark.parametrize(
    "labels,values,expected",
    [
        ([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.0], 0.5),  # Ties enter together.
        ([1, 0, 0], [None, 0.9, 0.1], 0.0),  # Failures-only already meets target.
        ([1, 1, 0, 0], [0.9, 0.1, None, 0.0], 0.5),  # Mandatory negative review.
        ([1] * 20 + [0, 0], [0.9] * 19 + [0.1, 0.5, 0.0], 0.0),  # Exactly 95%.
        ([1] * 21 + [0, 0], [0.9] * 19 + [0.1, 0.1, 0.5, 0.0], 0.5),  # Overshoot.
        ([1, 0, 0], [0.50004, 0.50003, 0.0], 0.0),  # No rounded thresholds.
        ([1, 0], [None, None], None),
        ([1, 1], [0.9, 0.1], None),
        ([0, 0], [0.9, 0.1], None),
    ],
)
def test_target_recall_hand_calculated(
    labels: list[int], values: list[float | None], expected: float | None
) -> None:
    results = scores(values)
    metrics = screening_metrics(labels, results)
    assert metrics["fpr_at_95_tpr"] == expected
    if not 0 < sum(labels) < len(labels):
        assert metrics["balanced_accuracy"] is None
    legacy = policy_evaluate(labels, results)
    assert {key: metrics[key] for key in legacy} == legacy
    if 0 < sum(labels) < len(labels) and any(value is not None for value in values):
        reference = min(fpr for fpr, recall in policy_roc_curve(labels, results) if recall >= 0.95)
        assert metrics["fpr_at_95_tpr"] == reference


def test_binary_has_balanced_accuracy_but_no_target_recall() -> None:
    metrics = screening_metrics([1, 0, 0], scores([1, 1, 0], "binary"))
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["fpr_at_95_tpr"] is None


@pytest.mark.parametrize("failed", [False, True])
def test_runner_exports_provenance_failures_and_cached_metrics(
    data: EvaluationData, prompt: Prompt, tmp_path: Path, failed: bool
) -> None:
    class InventedScorer(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            results = super().score(adverts)
            return (
                [replace(r, p_relevant=None, error="Invented failure") for r in results]
                if failed
                else results
            )

    config = ModelConfig.load(ROOT / "configs/models/fake.yaml")
    output = tmp_path / "run"
    for iteration in range(2):
        results, manifest_path = run_arm(
            data, config, prompt, output, tmp_path / "cache", scorer=InventedScorer()
        )
        manifest = json.loads(manifest_path.read_text())
        artifact = json.loads(
            manifest_path.with_name(manifest_path.name.replace("manifest-", "metrics-")).read_text()
        )
        assert artifact["metrics"] == screening_metrics(data.labels, results)
        for key in (
            "task",
            "run_id",
            "arm",
            "prompt_hash",
            "model_config_hash",
            "data_file_hash",
            "evaluation_policy",
        ):
            assert artifact[key] == manifest[key]
        assert artifact["decision_rule"] == "score >= 0.5"
        assert artifact["metrics"]["failures"] == (len(data.labels) if failed else 0)
        assert manifest["cached_results"] == (len(data.labels) if iteration else 0)
        score_bytes = manifest_path.with_name(
            manifest_path.name.replace("manifest-", "scores-")
        ).read_bytes()
        assert manifest["scores_sha256"] == hashlib.sha256(score_bytes).hexdigest()
    assert len(list(output.glob("metrics-*.json"))) == 2
    if failed:
        assert artifact["metrics"]["fpr_at_95_tpr"] is None


def test_dry_and_aborted_runs_have_no_metrics(
    data: EvaluationData, prompt: Prompt, tmp_path: Path
) -> None:
    config = ModelConfig.load(ROOT / "configs/models/fake.yaml")
    run_arm(data, config, prompt, tmp_path / "dry", tmp_path / "cache", dry_run=True)

    class BrokenScorer(FakeScorer):
        def score(self, adverts: list[Advert]) -> list[ScoreResult]:
            raise RuntimeError("Invented abort")

    with pytest.raises(RuntimeError):
        run_arm(data, config, prompt, tmp_path / "abort", tmp_path / "cache", scorer=BrokenScorer())
    assert not list(tmp_path.rglob("metrics-*.json"))


@pytest.mark.parametrize("score_only", [False, True])
def test_cli_automatically_exports_metrics(tmp_path: Path, score_only: bool) -> None:
    args = [
        "run",
        "--data",
        str(ROOT / "tests/fixtures/synthetic20.csv"),
        "--model",
        str(ROOT / "configs/models/fake_binary.yaml"),
        "--zero-shot",
        "--output",
        str(tmp_path / "out"),
        "--cache",
        str(tmp_path / "cache"),
    ]
    assert main(args + (["--score-only"] if score_only else [])) == 0
    artifact = json.loads(next((tmp_path / "out").rglob("metrics-*.json")).read_text())
    assert artifact["decision_rule"] == "binary decision"
    assert artifact["metrics"]["fpr_at_95_tpr"] is None
    assert artifact["metrics"]["balanced_accuracy"] is not None
    assert bool(list((tmp_path / "out").rglob("table1.csv"))) != score_only


def test_legacy_csv_and_requested_summaries(tmp_path: Path) -> None:
    rows = create_report(
        [1, 0, 0],
        {"Invented": scores([1, 1, 0], "binary")},
        tmp_path,
        ROOT / "baselines/published_sharks.yaml",
    )
    with (tmp_path / "table1.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == COLUMNS
        csv_rows = list(reader)
    # Frozen legacy field order is a compatibility contract for positional readers.
    old_columns = [
        "arm",
        "task",
        "data_sha256",
        "positive_label",
        "provenance",
        "input_modality",
        "input_condition",
        "label_review_sha256",
        "condition",
        "prompt_family",
        "prompt_used",
        "example_count",
        "prompt_version",
        "prompt_hash",
        "prompt_rendering",
        "status",
        "score_type",
        "cutoff",
        "n",
        "positives",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "missed",
        "recall_ci_low",
        "recall_ci_high",
        "false_positive_rate",
        "published_precision",
        "published_recall",
        "published_f1",
        "tokens_in",
        "tokens_out",
        "cost",
        "latency_s_total",
        "errors",
        "evaluation_policy",
        "valid_predictions",
        "failures",
        "coverage",
        "note",
    ]
    assert COLUMNS[:-2] == old_columns
    assert COLUMNS[-2:] == ["balanced_accuracy", "fpr_at_95_tpr"]
    assert csv_rows[0]["balanced_accuracy"] == "0.750000"
    assert csv_rows[0]["fpr_at_95_tpr"] == "N/A"
    assert csv_rows[1]["fn"] == "unknown" and csv_rows[1]["recall_ci_low"] == "unknown"
    assert rows[1]["balanced_accuracy"] is not None
    assert rows[-1]["balanced_accuracy"] is None
    markdown = (tmp_path / "table1.md").read_text()
    assert "| " + " | ".join(PERFORMANCE_HEADERS) + " |" in markdown
    assert "Weighted P / R / F1" in markdown and "## Resource accounting and coverage" in markdown
    assert "[N/A]" in markdown and " (est.)" in markdown


def test_website_headers_cells_and_interval() -> None:
    metrics = screening_metrics([1, 0, 0], scores([0.9, 0.8, 0.0]))
    row: dict[str, Any] = {
        "arm": "Invented",
        "kind": "measured",
        "status": "complete",
        "input": "title",
        "prompt": "zero-shot",
        "metrics": metrics,
        "coverage": 1.0,
        "failures": 0,
    }
    table = _table([row])
    assert re.findall(r"<th>(.*?)</th>", table) == PERFORMANCE_HEADERS
    cells = re.findall(r"<td>(.*?)</td>", table.split("<tbody>")[1].split("</tr>")[0])
    assert len(cells) == 11
    assert cells[5] == "<strong>100.0% [20.7%, 100.0%]</strong>"
    assert cells[8] == "<strong>0.0%</strong>"
    assert cells[10] == "100.0% / 0"


def test_evaluation_markdown_nested_resources() -> None:
    source = (
        "1. TPR\n2. FPR\n3. Resource usage:\n    - Pricing\n    - Time\n"
        "4. Preparation\n\n#### Metrics\n\nUnchanged paragraph.\n"
    )
    rendered = render_markdown(source)
    assert "<li>Resource usage:<ul>" in rendered
    assert "<li>Pricing</li>\n<li>Time</li>" in rendered
    assert "</ul>\n</li>\n<li>Preparation</li>" in rendered
    assert "<h4>Metrics</h4>" in rendered
