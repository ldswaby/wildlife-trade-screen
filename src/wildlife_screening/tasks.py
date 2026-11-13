"""Fixed task definitions and optional historical label-review records."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import file_hash


@dataclass(frozen=True)
class Task:
    """Describe one evaluation question without reading its private adverts."""

    name: str
    data: Path
    baseline: Path | None
    pool: Path | None = None
    positive_label: str | None = None
    input_modes: tuple[str, ...] = ("title", "title_and_desc")
    shot_conditions: tuple[str, ...] = ("few-shot", "zero-shot")


TASKS = {
    "birds": Task(
        "birds",
        Path("data/birds_processed.csv"),
        None,
        positive_label="1",
        input_modes=("combined_text",),
        shot_conditions=("zero-shot",),
    ),
    "sharks": Task(
        "sharks",
        Path("data/validation_sharks.csv"),
        Path("baselines/published_sharks.yaml"),
        Path("data/shark_trophy.csv"),
    ),
    "animals": Task(
        "animals",
        Path("data/validation_animals.csv"),
        Path("baselines/published_animals.yaml"),
        positive_label="1",
        input_modes=("title",),
    ),
    "leather": Task(
        "leather",
        Path("data/leather_validation.csv"),
        Path("baselines/published_leather.yaml"),
        positive_label="1",
        input_modes=("title",),
    ),
}


def label_review(task: Task, data: Path, path: Path | None) -> dict[str, Any]:
    """Verify a human review declaration against the exact source CSV.

    Args:
        task: Selected research task.
        data: Evaluation file being scored.
        path: Optional historical private JSON review record.

    Returns:
        Review provenance and an explicit positive numeric label.

    Raises:
        ValueError: A supplied review is incomplete, mismatched, or insufficient.
    """
    if path is None:
        return {}
    if task.name == "sharks":
        raise ValueError("--label-review is only used for animals or leather")
    review = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("Label review must be a JSON object")
    if review.get("task") != task.name or review.get("data_sha256") != file_hash(data):
        raise ValueError("Label review task or CSV hash does not match this run")
    if review.get("positive_label") not in ("0", "1"):
        raise ValueError("Label review must declare positive_label as '0' or '1'")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("Label review must name the reviewer")
    if review.get("basis") not in {"source_documentation", "independent_adjudication"}:
        raise ValueError("Label review needs source documentation or independent adjudication")
    if not isinstance(review.get("evidence"), str) or not review["evidence"].strip():
        raise ValueError("Label review must describe the evidence")
    if task.name == "leather" and review.get("basis") == "independent_adjudication":
        if not isinstance(review.get("audit_scope"), str) or not review["audit_scope"].strip():
            raise ValueError("Leather adjudication review must describe its audit scope")
    return {
        "task": task.name,
        "label_review_sha256": file_hash(path),
        "label_review_basis": review["basis"],
        "positive_label": review["positive_label"],
    }
