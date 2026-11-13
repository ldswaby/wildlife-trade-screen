"""Load the validated runs and prompt provenance selected for each dataset tab."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wildlife_screening.metrics import EVALUATION_POLICY
from wildlife_screening.snapshots import _load_pinned, _published_rows
from wildlife_screening.tasks import TASKS
from wildlife_screening.types import EvaluationData
from wildlife_screening.utils import file_hash


@dataclass
class DatasetRuns:
    """Collect measured runs sharing a dataset hash in the selected order."""

    task: str
    data: EvaluationData
    rows: list[dict[str, Any]] = field(default_factory=list)
    prompts: dict[str, dict[str, str]] = field(default_factory=dict)

    def comparison_rows(self) -> list[dict[str, Any]]:
        """Include published context only for the original dataset file identity."""
        original = TASKS[self.task].data
        matched = original.exists() and self.data.sha256 == file_hash(original)
        return self.rows + _published_rows(self.task, self.data, matched)


def load_dataset_runs(selection: dict[str, Any]) -> dict[str, DatasetRuns]:
    """Group explicit pins after checking policy, row coverage and configuration identity."""
    if selection.get("evaluation_policy", EVALUATION_POLICY) != EVALUATION_POLICY:
        raise ValueError("Unsupported report evaluation policy")
    # Discovery lists available runs; only explicit pins determine report tabs.
    pinned_paths = [Path(item) for item in selection["pins"]]
    if len(pinned_paths) != len(set(pinned_paths)):
        raise ValueError("Report pins contain the same manifest twice")
    datasets: dict[str, DatasetRuns] = {}
    for path in pinned_paths:
        # Explicit pins outside the canonical root are valid legacy records.
        if not path.is_file():
            raise FileNotFoundError(path)
        row, data = _load_pinned(path, selection)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        key = row["dataset"]
        group = datasets.setdefault(key, DatasetRuns(str(manifest["task"]), data))
        if group.data.sha256 != data.sha256:
            raise ValueError("Two source CSV hashes collide in the dataset tab identifier")
        if any(existing["series"] == row["series"] for existing in group.rows):
            raise ValueError(f"Two pinned snapshots have the same configuration: {row['series']}")
        group.rows.append(row)
        group.prompts[manifest["prompt_hash"]] = {
            "family": manifest["prompt_family"],
            "condition": manifest["condition"],
            "version": str(manifest.get("prompt_version") or "Unreported version"),
            "sha256": manifest["prompt_hash"],
        }
    return datasets
