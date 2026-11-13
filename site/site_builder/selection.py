"""Explicit, atomic selection of immutable experiment snapshots for the site."""

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from wildlife_screening.metrics import EVALUATION_POLICY
from wildlife_screening.snapshots import _identity, _load_pinned, _read_selection


def resolved(selection: dict[str, Any], root: Path) -> dict[str, Any]:
    """Resolve stored repository-relative references independently of the draft location."""
    result = copy.deepcopy(selection)
    result["pins"] = [str((root / pin).resolve()) for pin in selection["pins"]]
    for settings in result.get("datasets", {}).values():
        if not isinstance(settings, dict):
            raise ValueError("Each dataset setting must be a mapping")
        for field in ("data", "label_review"):
            if field in settings:
                settings[field] = str((root / settings[field]).resolve())
    return result


def validate(selection: dict[str, Any], root: Path) -> None:
    """Reject unsupported policy, invalid snapshots and ambiguous configuration replacements."""
    if selection.get("evaluation_policy", EVALUATION_POLICY) != EVALUATION_POLICY:
        raise ValueError("Unsupported report evaluation policy")
    normalized = resolved(selection, root)
    seen: set[tuple[str, str]] = set()
    for pin in normalized["pins"]:
        path = Path(pin)
        if not path.name.startswith("manifest-") or path.suffix != ".json":
            raise ValueError("Select an immutable manifest-<run_id>.json snapshot")
        row, _ = _load_pinned(path, normalized)
        identity = (row["dataset"], row["series"])
        if identity in seen:
            raise ValueError(
                f"Conflicting snapshots for {identity}; edit selection.yaml explicitly"
            )
        seen.add(identity)


def _stored(path: Path, root: Path) -> str:
    absolute = (root / path).resolve()
    return str(absolute.relative_to(root) if absolute.is_relative_to(root) else absolute)


def import_selection(
    source: Path,
    destination: Path,
    root: Path,
    *,
    candidate: bool = False,
    data: Path | None = None,
    positive_label: str | None = None,
    label_review: Path | None = None,
) -> int:
    """Validate an entire merged selection before atomically replacing the active file.

    Candidates and manifests use repository-relative references, never references
    relative to their own directory. Exact duplicate pins are harmless; a second
    snapshot of the same experimental configuration requires deliberate manual editing.
    """
    if destination.is_symlink():
        raise ValueError("Selection file must not be a symbolic link")
    merged = _read_selection(destination) if destination.exists() else {"pins": []}
    incoming: dict[str, Any] = _read_selection(source) if candidate else {"pins": [str(source)]}
    for selection in (merged, incoming):
        if selection.get("evaluation_policy", EVALUATION_POLICY) != EVALUATION_POLICY:
            raise ValueError("Unsupported report evaluation policy")
    if not candidate:
        manifest = json.loads(source.read_text(encoding="utf-8"))
        dataset = _identity(manifest)["dataset"]
        settings: dict[str, str] = {}
        if data is not None:
            settings["data"] = _stored(data, root)
        if positive_label is not None:
            settings["positive_label"] = positive_label
        if label_review is not None:
            settings["label_review"] = _stored(label_review, root)
        incoming["datasets"] = {dataset: settings}
    datasets = merged.setdefault("datasets", {})
    for key, settings in incoming.get("datasets", {}).items():
        if not isinstance(settings, dict):
            raise ValueError("Each dataset setting must be a mapping")
        active = datasets.setdefault(key, {})
        for field, value in settings.items():
            if field in {"data", "label_review"}:
                value = _stored(Path(value), root)
            if field in active:
                previous = active[field]
                if field in {"data", "label_review"}:
                    previous = _stored(Path(previous), root)
                if previous != value:
                    raise ValueError(f"Conflicting dataset setting {key}/{field}")
            active[field] = value
    pins = list(dict.fromkeys(_stored(Path(pin), root) for pin in merged["pins"]))
    before = len(pins)
    for pin in incoming["pins"]:
        reference = _stored(Path(pin), root)
        if reference not in pins:
            pins.append(reference)
    merged["pins"] = pins
    merged["evaluation_policy"] = EVALUATION_POLICY
    # The website has its own documented display defaults; retired UI settings
    # must not become a second source of truth.
    merged.pop("default_visible", None)
    validate(merged, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".selection-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(merged, handle, sort_keys=False)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return len(pins) - before
