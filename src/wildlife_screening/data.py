"""Validate labels locally and expose only approved text fields to scorers."""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .types import Advert, EvaluationData
from .utils import file_hash

TEXT_LABELS = {"animal origin": 1, "not animal origin": 0}
INPUT_MODES = {
    ("title",): "title",
    ("title", "description"): "title_and_desc",
    ("title_and_desc",): "combined_text",
}


def input_mode(fields: tuple[str, ...]) -> str:
    """Resolve supported field selections without inferring from their length."""
    if fields not in INPUT_MODES:
        raise ValueError("Field allowlist permits title, title with description, or combined text")
    return INPUT_MODES[fields]


def input_fields(mode: str) -> tuple[str, ...]:
    """Resolve a declared input condition using the same allowlist as the loader."""
    for fields, name in INPUT_MODES.items():
        if name == mode:
            return fields
    raise ValueError(f"Unsupported input condition: {mode}")


def read_rows(path: Path) -> list[dict[str, str]]:
    """Reject malformed CSV structure before labels or text can enter a run.

    Args:
        path: Local CSV file, optionally with a UTF-8 byte order mark.

    Returns:
        Rows keyed by unique column names, preserving string values.

    Raises:
        ValueError: Headers are missing or duplicated, or rows are empty/malformed.
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing or duplicate CSV headers")
        rows = list(reader)
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Empty data or malformed CSV rows")
    return rows


def label_polarity(rows: list[dict[str, str]], positive_label: str | None = None) -> str:
    """Resolve relevance from semantic evidence rather than guessing numeric polarity.

    Args:
        rows: CSV rows with 0/1 labels and optional textual relevant values.
        positive_label: Explicit positive value after local review; None requires
            all populated recognised text labels to support one mapping.

    Returns:
        Numeric value meaning relevant. Blank text labels are not negatives.

    Raises:
        ValueError: Numeric/text labels are unknown or the mapping is ambiguous.
    """
    if any(r.get("label", "").strip() not in {"0", "1"} for r in rows):
        raise ValueError("Ambiguous label mapping: expected only explicit 0/1 labels")
    if positive_label not in {None, "0", "1"}:
        raise ValueError("positive_label must be 0 or 1")
    evidence: set[str] = set()
    for row in rows:
        relevant = row.get("relevant", "").strip().lower()
        if not relevant:
            continue
        if relevant not in TEXT_LABELS:
            raise ValueError(f"Unknown relevant label: {relevant!r}")
        raw = row["label"].strip()
        # A semantic negative supports the opposite numeric value as positive.
        evidence.add(raw if TEXT_LABELS[relevant] == 1 else str(1 - int(raw)))
    if positive_label is None:
        if len(evidence) != 1:
            raise ValueError("Ambiguous label mapping: supply --positive-label after local review")
        positive_label = evidence.pop()
    return positive_label


def numeric_labels(rows: list[dict[str, str]], positive_label: str | None = None) -> list[int]:
    """Map validated numeric labels to binary relevance without changing rows."""
    positive = label_polarity(rows, positive_label)
    return [int(row["label"].strip() == positive) for row in rows]


def load_data(
    path: Path, fields: tuple[str, ...] = ("title",), positive_label: str | None = None
) -> EvaluationData:
    """Separate gold labels from the fields authorised for model input.

    Args:
        path: Local evaluation CSV file.
        fields: Title by default; description or combined text require explicit selection.
        positive_label: Optional reviewed numeric polarity override.

    Returns:
        Aligned adverts and gold labels with the original file hash.

    Raises:
        ValueError: Fields, CSV schema, labels, titles or advert IDs are invalid.
    """
    mode = input_mode(fields)
    rows = read_rows(path)
    positive_label = label_polarity(rows, positive_label)
    labels = [int(row["label"].strip() == positive_label) for row in rows]
    adverts: list[Advert] = []
    for index, row in enumerate(rows):
        primary = "title_and_desc" if mode == "combined_text" else "title"
        if any(field not in row for field in fields) or not row[primary].strip():
            raise ValueError(f"Missing allowlisted text at CSV row {index + 2}")
        advert_id = row.get("advert_id") or row.get("id") or f"row-{index + 2}"
        adverts.append(
            Advert(advert_id, title_and_desc=row[primary])
            if mode == "combined_text"
            else Advert(
                advert_id, row["title"], row.get("description") if "description" in fields else None
            )
        )
    if len({a.advert_id for a in adverts}) != len(adverts):
        raise ValueError("Duplicate advert IDs; caching would be ambiguous")
    return EvaluationData(adverts, labels, file_hash(path), positive_label)


def inspect_data(
    validation: Path, pool: Path | None = None, positive_label: str | None = None
) -> dict[str, Any]:
    """Expose label uncertainty and keyword matches without modifying the gold data.

    Args:
        validation: Local evaluation CSV file.
        pool: Optional pool CSV; its label values are counted without inferring origin.
        positive_label: Optional reviewed numeric polarity override.

    Returns:
        A private report containing counts, disagreements, duplicates, keyword
        spot-checks and mapping errors. Keyword matches are not adjudications.

    Raises:
        ValueError: A CSV has malformed structural content.
    """
    rows = read_rows(validation)
    result: dict[str, Any] = {
        "rows": len(rows),
        "data_sha256": file_hash(validation),
        "label_values": dict(Counter(r.get("label", "") for r in rows)),
        "relevant_values": dict(Counter(r.get("relevant", "") for r in rows)),
        "description_present": sum(bool(r.get("description", "").strip()) for r in rows),
        "title_present": sum(bool(r.get("title", "").strip()) for r in rows),
        "title_and_desc_present": sum(bool(r.get("title_and_desc", "").strip()) for r in rows),
        "gpt_label_values": dict(Counter(r.get("GPT_label", "") for r in rows))
        if "GPT_label" in rows[0]
        else None,
    }
    disagreements: list[dict[str, Any]] = []
    mapping_error = None
    try:
        labels = numeric_labels(rows, positive_label)
        result["positives"] = sum(labels)
        result["positive_label"] = (
            rows[0]["label"].strip() if labels[0] else str(1 - int(rows[0]["label"]))
        )
        for index, (row, label) in enumerate(zip(rows, labels, strict=True)):
            text_label = row.get("relevant", "").strip().lower()
            if text_label and TEXT_LABELS[text_label] != label:
                disagreements.append(
                    {
                        "row": index + 2,
                        "label": row["label"],
                        "relevant": text_label,
                        "title": row.get("title", ""),
                    }
                )
    except ValueError as exc:
        mapping_error = str(exc)
        # If mappings conflict, list the competing evidence rather than picking a winner.
        supported = {
            r["label"].strip()
            if TEXT_LABELS[r["relevant"].strip().lower()]
            else str(1 - int(r["label"]))
            for r in rows
            if r.get("relevant", "").strip().lower() in TEXT_LABELS
            and r.get("label", "").strip() in {"0", "1"}
        }
        if len(supported) > 1:
            disagreements = [
                {
                    "row": i + 2,
                    "label": r["label"],
                    "relevant": r["relevant"],
                    "title": r.get("title", ""),
                    "reason": "competing mapping evidence; no gold value chosen",
                }
                for i, r in enumerate(rows)
                if r.get("relevant", "").strip()
            ]
    result["mapping_error"] = mapping_error
    result["disagreements"] = disagreements
    result["populated_columns_agree"] = not disagreements and mapping_error is None
    result["missing_relevant"] = sum(not r.get("relevant", "").strip() for r in rows)
    groups: dict[str, list[int]] = defaultdict(list)
    normalised_groups: dict[str, list[int]] = defaultdict(list)
    spots: dict[str, list[dict[str, Any]]] = {"fossil": [], "replica": []}
    for index, row in enumerate(rows):
        title = row.get("title", "")
        if title.strip():
            groups[title].append(index + 2)
            normalised_groups[" ".join(title.casefold().split())].append(index + 2)
        for category in spots:
            if re.search(rf"\b{category}\w*\b", title, flags=re.I):
                spots[category].append(
                    {
                        "row": index + 2,
                        "title": title,
                        "label": row["label"],
                        "relevant": row.get("relevant", ""),
                    }
                )
    result["spot_checks"] = spots
    result["duplicate_titles"] = [
        {"title": t, "rows": ids} for t, ids in groups.items() if len(ids) > 1
    ]
    result["normalised_duplicate_rows"] = [
        ids for ids in normalised_groups.values() if len(ids) > 1
    ]
    if pool:
        pool_rows = read_rows(pool)
        result["pool"] = {
            "rows": len(pool_rows),
            "data_sha256": file_hash(pool),
            "label_values": dict(Counter(r.get("label", "") for r in pool_rows)),
            "provenance": "unknown; numeric values do not prove pseudo-label origin",
        }
    return result
