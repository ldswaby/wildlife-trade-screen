"""Prepare the Stringham birds composite target without modifying source annotations."""

import argparse
import csv
import io
from pathlib import Path
from typing import Any

from .utils import digest, file_hash, write_json

LABELS = ("label1", "label2", "label3")
SOURCE_FIELDS = {"classifieds_unique_listing_id", "all_text_raw", *LABELS}
RULE = "0 if any exclusion is 1; 1 if all three are 0; otherwise unresolved"


def derive_label(values: tuple[str, ...]) -> int | None:
    """Apply the composite policy without treating missing annotations as negatives."""
    if len(values) != 3 or any(value not in {"0", "1", "NA"} for value in values):
        raise ValueError("Expected three exclusion labels containing only 0, 1 or NA")
    if "1" in values:
        return 0
    return 1 if all(value == "0" for value in values) else None


def _csv_bytes(fields: list[str], rows: list[dict[str, str]]) -> bytes:
    """Serialize reproducible UTF-8 CSV bytes, preserving embedded text exactly."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    """Replace one derived CSV atomically after all source validation succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def prepare_birds(
    source: Path = Path("data/birds.csv"),
    output: Path = Path("data/birds_processed.csv"),
    unresolved: Path = Path("data/birds_unresolved.csv"),
    summary: Path = Path("data/birds_preprocessing.json"),
) -> dict[str, Any]:
    """Write a labelled evaluation CSV, unresolved records and an audit summary.

    Args:
        source: Original ISO-8859-1 author CSV, retained unchanged.
        output: UTF-8 evaluation CSV containing only ID, combined text and label.
        unresolved: UTF-8 CSV retaining original fields plus an exclusion reason.
        summary: Private JSON provenance and denominator reconciliation.

    Returns:
        Deterministic provenance with counts and hashes, never advert text.

    Raises:
        ValueError: Paths overlap, or source structure, IDs, text or labels are invalid.
        OSError: Source cannot be read or derived files cannot be written.
    """
    paths = [source, output, unresolved, summary]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Source and output paths must be distinct")
    # Also protect the source from the atomic writers' temporary paths.
    temporary_paths = [path.with_suffix(path.suffix + ".tmp").resolve() for path in paths[1:]]
    if set(temporary_paths) & {path.resolve() for path in paths}:
        raise ValueError("Output temporary paths must not overlap source or outputs")
    raw = source.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("iso-8859-1"), newline=""))
    fields = reader.fieldnames
    if (
        not fields
        or len(fields) != len(set(fields))
        or not SOURCE_FIELDS <= set(fields)
        or "exclusion_reason" in fields
    ):
        raise ValueError("Missing, duplicate or reserved source columns")
    rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Empty data or malformed CSV rows")
    retained: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=2):
        advert_id = row["classifieds_unique_listing_id"]
        if not advert_id.strip() or advert_id in seen:
            raise ValueError(f"Blank or duplicate advert ID at CSV row {index}")
        seen.add(advert_id)
        if not row["all_text_raw"].strip():
            raise ValueError(f"Empty raw text at CSV row {index}")
        label = derive_label(tuple(row[key] for key in LABELS))
        if label is None:
            excluded.append(
                {**row, "exclusion_reason": "No known exclusion; annotation incomplete"}
            )
        else:
            retained.append(
                {"advert_id": advert_id, "title_and_desc": row["all_text_raw"], "label": str(label)}
            )
    processed_bytes = _csv_bytes(["advert_id", "title_and_desc", "label"], retained)
    unresolved_bytes = _csv_bytes([*fields, "exclusion_reason"], excluded)
    audit: dict[str, Any] = {
        "preprocessing_version": 1,
        "source": str(source),
        "source_sha256": digest(raw),
        "source_encoding": "ISO-8859-1",
        "reference_release": "https://doi.org/10.6084/m9.figshare.14032742.v2",
        "output_encoding": "UTF-8",
        "rule": RULE,
        "positive_label": "1",
        "input_condition": "combined_text",
        "source_rows": len(rows),
        "evaluation_rows": len(retained),
        "positives": sum(row["label"] == "1" for row in retained),
        "negatives": sum(row["label"] == "0" for row in retained),
        "unresolved_rows": len(excluded),
        "outputs": {
            "processed": {"path": str(output), "sha256": digest(processed_bytes)},
            "unresolved": {"path": str(unresolved), "sha256": digest(unresolved_bytes)},
        },
    }
    if file_hash(source) != audit["source_sha256"]:
        raise ValueError("Source changed during preprocessing")
    _write_bytes(output, processed_bytes)
    _write_bytes(unresolved, unresolved_bytes)
    write_json(summary, audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    """Prepare local files offline and print aggregate counts only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/birds.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/birds_processed.csv"))
    parser.add_argument("--unresolved", type=Path, default=Path("data/birds_unresolved.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/birds_preprocessing.json"))
    args = parser.parse_args(argv)
    try:
        audit = prepare_birds(args.source, args.output, args.unresolved, args.summary)
    except (ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(
        f"Prepared {audit['evaluation_rows']} adverts: {audit['positives']} relevant, "
        f"{audit['negatives']} irrelevant; "
        f"{audit['unresolved_rows']} unresolved retained separately."
    )
    print(f"Evaluation CSV: {args.output}; unresolved: {args.unresolved}; audit: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
