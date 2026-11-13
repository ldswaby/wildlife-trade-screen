"""Exercise preparation with invented annotations and text, never private adverts."""

import csv
import itertools
from pathlib import Path

import pytest

from wildlife_screening.data import load_data, read_rows
from wildlife_screening.prepare_birds import LABELS, derive_label, main, prepare_birds
from wildlife_screening.utils import file_hash


@pytest.mark.parametrize("values", list(itertools.product(("0", "1", "NA"), repeat=3)))
def test_composite_truth_table(values: tuple[str, ...]) -> None:
    # A missing annotation cannot undo an established exclusion.
    expected = 0 if values.count("1") else (1 if values.count("0") == 3 else None)
    assert derive_label(values) == expected


def source_csv(path: Path) -> Path:
    rows = [
        ["keep", 'Invented café bird\r\nSecond line, quoted "text"', "0", "0", "0", "DROP_ME"],
        ["exclude", "Invented cage", "NA", "1", "NA", "DROP_ME"],
        ["unknown", "Invented unidentified listing", "0", "NA", "0", "DROP_ME"],
    ]
    with path.open("w", encoding="iso-8859-1", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["classifieds_unique_listing_id", "all_text_raw", *LABELS, "metadata"])
        writer.writerows(rows)
    return path


def test_prepare_preserves_source_text_and_reconciles_outputs(tmp_path: Path) -> None:
    source = source_csv(tmp_path / "source.csv")
    output, excluded, summary = [tmp_path / name for name in ("out.csv", "na.csv", "audit.json")]
    before = source.read_bytes()
    audit = prepare_birds(source, output, excluded, summary)
    assert source.read_bytes() == before
    assert audit["source_rows"] == 3
    assert audit["evaluation_rows"] == 2
    assert audit["positives"] == audit["negatives"] == audit["unresolved_rows"] == 1
    rows = read_rows(output)
    assert list(rows[0]) == ["advert_id", "title_and_desc", "label"]
    assert [row["advert_id"] for row in rows] == ["keep", "exclude"]
    assert rows[0]["title_and_desc"] == 'Invented café bird\r\nSecond line, quoted "text"'
    assert "DROP_ME" not in output.read_text(encoding="utf-8")
    assert read_rows(excluded)[0]["classifieds_unique_listing_id"] == "unknown"
    assert read_rows(excluded)[0]["metadata"] == "DROP_ME"
    assert audit["outputs"]["processed"]["sha256"] == file_hash(output)
    data = load_data(output, ("title_and_desc",), "1")
    assert data.labels == [1, 0]
    snapshots = [p.read_bytes() for p in (output, excluded, summary)]
    assert prepare_birds(source, output, excluded, summary) == audit
    assert snapshots == [p.read_bytes() for p in (output, excluded, summary)]


@pytest.mark.parametrize(
    "content",
    [
        "",
        "classifieds_unique_listing_id,all_text_raw,label1,label2\na,text,0,0\n",
        "classifieds_unique_listing_id,all_text_raw,label1,label1,label3\na,text,0,0,0\n",
        "classifieds_unique_listing_id,all_text_raw,label1,label2,label3\na,text,0,0\n",
        "classifieds_unique_listing_id,all_text_raw,label1,label2,label3\na,text,0,0,0,extra\n",
        "classifieds_unique_listing_id,all_text_raw,label1,label2,label3\na,text,2,0,0\n",
        "classifieds_unique_listing_id,all_text_raw,label1,label2,label3\na,text,,0,0\n",
        "classifieds_unique_listing_id,all_text_raw,label1,label2,label3\n,text,0,0,0\n",
        "classifieds_unique_listing_id,all_text_raw,label1,label2,label3\na, ,0,0,0\n",
        "classifieds_unique_listing_id,all_text_raw,label1,label2,label3\na,text,0,0,0\na,x,1,0,0\n",
    ],
)
def test_invalid_source_leaves_outputs_untouched(tmp_path: Path, content: str) -> None:
    source = tmp_path / "source.csv"
    source.write_text(content)
    outputs = [tmp_path / name for name in ("out.csv", "na.csv", "audit.json")]
    for path in outputs:
        path.write_text("sentinel")
    with pytest.raises(ValueError):
        prepare_birds(source, *outputs)
    assert all(path.read_text() == "sentinel" for path in outputs)
    assert source.read_text() == content


def test_protect_source_and_output_paths(tmp_path: Path) -> None:
    source = source_csv(tmp_path / "source.csv")
    with pytest.raises(ValueError, match="distinct"):
        prepare_birds(source, source, tmp_path / "na.csv", tmp_path / "audit.json")
    with pytest.raises(ValueError, match="distinct"):
        prepare_birds(source, tmp_path / "out", tmp_path / "out", tmp_path / "audit.json")
    assert main(["--source", str(tmp_path / "absent.csv")]) == 1
