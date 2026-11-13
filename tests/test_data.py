from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from wildlife_screening.cli import main
from wildlife_screening.data import inspect_data, load_data, numeric_labels
from wildlife_screening.types import EvaluationData


def test_allowlist(data: EvaluationData) -> None:
    assert len(data.adverts) == 20
    assert sum(data.labels) == 5
    assert set(asdict(data.adverts[0])) == {"advert_id", "title", "description", "title_and_desc"}
    assert data.adverts[0].description is None
    assert "DO_NOT_SEND" not in str(data.adverts)


def test_description_opt_in() -> None:
    from conftest import ROOT

    data = load_data(ROOT / "tests/fixtures/synthetic20.csv", ("title", "description"))
    assert data.adverts[0].description == "Synthetic description 00"
    with pytest.raises(ValueError, match="allowlist"):
        load_data(ROOT / "tests/fixtures/synthetic20.csv", ("title", "seller"))


@pytest.mark.parametrize(
    "rows",
    [
        [{"label": "0"}, {"label": "1"}],
        [{"label": "1", "relevant": "animal origin"}, {"label": "0", "relevant": "animal origin"}],
        [{"label": "1", "relevant": "maybe"}],
        [{"label": "-1"}],
        [{"label": ""}],
    ],
)
def test_ambiguous_mapping_fails(rows: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError):
        numeric_labels(rows)


def test_inverse_mapping_and_missing() -> None:
    assert numeric_labels(
        [{"label": "0", "relevant": "animal origin"}, {"label": "1", "relevant": ""}]
    ) == [1, 0]
    assert numeric_labels([{"label": "0"}, {"label": "1"}], "1") == [0, 1]


def test_inspection_disagreements_duplicates_and_spots(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "label", "relevant"])
        writer.writeheader()
        writer.writerows(
            [
                {"title": "synthetic fossil", "label": "0", "relevant": "not animal origin"},
                {"title": "synthetic fossil", "label": "1", "relevant": "not animal origin"},
                {"title": "synthetic replica", "label": "0", "relevant": ""},
            ]
        )
    ambiguous = inspect_data(path, path)
    assert ambiguous["mapping_error"]
    assert len(ambiguous["disagreements"]) == 2
    result = inspect_data(path, path, "1")
    assert result["positives"] == 1
    assert len(result["disagreements"]) == 1
    assert result["disagreements"][0]["row"] == 3
    assert result["duplicate_titles"][0]["rows"] == [2, 3]
    assert len(result["spot_checks"]["fossil"]) == 2
    assert len(result["spot_checks"]["replica"]) == 1
    assert result["pool"]["label_values"] == {"0": 2, "1": 1}


def test_duplicate_ids_fail(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("id,title,label,relevant\na,x,1,animal origin\na,y,0,not animal origin\n")
    with pytest.raises(ValueError, match="Duplicate advert"):
        load_data(p)


def test_default_shark_inspection_with_optional_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir()
    validation = Path("data/validation_sharks.csv")
    validation.write_text("title,label,relevant\nInvented jaw,1,animal origin\n")
    output = tmp_path / "inspection.json"
    inspect = ["inspect", "--output", str(output)]
    assert main(inspect) == 0
    assert "pool" not in json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out)["pool"].startswith("not inspected")

    pool = Path("data/shark_trophy.csv")
    assert main([*inspect, "--pool", str(pool)]) == 1
    pool.write_text("title,label\nInvented tooth,0\n")
    capsys.readouterr()
    assert main(inspect) == 0
    assert json.loads(output.read_text())["pool"]["rows"] == 1
    assert json.loads(capsys.readouterr().out)["pool"] == f"included: {pool}"

    custom = [*inspect, "--data", str(validation)]
    assert main(custom) == 0
    assert "pool" not in json.loads(output.read_text())
    assert main([*custom, "--pool", str(pool)]) == 0
    assert json.loads(output.read_text())["pool"]["label_values"] == {"0": 1}
