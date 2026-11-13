"""Check the standalone export boundary with invented snapshots, never private runs."""

import csv
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from wildlife_screening import snapshots as saved_snapshots
from wildlife_screening.config import ModelConfig
from wildlife_screening.types import ScoreResult
from wildlife_screening.utils import file_hash, hash_json

SPEC = importlib.util.spec_from_file_location(
    "aggregate_export", Path(__file__).resolve().parents[1] / "scripts/export_aggregate_results.py"
)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def snapshot(
    root: Path,
    *,
    model: str = "invented",
    binary: bool = False,
    failure: bool = False,
    adapter: str = "openai_decisions",
    unknown_cost: bool = False,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    data = root / "invented.csv"
    data.write_text(
        "id,title,label\na,PRIVATE ADVERT,1\nb,Invented toy,0\n"
        "c,Invented item,1\nd,Invented object,0\n"
    )
    config = ModelConfig(
        arm="PRIVATE ARM NOTE",
        model_id=model,
        adapter=adapter,
        endpoint="http://localhost:9999" if adapter == "system_one" else "https://example.invalid",
        api_key_env="PRIVATE_KEY_NAME",
        options={"path": str(data)} if adapter == "precomputed" else {},
        billable=adapter != "system_one",
    )
    manifest = {
        "task": "animals",
        "data_file_hash": file_hash(data),
        "allowlisted_fields": ["title"],
        "condition": "zero-shot",
        "model_config": asdict(config),
        "positive_label": "1",
        "prompt_hash": "a" * 64,
        "prompt_family": "decision",
        "example_count": 0,
        "model_config_hash": hash_json(
            {"model_config_hash": config.fingerprint(("title",)), "task": "animals"}
        ),
        "status": "incomplete" if failure else "complete",
        "arm": config.arm,
        "seller": "PRIVATE SELLER",
        "note": "PRIVATE NOTE",
    }
    scores = [
        asdict(
            ScoreResult(
                advert_id=name,
                p_relevant=None if failure and name == "c" else score,
                score_type="binary" if binary else "native_prob",
                raw_output="PRIVATE RESPONSE",
                tokens_in=10,
                tokens_out=1,
                cost=None if unknown_cost else 0.01,
                latency_s=1.0,
                error="PRIVATE ERROR" if failure and name == "c" else None,
            )
        )
        for name, score in zip(
            "abcd", [1.0, 1.0, 0.0, 0.0] if binary else [0.9, 0.8, 0.1, 0.2], strict=True
        )
    ]
    pin = root / "manifest-invented.json"
    pin.write_text(json.dumps(manifest))
    pin.with_name("scores-invented.json").write_text(json.dumps(scores))
    return pin, data


def selection(root: Path, snapshots: list[tuple[Path, Path]]) -> Path:
    path = root / "selection.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "pins": [str(pin) for pin, _ in snapshots],
                "datasets": {
                    f"animals-{file_hash(data)[:12]}": {"data": str(data)} for _, data in snapshots
                },
            }
        )
    )
    return path


def test_counts_rates_and_private_fields(tmp_path: Path) -> None:
    pin = snapshot(tmp_path / "run", failure=True)
    selected = selection(tmp_path, [pin])
    output = tmp_path / "public.csv"
    assert EXPORTER.export_aggregates(selected, output) == 1
    with output.open() as stream:
        row = next(csv.DictReader(stream))
    assert tuple(row) == EXPORTER.COLUMNS
    assert [int(row[k]) for k in ("tp", "fp", "tn", "fn")] == [2, 1, 1, 0]
    assert row["n"] == "4" and row["failures"] == "1"
    assert float(row["coverage"]) == 0.75 and float(row["balanced_accuracy"]) == 0.75
    assert float(row["precision"]) == pytest.approx(2 / 3)
    assert float(row["api_cost_usd"]) == 0.04
    assert float(row["summed_request_seconds_per_1000_adverts"]) == 1000
    for forbidden in ("PRIVATE", "example.invalid", "localhost", str(tmp_path), "raw_output"):
        assert forbidden not in output.read_text()


@pytest.mark.parametrize(
    "adapter,scope",
    [
        ("openai_decisions", "unknown"),
        ("system_one", "external_api_only_local_infrastructure_unknown"),
        ("fake", "not_measured"),
        ("precomputed", "not_measured"),
    ],
)
def test_unknown_and_local_resource_scope(tmp_path: Path, adapter: str, scope: str) -> None:
    selected = selection(tmp_path, [snapshot(tmp_path / "run", adapter=adapter, unknown_cost=True)])
    row = EXPORTER.aggregate_rows(selected)[0]
    assert row["api_cost_scope"] == scope
    assert row["api_cost_usd"] == (0 if adapter == "system_one" else None)
    if adapter in {"fake", "precomputed"}:
        assert row["summed_request_seconds"] is None


def test_binary_blanks_and_deterministic_order(tmp_path: Path) -> None:
    a = snapshot(tmp_path / "a", model="a", binary=True)
    b = snapshot(tmp_path / "b", model="b")
    selected = selection(tmp_path, [b, a])
    output = tmp_path / "public.csv"
    EXPORTER.export_aggregates(selected, output)
    first = output.read_bytes()
    selected = selection(tmp_path, [a, b])
    EXPORTER.export_aggregates(selected, output)
    assert output.read_bytes() == first
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["model_id"] for r in rows] == ["a", "b"]
    assert rows[0]["cutoff"] == rows[0]["fpr_at_95_tpr"] == ""
    assert rows[1]["cutoff"] == "0.5"


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "json",
        "duplicate_pin",
        "duplicate_config",
        "data_hash",
        "config_hash",
        "polarity",
        "score_ids",
        "score_coverage",
        "policy",
    ],
)
def test_bad_selection_preserves_destination(tmp_path: Path, damage: str) -> None:
    pin, data = snapshot(tmp_path / "a")
    selected = selection(tmp_path, [(pin, data)])
    manifest: dict[str, Any] = json.loads(pin.read_text())
    if damage == "missing":
        pin.unlink()
    elif damage == "json":
        pin.write_text("invalid JSON")
    elif damage == "duplicate_pin":
        selection(tmp_path, [(pin, data), (pin, data)])
    elif damage == "duplicate_config":
        other = snapshot(tmp_path / "b")
        selection(tmp_path, [(pin, data), other])
    elif damage in {"data_hash", "config_hash", "polarity"}:
        manifest[
            {
                "data_hash": "data_file_hash",
                "config_hash": "model_config_hash",
                "polarity": "positive_label",
            }[damage]
        ] = "0" if damage == "polarity" else "b" * 64
        pin.write_text(json.dumps(manifest))
    elif damage in {"score_ids", "score_coverage"}:
        scores_path = pin.with_name("scores-invented.json")
        scores = json.loads(scores_path.read_text())
        if damage == "score_ids":
            scores.reverse()
        else:
            scores.pop()
        scores_path.write_text(json.dumps(scores))
    else:
        content = yaml.safe_load(selected.read_text())
        content["evaluation_policy"] = "invented-policy"
        selected.write_text(yaml.safe_dump(content))
    output = tmp_path / "public.csv"
    output.write_bytes(b"existing export\n")
    with pytest.raises((OSError, ValueError, KeyError, TypeError)):
        EXPORTER.export_aggregates(selected, output)
    assert output.read_bytes() == b"existing export\n"
    assert EXPORTER.main(["--selection", str(selected), "--output", str(output)]) == 1


def test_write_failure_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = selection(tmp_path, [snapshot(tmp_path / "run")])
    output = tmp_path / "public.csv"
    output.write_bytes(b"old\n")

    def fail_replace(*args: Any) -> None:
        raise OSError("invented write failure")

    monkeypatch.setattr(EXPORTER.os, "replace", fail_replace)
    with pytest.raises(OSError):
        EXPORTER.export_aggregates(selected, output)
    assert output.read_bytes() == b"old\n"
    assert {p.name for p in tmp_path.iterdir()} == {"run", "selection.yaml", "public.csv"}


def test_export_skips_only_curve_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin, data = snapshot(tmp_path / "run")
    selected = selection(tmp_path, [(pin, data)])
    settings = yaml.safe_load(selected.read_text())
    original, _ = saved_snapshots._load_pinned(pin, settings)

    def forbidden(*args: Any) -> None:
        raise AssertionError("Aggregate export must not compute ROC coordinates")

    monkeypatch.setattr(saved_snapshots, "policy_roc_curve", forbidden)
    aggregate, _ = saved_snapshots._load_pinned(pin, settings, include_curve=False)
    assert aggregate["metrics"] == original["metrics"]
    assert aggregate["cost"] == original["cost"]
    assert aggregate["latency_s_total"] == original["latency_s_total"]
    assert original["points"] and aggregate["points"] == []
    assert EXPORTER.aggregate_rows(selected)[0]["balanced_accuracy"] == 0.5


def test_release_checksum_rejects_changed_score_file(tmp_path: Path) -> None:
    pin, data = snapshot(tmp_path / "run")
    selected = selection(tmp_path, [(pin, data)])
    scores = pin.with_name("scores-invented.json")
    manifest = json.loads(pin.read_text())
    manifest["scores_sha256"] = file_hash(scores)
    pin.write_text(json.dumps(manifest))
    assert len(EXPORTER.aggregate_rows(selected)) == 1
    scores.write_bytes(scores.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        EXPORTER.aggregate_rows(selected)
