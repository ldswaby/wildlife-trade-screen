"""Verify explicit combined inputs, historical cache separation and offline reports."""

import csv
import json
from pathlib import Path

import pytest
import yaml
from report_support import render_evidence

from wildlife_screening.cli import main, parser
from wildlife_screening.config import ModelConfig
from wildlife_screening.data import inspect_data, load_data
from wildlife_screening.prompts import Prompt
from wildlife_screening.runner import run_arm
from wildlife_screening.scorers.api import OpenAIDecisionsScorer, PromptedLLMScorer, SystemOneScorer
from wildlife_screening.scorers.local import FakeScorer
from wildlife_screening.snapshots import _identity, _load_pinned
from wildlife_screening.types import Advert
from wildlife_screening.utils import hash_json

ROOT = Path(__file__).resolve().parents[1]


def combined_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["advert_id", "title_and_desc", "label", "seller"])
        writer.writerows(
            [
                ["one", "Invented bird. Full description.", "1", "DO_NOT_SEND"],
                ["two", "Invented empty cage.", "0", "DO_NOT_SEND"],
            ]
        )
    return path


def test_combined_allowlist_and_provider_evidence(tmp_path: Path) -> None:
    path = combined_csv(tmp_path / "combined.csv")
    with pytest.raises(ValueError, match="Missing allowlisted"):
        load_data(path, positive_label="1")
    data = load_data(path, ("title_and_desc",), "1")
    advert = data.adverts[0]
    assert advert.title == "" and advert.description is None
    assert advert.text() == "advert text: Invented bird. Full description."
    assert "DO_NOT_SEND" not in str(data.adverts)
    report = inspect_data(path, positive_label="1")
    assert report["title_and_desc_present"] == 2
    assert report["title_present"] == 0 and report["duplicate_titles"] == []
    for fields in [("title", "title_and_desc"), ("title_and_desc", "seller")]:
        with pytest.raises(ValueError, match="allowlist"):
            load_data(path, fields, "1")
    with pytest.raises(ValueError, match="cannot be mixed"):
        Advert("one", "title", title_and_desc="combined")
    native = Prompt.load(ROOT / "prompts/sharks/decision_zeroshot_v0.yaml")
    chat = Prompt.load(ROOT / "prompts/sharks/llm_zeroshot_v0.txt")
    for scorer_type, config_name, prompt in [
        (PromptedLLMScorer, "gpt6_luna_chat_anchor", chat),
        (OpenAIDecisionsScorer, "gpt6_luna_decisions", native),
        (SystemOneScorer, "microsoft_decision1_openrouter", native),
    ]:
        scorer = scorer_type(ModelConfig.load(ROOT / f"configs/models/{config_name}.yaml"), prompt)
        payload = json.dumps(scorer.payload(advert))
        assert advert.text() in payload
        assert "DO_NOT_SEND" not in payload and '"label"' not in payload


def test_combined_cli_cache_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = combined_csv(tmp_path / "combined.csv")
    monkeypatch.chdir(tmp_path)
    args = [
        "run",
        "--data",
        str(path),
        "--positive-label",
        "1",
        "--combined-text",
        "--model",
        str(ROOT / "configs/models/fake.yaml"),
        "--decision-prompt",
        str(ROOT / "prompts/sharks/decision_zeroshot_v0.yaml"),
        "--zero-shot",
        "--cache",
        str(tmp_path / "cache"),
    ]
    assert main(args) == 0
    pin = next((tmp_path / "outputs/comparison").rglob("manifest-*.json"))
    assert "combined_text" in pin.parts
    manifest = json.loads(pin.read_text())
    assert manifest["allowlisted_fields"] == ["title_and_desc"]
    assert manifest["input_condition"] == "combined_text"
    assert _identity(manifest)["input"] == "combined_text"
    selection = {
        "pins": [str(pin)],
        "datasets": {
            f"sharks-{manifest['data_file_hash'][:12]}": {
                "data": str(path),
                "positive_label": "1",
            }
        },
    }
    row, _ = _load_pinned(pin, selection)
    assert row["n"] == 2 and row["input"] == "combined_text"
    assert main(args) == 0
    manifests = [json.loads(p.read_text()) for p in pin.parent.glob("manifest-*.json")]
    assert sorted(m["estimate"]["cached_adverts"] for m in manifests) == [0, 2]
    output = pin.parent.parent
    with (output / "table1.csv").open() as stream:
        assert next(csv.DictReader(stream))["input_condition"] == "combined_text"
    assert "combined text" in (output / "table1.md").read_text()
    selection_path = tmp_path / "selection.yaml"
    selection_path.write_text(yaml.safe_dump(selection))
    html = tmp_path / "report.html"
    render_evidence(selection_path, html)
    assert "title + desc" in html.read_text()
    assert "Invented bird. Full description." not in html.read_text()
    with pytest.raises(SystemExit):
        parser().parse_args(args + ["--include-description"])


def test_combined_cache_is_separate_from_legacy_input(
    tmp_path: Path,
) -> None:
    config = ModelConfig.load(ROOT / "configs/models/fake.yaml")
    assert (
        len(
            {
                config.fingerprint(fields)
                for fields in [("title",), ("title", "description"), ("title_and_desc",)]
            }
        )
        == 3
    )
    path = combined_csv(tmp_path / "combined.csv")
    data = load_data(path, ("title_and_desc",), "1")
    prompt = Prompt.load(ROOT / "prompts/sharks/decision_zeroshot_v0.yaml")
    scorer = FakeScorer()
    _, first = run_arm(
        data,
        config,
        prompt,
        tmp_path / "out",
        tmp_path / "cache",
        fields=("title_and_desc",),
        scorer=scorer,
    )
    _, repeat = run_arm(
        data,
        config,
        prompt,
        tmp_path / "out",
        tmp_path / "cache",
        fields=("title_and_desc",),
        scorer=scorer,
    )
    assert scorer.calls == 2
    assert (
        json.loads(first.read_text())["model_config_hash"]
        == json.loads(repeat.read_text())["model_config_hash"]
    )


@pytest.mark.parametrize("fields", [("title",), ("title", "description")])
def test_historical_input_hash_and_serialization_are_unchanged(
    tmp_path: Path, fields: tuple[str, ...]
) -> None:
    data = load_data(ROOT / "tests/fixtures/synthetic20.csv", fields)
    config = ModelConfig.load(ROOT / "configs/models/fake.yaml")
    prompt = Prompt.load(ROOT / "prompts/sharks/decision_zeroshot_v0.yaml")
    scorer = FakeScorer()
    run_arm(
        data, config, prompt, tmp_path / "out", tmp_path / "cache", fields=fields, scorer=scorer
    )
    for advert in data.adverts:
        old_shape = {
            "advert_id": advert.advert_id,
            "title": advert.title,
            "description": advert.description,
        }
        entry = (
            tmp_path
            / "cache"
            / config.fingerprint(fields)
            / prompt.sha256
            / (hash_json(advert.advert_id) + ".json")
        )
        assert json.loads(entry.read_text())["input_hash"] == hash_json(old_shape)
        assert advert.text() == f"title: {advert.title}" + (
            f"\ndescription: {advert.description}" if advert.description is not None else ""
        )
    run_arm(
        data, config, prompt, tmp_path / "out", tmp_path / "cache", fields=fields, scorer=scorer
    )
    assert scorer.calls == 20
