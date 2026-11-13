from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, NoReturn

import pytest

from wildlife_screening.data import load_data
from wildlife_screening.prompts import Prompt
from wildlife_screening.types import EvaluationData

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("Tests must run offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def data() -> EvaluationData:
    return load_data(ROOT / "tests/fixtures/synthetic20.csv")


@pytest.fixture
def prompt() -> Prompt:
    return Prompt.load(ROOT / "prompts/sharks/decision_zeroshot_v0.yaml")
