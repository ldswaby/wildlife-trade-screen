"""Stable hashes and atomic files make local runs auditable and resumable."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def digest(value: bytes) -> str:
    """Return a SHA-256 hex digest for cache and provenance checks."""
    return hashlib.sha256(value).hexdigest()


def hash_json(value: Any) -> str:
    """Hash JSON deterministically so dictionary ordering cannot change cache identity."""
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def file_hash(path: Path) -> str:
    """Hash the original file bytes to detect data or prompt changes."""
    return digest(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON file atomically so interrupted runs cannot leave partial records.

    Args:
        path: Destination file; missing parent directories are created.
        value: JSON-serialisable content. NaN and infinity are rejected.

    Raises:
        ValueError: Content contains non-finite JSON numbers.
        OSError: Writing, syncing or replacing the file fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
