"""Deterministic report serialization with raw-data write protection."""

import csv
import hashlib
import json
from pathlib import Path

from src.data_loader import PROJECT_ROOT


def output_directory(path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Allow report writes inside the repository, outside its raw directory."""
    path = Path(path).resolve()
    root = Path(project_root).resolve()
    raw = (root / "data" / "raw").resolve()
    if not path.is_relative_to(root) or path.is_relative_to(raw):
        raise ValueError("Outputs must be inside the project and outside raw data.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    """Serialize CSV consistently across operating systems."""
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    """Write an already validated report target."""
    path.write_bytes(csv_bytes(rows, fields))


def write_json(path: Path, value: dict) -> None:
    """Write a readable JSON report without volatile timestamps."""
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def inventory_digest(rows: list[dict]) -> str:
    """Fingerprint paths and file contents, independent of row order."""
    payload = sorted((row["image_path"], row["sha256"]) for row in rows)
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
