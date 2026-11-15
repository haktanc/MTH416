"""Locate and inspect the existing dataset without changing its contents."""

from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


CLASS_LABELS = {"ok_front": 0, "def_front": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_DATASET = PROJECT_ROOT / "data" / "raw" / "casting_512x512"
EXPECTED_PRIMARY_COUNTS = {"def_front": 781, "ok_front": 519}
MANIFEST_FIELDS = ["image_path", "label", "class_name", "split", "image_id"]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str
    class_name: str
    label: int


def collect_primary_records(
    dataset_directory: Path = PRIMARY_DATASET,
) -> list[ImageRecord]:
    """Index the flat, unsplit collection without assigning train/test sets.

    Paths are resolved explicitly so that the augmented collection cannot be
    selected accidentally. Future split manifests can reference these paths.
    """
    root = Path(dataset_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    unexpected = sorted(
        path.name for path in root.iterdir()
        if not path.is_dir() or path.name not in CLASS_LABELS
    )
    if unexpected:
        raise ValueError(f"Unexpected dataset entries: {unexpected}")

    for name in CLASS_LABELS:
        directory = root / name
        if not directory.is_dir():
            raise ValueError(f"Missing class directory: {directory}")

    records = []
    for name, label in CLASS_LABELS.items():
        directory = root / name
        entries = sorted(directory.iterdir())
        invalid = [
            path.name for path in entries
            if (
                not path.is_file()
                or path.suffix.lower() not in IMAGE_EXTENSIONS
            )
        ]
        if invalid:
            raise ValueError(f"Unexpected entries in {directory}: {invalid}")
        if not entries:
            raise ValueError(f"No images found in {directory}")
        records.extend(
            ImageRecord(path, "unsplit", name, label) for path in entries
        )
    return records


def primary_image_counts(records: list[ImageRecord]) -> dict[str, int]:
    """Count primary images without assigning training/testing partitions."""
    return {
        name: sum(record.class_name == name for record in records)
        for name in CLASS_LABELS
    }


def image_identifier(relative_path: str) -> str:
    """Identify a file path without claiming a physical-part identity."""
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def resolve_manifest_path(
    relative_path: str, project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a canonical repository-relative path in the primary collection."""
    relative = PurePosixPath(relative_path)
    if (
        not relative_path or relative.is_absolute() or ".." in relative.parts
        or "\\" in relative_path or ":" in relative_path
        or relative.as_posix() != relative_path
    ):
        raise ValueError(f"Invalid repository-relative image path: {relative_path}")
    root = Path(project_root).resolve()
    path = (root / relative_path).resolve()
    primary = (root / "data" / "raw" / "casting_512x512").resolve()
    if not path.is_relative_to(primary) or not path.is_file():
        raise ValueError(f"Image is missing or outside the primary set: {path}")
    return path


def read_manifest(
    path: Path, expected_split: str, project_root: Path = PROJECT_ROOT,
) -> list[dict]:
    """Validate a manifest's paths, identities, labels, and split membership."""
    if expected_split not in {"train", "test"}:
        raise ValueError("Expected split must be train or test.")
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(f"Unexpected manifest schema: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Manifest row has missing or extra CSV values.")
        image_path = resolve_manifest_path(row["image_path"], project_root)
        name = row["class_name"]
        if name not in CLASS_LABELS or image_path.parent.name != name:
            raise ValueError("Manifest class does not match the image directory.")
        if row["label"] != str(CLASS_LABELS[name]):
            raise ValueError("Manifest label does not match its class.")
        if row["split"] != expected_split:
            raise ValueError("Manifest contains an unexpected split.")
        if row["image_id"] != image_identifier(row["image_path"]):
            raise ValueError("Manifest image identity does not match its path.")
        if row["image_path"] in seen:
            raise ValueError("Manifest contains a repeated image path.")
        seen.add(row["image_path"])
        row["label"] = int(row["label"])
    if rows != sorted(rows, key=lambda row: row["image_path"]):
        raise ValueError("Manifest paths must be deterministically sorted.")
    return rows


def find_dataset(raw_directory: Path) -> Path:
    """Find exactly one root with train/test and both expected classes."""
    raw_directory = Path(raw_directory).resolve()
    if not raw_directory.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {raw_directory}"
        )
    candidates = [raw_directory]
    candidates.extend(path.parent for path in raw_directory.rglob("train"))
    matches = sorted({
        root for root in candidates
        if all((root / split / name).is_dir()
               for split in ("train", "test") for name in CLASS_LABELS)
    })
    if len(matches) != 1:
        raise ValueError(
            f"Expected one train/test dataset, found {len(matches)}."
        )
    return matches[0]


def collect_records(dataset_directory: Path) -> list[ImageRecord]:
    """Index files deterministically; never mix train and test images."""
    records = []
    for split in ("train", "test"):
        split_directory = Path(dataset_directory) / split
        unexpected = [p.name for p in split_directory.iterdir()
                      if p.is_dir() and p.name not in CLASS_LABELS]
        if unexpected:
            raise ValueError(f"Unexpected classes in {split}: {unexpected}")
        for name, label in CLASS_LABELS.items():
            directory = split_directory / name
            paths = sorted(
                p for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not paths:
                raise ValueError(f"No images found in {directory}")
            records.extend(
                ImageRecord(p, split, name, label) for p in paths
            )
    return records


def load_grayscale(path: Path) -> np.ndarray:
    """Decode grayscale images, including paths with non-ASCII characters."""
    encoded = np.fromfile(Path(path), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Cannot decode empty image: {path}")
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    except cv2.error as error:
        raise ValueError(f"Cannot decode image: {path}") from error
    if image is None or image.size == 0:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def image_counts(records: list[ImageRecord]) -> list[dict]:
    """Return one count per split and class for console and CSV reporting."""
    return [
        {
            "split": split,
            "class_name": name,
            "label": label,
            "image_count": sum(
                r.split == split and r.label == label for r in records
            ),
        }
        for split in ("train", "test") for name, label in CLASS_LABELS.items()
    ]


def check_integrity(records: list[ImageRecord]) -> list[dict[str, str]]:
    """Decode indexed images and report failures without repairing files."""
    failures = []
    for record in records:
        try:
            load_grayscale(record.path)
        except (OSError, ValueError, cv2.error) as error:
            failures.append({"path": str(record.path), "error": str(error)})
    return failures
