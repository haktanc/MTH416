"""Test integrity and similarity screening on temporary fixture images only."""

import hashlib
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest

from src.dataset_audit import (
    assert_raw_unchanged,
    audit_dataset,
    perceptual_hash,
    transform_image,
    write_audit_reports,
)
from src.reporting import output_directory


def make_images(root: Path, counts: dict[str, int]) -> Path:
    """Create deterministic grayscale JPEG fixtures, never real raw images."""
    primary = root / "data/raw/casting_512x512"
    random = np.random.default_rng(123)
    for class_name, count in counts.items():
        directory = primary / class_name
        directory.mkdir(parents=True)
        for index in range(count):
            small = random.integers(0, 256, (32, 32), dtype=np.uint8)
            image = Image.fromarray(small).resize((512, 512))
            image.convert("RGB").save(directory / f"fixture_{index}.jpeg")
    return primary


@pytest.fixture
def primary(tmp_path):
    counts = {"def_front": 2, "ok_front": 2}
    return make_images(tmp_path, counts), counts, tmp_path


def run_audit(primary, **kwargs):
    directory, counts, root = primary
    return audit_dataset(directory, root, expected_counts=counts, **kwargs)


def test_full_readability_and_actual_counts(primary):
    audit = run_audit(primary)
    assert audit.summary["readable_count"] == 4
    assert audit.summary["class_counts"] == primary[1]
    assert audit.summary["dimensions"] == {"512x512": 4}
    assert audit.summary["modes"] == {"RGB": 4}
    assert audit.summary["grayscale_count"] == 4
    assert not audit.summary["blocking_issues"]
    assert_raw_unchanged(audit, primary[2])


def test_exact_file_duplicates(primary):
    directory, _, _ = primary
    source = directory / "def_front/fixture_0.jpeg"
    (directory / "def_front/fixture_1.jpeg").write_bytes(source.read_bytes())
    audit = run_audit(primary)
    assert audit.summary["exact_file_pairs"] == 1
    assert audit.summary["exact_pixel_only_pairs"] == 0
    assert audit.exact_pairs[0]["class_a"] == "def_front"


def test_same_pixels_with_different_jpeg_metadata(primary):
    directory, _, _ = primary
    original = (directory / "def_front/fixture_0.jpeg").read_bytes()
    comment = b"Temporary test-only JPEG comment"
    segment = b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment
    (directory / "def_front/fixture_1.jpeg").write_bytes(
        original[:2] + segment + original[2:]
    )
    audit = run_audit(primary)
    assert audit.summary["exact_file_pairs"] == 0
    assert audit.summary["exact_pixel_only_pairs"] == 1


def test_cross_label_duplicate_is_blocking(primary):
    directory, _, _ = primary
    original = (directory / "def_front/fixture_0.jpeg").read_bytes()
    (directory / "ok_front/fixture_0.jpeg").write_bytes(original)
    audit = run_audit(primary)
    assert "Exact duplicates have conflicting class labels" in audit.summary["blocking_issues"]


@pytest.mark.parametrize("content", [b"", b"not a JPEG"])
def test_corrupt_image_is_reported_and_not_dropped(primary, content):
    (primary[0] / "def_front/fixture_0.jpeg").write_bytes(content)
    audit = run_audit(primary)
    assert len(audit.rows) == 4
    assert audit.summary["readable_count"] == 3
    assert audit.summary["blocking_issues"]
    assert any(row["error"] for row in audit.rows)


def test_truncated_jpeg_is_rejected(primary):
    path = primary[0] / "def_front/fixture_0.jpeg"
    path.write_bytes(path.read_bytes()[:100])
    audit = run_audit(primary)
    assert audit.summary["eligible_count"] == 3


def test_wrong_size_and_color_are_reported(primary):
    Image.new("RGB", (64, 64), "red").save(
        primary[0] / "def_front/fixture_0.jpeg"
    )
    audit = run_audit(primary)
    invalid = [row for row in audit.rows if not row["eligible"]]
    assert len(invalid) == 1
    assert "dimensions" in invalid[0]["error"]
    assert "Non-grayscale" in invalid[0]["error"]


def test_count_mismatch_blocks_preparation(primary):
    audit = audit_dataset(primary[0], primary[2], expected_counts={
        "def_front": 3, "ok_front": 2,
    })
    assert any("Count mismatch" in issue for issue in audit.summary["blocking_issues"])


def test_rotated_and_mirrored_screen_is_not_a_physical_id():
    image = np.random.default_rng(7).integers(0, 256, (512, 512), dtype=np.uint8)
    mirrored = transform_image(image, 4)
    assert perceptual_hash(image) == perceptual_hash(transform_image(mirrored, 4))
    assert hashlib.sha256(image.tobytes()).digest() != hashlib.sha256(mirrored.tobytes()).digest()


def test_similarity_candidates_remain_unresolved(primary):
    audit = run_audit(primary, near_threshold=63)
    assert len(audit.near_pairs) == 6
    assert {pair["review_status"] for pair in audit.near_pairs} == {"unresolved"}
    assert not audit.summary["physical_part_groups_verified"]


def test_reports_are_deterministic_and_do_not_change_raw(primary):
    audit = run_audit(primary, near_threshold=63)
    directory = primary[2] / "results/tables"
    write_audit_reports(audit, directory, primary[2])
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    write_audit_reports(run_audit(primary, near_threshold=63), directory, primary[2])
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert "unresolved" in (directory / "near_duplicate_review.html").read_text()
    assert_raw_unchanged(audit, primary[2])


def test_changed_or_added_raw_files_are_detected(primary):
    audit = run_audit(primary)
    path = primary[0] / "def_front/fixture_0.jpeg"
    path.write_bytes(path.read_bytes() + b"changed test fixture")
    with pytest.raises(ValueError, match="Raw image changed"):
        assert_raw_unchanged(audit, primary[2])


def test_added_raw_file_is_detected(primary):
    audit = run_audit(primary)
    path = primary[0] / "def_front/fixture_0.jpeg"
    (path.parent / "new_fixture.jpeg").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="membership changed"):
        assert_raw_unchanged(audit, primary[2])


def test_raw_output_target_is_rejected(primary):
    with pytest.raises(ValueError, match="outside raw data"):
        output_directory(primary[0], primary[2])


def test_augmented_collection_cannot_be_selected(primary):
    with pytest.raises(ValueError, match="only data/raw/casting_512x512"):
        audit_dataset(primary[2] / "data/raw/casting_data", primary[2])


@pytest.mark.parametrize("threshold", [-1, 64])
def test_invalid_similarity_threshold(primary, threshold):
    with pytest.raises(ValueError, match="threshold"):
        run_audit(primary, near_threshold=threshold)
