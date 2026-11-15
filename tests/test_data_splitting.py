"""Exercise leakage guards and reproducibility without extracting features."""

from copy import deepcopy
import hashlib

import numpy as np
import pytest

from src.data_loader import MANIFEST_FIELDS, image_identifier, read_manifest
from src.dataset_audit import AuditResult
import src.prepare_splits as preparation
from src.prepare_splits import (
    cv_assignments,
    generate_cv_indices,
    independence_groups,
    make_partitions,
    validate_partitions,
)
from src.reporting import csv_bytes, write_csv
from tests.test_dataset_audit import make_images


@pytest.fixture
def inventory(tmp_path):
    """Use real tiny fixture files to isolate split logic from image decoding."""
    rows = []
    for name, label in (("def_front", 1), ("ok_front", 0)):
        directory = tmp_path / "data/raw/casting_512x512" / name
        directory.mkdir(parents=True)
        for index in range(30):
            path = directory / f"fixture_{index:03}.jpeg"
            content = f"Temporary identity fixture {name} {index}".encode()
            path.write_bytes(content)
            relative = path.relative_to(tmp_path).as_posix()
            digest = hashlib.sha256(content).hexdigest()
            rows.append({
                "image_path": relative, "image_id": image_identifier(relative),
                "class_name": name, "label": label, "sha256": digest,
                "pixel_sha256": digest, "eligible": True,
            })
    audit = AuditResult(rows, [], [], {"blocking_issues": []})
    return audit, tmp_path


def test_stratified_split_counts_and_reproducibility(inventory):
    audit, root = inventory
    groups, _ = independence_groups(audit)
    train, test, strategy = make_partitions(audit, groups)
    assert (len(train), len(test)) == (48, 12)
    assert sum(row["label"] for row in train) == 24
    assert sum(row["label"] for row in test) == 6
    assert strategy == "stratified_image_split"
    again_train, again_test, _ = make_partitions(audit, groups)
    assert csv_bytes(train, MANIFEST_FIELDS) == csv_bytes(again_train, MANIFEST_FIELDS)
    assert csv_bytes(test, MANIFEST_FIELDS) == csv_bytes(again_test, MANIFEST_FIELDS)
    assert test != make_partitions(audit, groups, seed=43)[1]
    validate_partitions(train, test, audit, groups, root)


def test_training_only_five_fold_cv(inventory):
    audit, _ = inventory
    groups, _ = independence_groups(audit)
    train, test, _ = make_partitions(audit, groups)
    folds, strategy = generate_cv_indices(train, test, audit, groups)
    assert strategy == "StratifiedKFold"
    assert len(folds) == 5
    validations = np.concatenate([validation for _, validation in folds])
    np.testing.assert_array_equal(np.sort(validations), np.arange(len(train)))
    test_ids = {row["image_id"] for row in test}
    for fit, validation in folds:
        assert not set(fit) & set(validation)
        assert not {train[int(i)]["image_id"] for i in fit} & test_ids
        assert not {train[int(i)]["image_id"] for i in validation} & test_ids
    second, _ = generate_cv_indices(train, test, audit, groups)
    assert cv_assignments(train, folds, groups) == cv_assignments(train, second, groups)


def test_exact_content_groups_remain_isolated(inventory):
    audit, _ = inventory
    audit.rows[1]["sha256"] = audit.rows[0]["sha256"]
    audit.rows[1]["pixel_sha256"] = audit.rows[0]["pixel_sha256"]
    groups, _ = independence_groups(audit)
    assert groups[audit.rows[0]["image_path"]] == groups[audit.rows[1]["image_path"]]
    train, test, strategy = make_partitions(audit, groups)
    assert strategy.startswith("group_shuffle")
    folds, _ = generate_cv_indices(train, test, audit, groups)
    for fit, validation in folds:
        assert not {groups[train[int(i)]["image_path"]] for i in fit} & {
            groups[train[int(i)]["image_path"]] for i in validation
        }


def test_verified_source_groups_control_holdout_and_cv(inventory):
    audit, root = inventory
    path = root / "verified_groups.csv"
    rows = [{
        "image_path": row["image_path"], "group_id": f"fixture_group_{i // 2}",
        "evidence": "Synthetic test-only acquisition record",
    } for i, row in enumerate(audit.rows)]
    write_csv(path, rows, ["image_path", "group_id", "evidence"])
    groups, _ = independence_groups(audit, path)
    train, test, strategy = make_partitions(audit, groups)
    folds, cv_strategy = generate_cv_indices(train, test, audit, groups)
    assert strategy.startswith("group_shuffle")
    assert cv_strategy == "StratifiedGroupKFold"
    assert len(folds) == 5


def test_similarity_flags_do_not_invent_groups(inventory):
    audit, _ = inventory
    audit.near_pairs = [{
        "image_path_a": audit.rows[0]["image_path"],
        "image_path_b": audit.rows[1]["image_path"],
    }]
    groups, details = independence_groups(audit)
    assert len(set(groups.values())) == len(audit.rows)
    assert {row["basis"] for row in details} == {"singleton"}


def test_grouping_unions_transitive_exact_relations(inventory):
    audit, _ = inventory
    audit.rows[1]["sha256"] = audit.rows[0]["sha256"]
    audit.rows[2]["pixel_sha256"] = audit.rows[1]["pixel_sha256"]
    groups, _ = independence_groups(audit)
    assert len({groups[row["image_path"]] for row in audit.rows[:3]}) == 1


@pytest.mark.parametrize("kind", ["missing_evidence", "unknown_path", "duplicate_path"])
def test_invalid_verified_groups_are_rejected(inventory, kind):
    audit, root = inventory
    path = root / "groups.csv"
    row = {"image_path": audit.rows[0]["image_path"], "group_id": "group", "evidence": "record"}
    if kind == "missing_evidence":
        row["evidence"] = ""
    if kind == "unknown_path":
        row["image_path"] = "data/raw/unknown.jpeg"
    write_csv(path, [row, row] if kind == "duplicate_path" else [row], [
        "image_path", "group_id", "evidence",
    ])
    with pytest.raises(ValueError):
        independence_groups(audit, path)


def test_content_overlap_guard_does_not_trust_supplied_groups(inventory):
    audit, _ = inventory
    groups, _ = independence_groups(audit)
    train, test, _ = make_partitions(audit, groups)
    by_path = {row["image_path"]: row for row in audit.rows}
    by_path[test[0]["image_path"]]["sha256"] = by_path[train[0]["image_path"]]["sha256"]
    with pytest.raises(ValueError, match="Exact content crosses"):
        validate_partitions(train, test, audit, groups)


def test_cv_duplicate_guard_catches_incorrect_singleton_groups(inventory):
    audit, _ = inventory
    groups, _ = independence_groups(audit)
    train, test, _ = make_partitions(audit, groups)
    by_path = {row["image_path"]: row for row in audit.rows}
    for row in train[:12]:
        by_path[row["image_path"]]["pixel_sha256"] = "test_only_shared_pixel_hash"
    with pytest.raises(ValueError, match="Duplicate content crosses a CV"):
        generate_cv_indices(train, test, audit, groups)


def test_manifest_membership_and_label_tampering(inventory):
    audit, root = inventory
    groups, _ = independence_groups(audit)
    train, test, _ = make_partitions(audit, groups)
    path = root / "train.csv"
    write_csv(path, train, MANIFEST_FIELDS)
    assert read_manifest(path, "train", root) == train
    invalid = deepcopy(train)
    invalid[0]["label"] = 1 - invalid[0]["label"]
    write_csv(path, invalid, MANIFEST_FIELDS)
    with pytest.raises(ValueError, match="label"):
        read_manifest(path, "train", root)
    with pytest.raises(ValueError):
        generate_cv_indices(test, train, audit, groups)


@pytest.mark.parametrize("bad_path", [
    "../outside.jpeg", "C:/outside.jpeg", "data\\raw\\image.jpeg",
    "data/raw/casting_data/casting_data/train/ok_front/image.jpeg",
])
def test_invalid_manifest_paths(inventory, bad_path):
    audit, root = inventory
    groups, _ = independence_groups(audit)
    train, _, _ = make_partitions(audit, groups)
    train[0]["image_path"] = bad_path
    train[0]["image_id"] = image_identifier(bad_path)
    path = root / "train.csv"
    write_csv(path, train, MANIFEST_FIELDS)
    with pytest.raises(ValueError):
        read_manifest(path, "train", root)


def test_failed_audit_is_not_silently_filtered(inventory):
    audit, _ = inventory
    audit.summary["blocking_issues"] = ["Corrupt test fixture"]
    groups, _ = independence_groups(audit)
    with pytest.raises(ValueError, match="no images will be discarded"):
        make_partitions(audit, groups)


def test_full_preparation_and_saved_manifest_validation(tmp_path, monkeypatch):
    counts = {"def_front": 25, "ok_front": 15}
    make_images(tmp_path, counts)
    real_audit = preparation.audit_dataset

    def fixture_audit(directory, root, threshold):
        return real_audit(directory, root, threshold, expected_counts=counts)

    monkeypatch.setattr(preparation, "audit_dataset", fixture_audit)
    summary = preparation.prepare_dataset(tmp_path)
    assert summary["train"]["total"] == 32
    assert summary["test"]["total"] == 8
    assert summary["split_status"] == "provisional"
    assert not summary["fully_leakage_safe"]
    manifests = tmp_path / "data/splits"
    before = {path.name: path.read_bytes() for path in manifests.iterdir()}
    validated = preparation.prepare_dataset(tmp_path, validate_only=True)
    assert validated == summary
    assert before == {path.name: path.read_bytes() for path in manifests.iterdir()}
    (manifests / "train.csv").write_bytes((manifests / "train.csv").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differ from regeneration"):
        preparation.prepare_dataset(tmp_path, validate_only=True)
