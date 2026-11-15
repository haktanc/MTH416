"""Prepare reproducible manifests and training-only folds without model fitting."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from src.data_loader import (
    MANIFEST_FIELDS,
    PROJECT_ROOT,
    read_manifest,
    resolve_manifest_path,
)
from src.dataset_audit import (
    PAIR_FIELDS,
    AuditResult,
    assert_raw_unchanged,
    audit_dataset,
    write_audit_reports,
    write_pair_review,
)
from src.reporting import csv_bytes, output_directory, write_csv, write_json


SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
CV_FIELDS = [
    "image_path", "image_id", "label", "class_name", "validation_fold", "group_id",
]


def independence_groups(
    audit: AuditResult, verified_groups: Path | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Union exact contents and explicitly evidenced source groups.

    Generated group IDs are bookkeeping identifiers, not physical-part IDs.
    Perceptual candidates and filename coincidences never create groups.
    """
    paths = sorted(row["image_path"] for row in audit.rows)
    parent = {path: path for path in paths}

    def find(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    def union(first, second):
        a, b = sorted((find(first), find(second)))
        parent[b] = a

    for key in ("sha256", "pixel_sha256"):
        seen = {}
        for row in audit.rows:
            value = row[key]
            if value:
                if value in seen:
                    union(row["image_path"], seen[value])
                else:
                    seen[value] = row["image_path"]
    evidence = {}
    if verified_groups is not None:
        with Path(verified_groups).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image_path", "group_id", "evidence"]:
                raise ValueError("Verified groups require image_path,group_id,evidence.")
            source_rows = list(reader)
        if not source_rows:
            raise ValueError("Verified groups file is empty.")
        first_for_group = {}
        for row in source_rows:
            path = row["image_path"]
            if path not in parent or path in evidence:
                raise ValueError("Unknown or repeated image in verified groups.")
            if not row["group_id"].strip() or not row["evidence"].strip():
                raise ValueError("Verified source groups require nonempty evidence.")
            evidence[path] = row["evidence"]
            group = row["group_id"]
            if group in first_for_group:
                union(path, first_for_group[group])
            else:
                first_for_group[group] = path
    members = defaultdict(list)
    for path in paths:
        members[find(path)].append(path)
    result, details = {}, []
    for representative, group_paths in sorted(members.items()):
        group_id = "g_" + hashlib.sha256(
            representative.encode("utf-8")
        ).hexdigest()
        reason = (
            "verified_source_and_or_exact_content"
            if any(path in evidence for path in group_paths)
            else "exact_content" if len(group_paths) > 1 else "singleton"
        )
        for path in group_paths:
            result[path] = group_id
            details.append({
                "image_path": path, "group_id": group_id,
                "group_size": len(group_paths), "basis": reason,
                "evidence": evidence.get(path, ""),
            })
    return result, sorted(details, key=lambda row: row["image_path"])


def _manifest(rows: list[dict], positions, split: str) -> list[dict]:
    return sorted([
        {**{key: rows[int(index)][key] for key in MANIFEST_FIELDS if key != "split"},
         "split": split}
        for index in positions
    ], key=lambda row: row["image_path"])


def make_partitions(
    audit: AuditResult, groups: dict[str, str],
    seed: int = SEED, test_size: float = TEST_SIZE,
) -> tuple[list[dict], list[dict], str]:
    """Stratify independent images, or keep verified groups intact."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if audit.summary["blocking_issues"]:
        raise ValueError("Audit contains blocking issues; no images will be discarded.")
    if not 0 < test_size < 1:
        raise ValueError("Test fraction must be between zero and one.")
    rows = sorted(audit.rows, key=lambda row: row["image_path"])
    labels = np.array([row["label"] for row in rows])
    group_ids = np.array([groups[row["image_path"]] for row in rows])
    positions = np.arange(len(rows))
    if len(set(group_ids)) == len(rows):
        train, test = train_test_split(
            positions, test_size=test_size, random_state=seed, stratify=labels
        )
        strategy = "stratified_image_split"
    else:
        # Search only membership/labels, never model performance. Size is the
        # first priority after group isolation; class balance breaks ties.
        splitter = GroupShuffleSplit(
            n_splits=1000, test_size=test_size, random_state=seed
        )
        candidates = []
        target_size = int(np.ceil(len(rows) * test_size))
        for train_index, test_index in splitter.split(positions, labels, group_ids):
            if set(labels[train_index]) != {0, 1} or set(labels[test_index]) != {0, 1}:
                continue
            error = sum(
                abs(np.sum(labels[test_index] == label)
                    - np.sum(labels == label) * test_size)
                for label in (0, 1)
            )
            score = (abs(len(test_index) - target_size), float(error))
            candidates.append((score, train_index, test_index))
        if not candidates:
            raise ValueError("No group-independent split containing both classes.")
        _, train, test = min(candidates, key=lambda item: item[0])
        strategy = "group_shuffle_search_1000_size_then_class_balance"
    train_rows, test_rows = _manifest(rows, train, "train"), _manifest(rows, test, "test")
    validate_partitions(train_rows, test_rows, audit, groups)
    return train_rows, test_rows, strategy


def validate_partitions(
    train: list[dict], test: list[dict], audit: AuditResult,
    groups: dict[str, str], project_root: Path | None = None,
) -> None:
    """Require complete primary coverage and isolate all identifiable groups."""
    if audit.summary["blocking_issues"]:
        raise ValueError("Cannot validate a split against a failed audit.")
    known = {row["image_path"]: row for row in audit.rows}
    if not train or not test:
        raise ValueError("Both training and test partitions must be nonempty.")
    for subset, name in ((train, "train"), (test, "test")):
        if subset != sorted(subset, key=lambda row: row["image_path"]):
            raise ValueError("Partition rows must be sorted by image path.")
        ids, paths = set(), set()
        for row in subset:
            path = row["image_path"]
            if path not in known or row["split"] != name:
                raise ValueError("Unknown image or incorrect split membership.")
            if any(row[key] != known[path][key] for key in (
                "image_id", "class_name", "label",
            )):
                raise ValueError("Manifest identity or label differs from the audit.")
            if row["image_id"] in ids or path in paths:
                raise ValueError("Repeated image identity or path in a partition.")
            ids.add(row["image_id"])
            paths.add(path)
            if project_root is not None:
                resolve_manifest_path(path, project_root)
        if {row["label"] for row in subset} != {0, 1}:
            raise ValueError("Each partition must contain both classes.")
    train_paths = {row["image_path"] for row in train}
    test_paths = {row["image_path"] for row in test}
    if train_paths & test_paths:
        raise ValueError("Training and test image identities overlap.")
    if {row["image_id"] for row in train} & {row["image_id"] for row in test}:
        raise ValueError("Training and test image identifiers overlap.")
    if train_paths | test_paths != set(known):
        raise ValueError("Manifests do not cover the complete audited dataset.")
    for key in ("sha256", "pixel_sha256"):
        a = {known[path][key] for path in train_paths}
        b = {known[path][key] for path in test_paths}
        if a & b:
            raise ValueError(f"Exact content crosses the holdout split: {key}")
    if {groups[path] for path in train_paths} & {groups[path] for path in test_paths}:
        raise ValueError("Verified independence group crosses the holdout split.")


def generate_cv_indices(
    train: list[dict], test: list[dict], audit: AuditResult,
    groups: dict[str, str], n_splits: int = N_FOLDS, seed: int = SEED,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Return positional indices into train.csv only, never the test manifest."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    validate_partitions(train, test, audit, groups)
    labels = np.array([row["label"] for row in train])
    group_ids = np.array([groups[row["image_path"]] for row in train])
    if n_splits < 2 or min(Counter(labels).values()) < n_splits:
        raise ValueError("Each training class needs at least n_splits samples.")
    grouped = len(set(group_ids)) < len(train)
    if grouped:
        if any(len(set(group_ids[labels == label])) < n_splits for label in (0, 1)):
            raise ValueError("Too few independent training groups for stratified CV.")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        iterator = splitter.split(np.zeros((len(train), 1)), labels, group_ids)
        strategy = "StratifiedGroupKFold"
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        iterator = splitter.split(np.zeros((len(train), 1)), labels)
        strategy = "StratifiedKFold"
    folds = list(iterator)
    known = {row["image_path"]: row for row in audit.rows}
    appearances = np.zeros(len(train), dtype=int)
    for fit, validation in folds:
        if set(fit) & set(validation) or set(fit) | set(validation) != set(range(len(train))):
            raise ValueError("CV training/validation indices overlap or omit samples.")
        if set(labels[fit]) != {0, 1} or set(labels[validation]) != {0, 1}:
            raise ValueError("Every CV partition must contain both classes.")
        if set(group_ids[fit]) & set(group_ids[validation]):
            raise ValueError("Verified group crosses a CV fold boundary.")
        for key in ("sha256", "pixel_sha256"):
            a = {known[train[int(i)]["image_path"]][key] for i in fit}
            b = {known[train[int(i)]["image_path"]][key] for i in validation}
            if a & b:
                raise ValueError("Duplicate content crosses a CV fold boundary.")
        appearances[validation] += 1
    if not np.all(appearances == 1):
        raise ValueError("Every training sample must validate exactly once.")
    return folds, strategy


def cv_assignments(train, folds, groups):
    """Persist validation membership; fit membership is its train complement."""
    result = []
    for fold_number, (_, validation) in enumerate(folds, 1):
        for index in validation:
            row = train[int(index)]
            result.append({
                **{key: row[key] for key in MANIFEST_FIELDS if key != "split"},
                "validation_fold": fold_number,
                "group_id": groups[row["image_path"]],
            })
    return sorted(result, key=lambda row: row["image_path"])


def _counts(rows):
    counts = Counter(row["class_name"] for row in rows)
    return {
        "total": len(rows), "class_counts": dict(sorted(counts.items())),
        "class_proportions": {
            name: count / len(rows) for name, count in sorted(counts.items())
        },
    }


def prepare_dataset(
    project_root: Path = PROJECT_ROOT,
    seed: int = SEED, near_threshold: int = 4,
    verified_groups: Path | None = None, validate_only: bool = False,
) -> dict:
    """Audit, prepare/validate manifests, and document identifiable leakage."""
    root = Path(project_root).resolve()
    tables = output_directory(root / "results/tables", root)
    splits = output_directory(root / "data/splits", root)
    audit = audit_dataset(root / "data/raw/casting_512x512", root, near_threshold)
    write_audit_reports(audit, tables, root)
    groups, group_details = independence_groups(audit, verified_groups)
    train, test, strategy = make_partitions(audit, groups, seed)
    second_train, second_test, _ = make_partitions(audit, groups, seed)
    if train != second_train or test != second_test:
        raise ValueError("Split regeneration is not deterministic.")
    folds, cv_strategy = generate_cv_indices(train, test, audit, groups, seed=seed)
    assignments = cv_assignments(train, folds, groups)
    repeat_folds, _ = generate_cv_indices(train, test, audit, groups, seed=seed)
    if assignments != cv_assignments(train, repeat_folds, groups):
        raise ValueError("CV regeneration is not deterministic.")
    assert_raw_unchanged(audit, root)
    if validate_only:
        loaded_train = read_manifest(splits / "train.csv", "train", root)
        loaded_test = read_manifest(splits / "test.csv", "test", root)
        validate_partitions(loaded_train, loaded_test, audit, groups, root)
        for path, rows, fields in (
            (splits / "train.csv", train, MANIFEST_FIELDS),
            (splits / "test.csv", test, MANIFEST_FIELDS),
            (tables / "cv_folds.csv", assignments, CV_FIELDS),
        ):
            if path.read_bytes() != csv_bytes(rows, fields):
                raise ValueError(f"Saved manifest/folds differ from regeneration: {path}")
    else:
        write_csv(splits / "train.csv", train, MANIFEST_FIELDS)
        write_csv(splits / "test.csv", test, MANIFEST_FIELDS)
        write_csv(tables / "cv_folds.csv", assignments, CV_FIELDS)
        validate_partitions(
            read_manifest(splits / "train.csv", "train", root),
            read_manifest(splits / "test.csv", "test", root), audit, groups, root,
        )
    write_csv(tables / "independence_groups.csv", group_details, [
        "image_path", "group_id", "group_size", "basis", "evidence",
    ])
    partition = {row["image_path"]: row["split"] for row in train + test}
    fold_for_path = {row["image_path"]: row["validation_fold"] for row in assignments}
    annotated = []
    for pair in audit.near_pairs:
        a, b = pair["image_path_a"], pair["image_path_b"]
        same_group = groups[a] == groups[b]
        annotated.append({
            **pair, "split_a": partition[a], "split_b": partition[b],
            "validation_fold_a": fold_for_path.get(a, ""),
            "validation_fold_b": fold_for_path.get(b, ""),
            "crosses_holdout": partition[a] != partition[b],
            "crosses_cv": a in fold_for_path and b in fold_for_path
            and fold_for_path[a] != fold_for_path[b],
            "same_verified_group": same_group,
            "review_status": "verified_group_isolated" if same_group else "unresolved",
        })
    write_csv(tables / "near_duplicate_partition_review.csv", annotated, PAIR_FIELDS + [
        "split_a", "split_b", "validation_fold_a", "validation_fold_b",
        "crosses_holdout", "crosses_cv", "same_verified_group",
    ])
    write_pair_review(
        annotated, tables / "near_duplicate_partition_review.html", root
    )
    summary = {
        "validation_passed": True, "split_status": "provisional",
        "fully_leakage_safe": False,
        "seed": seed, "target_test_fraction": TEST_SIZE,
        "holdout_strategy": strategy, "train": _counts(train), "test": _counts(test),
        "source_inventory_sha256": audit.summary["inventory_sha256"],
        "all_primary_raw_files_unchanged": True,
        "no_images_discarded": len(train) + len(test) == len(audit.rows),
        "near_threshold": near_threshold,
        "near_candidate_pairs": len(annotated),
        "unresolved_near_pairs": sum(p["review_status"] == "unresolved" for p in annotated),
        "near_pairs_crossing_holdout": sum(p["crosses_holdout"] for p in annotated),
        "near_pairs_crossing_cv": sum(p["crosses_cv"] for p in annotated),
        "verified_group_file_sha256": (
            hashlib.sha256(Path(verified_groups).read_bytes()).hexdigest()
            if verified_groups is not None else None
        ),
        "non_singleton_independence_groups": len({
            row["group_id"] for row in group_details if row["group_size"] > 1
        }),
        "cv": {
            "strategy": cv_strategy, "n_splits": N_FOLDS,
            "shuffle": True, "random_state": seed,
            "index_reference": "Zero-based positions in sorted train.csv rows only",
            "validation_appearances_per_training_image": 1,
            "test_images_in_folds": 0,
            "folds": [{
                "fold": number,
                "fit": _counts([train[int(index)] for index in fit]),
                "validation": _counts([train[int(index)] for index in validation]),
            } for number, (fit, validation) in enumerate(folds, 1)],
        },
        "checks": {
            "complete_primary_coverage": True, "valid_manifest_paths": True,
            "disjoint_image_identities": True, "no_exact_content_crosses_holdout": True,
            "verified_groups_isolated": True, "no_exact_content_crosses_cv": True,
            "training_only_cv": True, "split_regeneration_identical": True,
            "cv_regeneration_identical": True,
        },
        "manifest_sha256": {
            name: hashlib.sha256((splits / name).read_bytes()).hexdigest()
            for name in ("train.csv", "test.csv")
        },
        "versions": {name: version(name) for name in (
            "numpy", "opencv-python", "Pillow", "scikit-learn", "scipy",
        )},
        "warnings": [
            "Unknown physical-part relationships are not ruled out; the split is provisional.",
            "Similarity candidates are retained and not automatically grouped or discarded.",
            "Shared filename counters are not verified physical-part identifiers.",
            "Resolve cross-boundary candidates before interpreting final test performance.",
            "Fit future scaling and feature selection inside training/CV pipelines only.",
            "Do not select models, features, thresholds, or hyperparameters using test outcomes.",
        ],
    }
    write_json(tables / "split_config.json", {
        key: summary[key] for key in (
            "seed", "target_test_fraction", "holdout_strategy", "near_threshold",
            "source_inventory_sha256", "verified_group_file_sha256", "versions",
        )
    } | {"cv_strategy": cv_strategy, "n_splits": N_FOLDS, "shuffle": True})
    write_json(tables / "stage1_validation.json", summary)
    report = [
        "# Stage 1 Split Validation", "",
        "**Status: provisional. Identifiable exact-content and verified-group "
        "leakage checks pass; complete physical-part independence is not established.**", "",
        f"Training: {summary['train']}", f"Testing: {summary['test']}",
        f"Holdout method: `{strategy}`; target 80/20; seed {seed}.",
        f"Five-fold method: `{cv_strategy}`; shuffle=True; random_state={seed}.",
        "For each fold, indices refer only to train.csv rows. Validation membership "
        "is recorded in cv_folds.csv; the fit set is the remaining training rows. "
        "Every training image validates exactly once; no held-out test image is used.", "",
        f"Suspicious pairs: {len(annotated)}; crossing holdout: "
        f"{summary['near_pairs_crossing_holdout']}; crossing training CV folds: "
        f"{summary['near_pairs_crossing_cv']}.",
        "These counts flag unresolved similarities, not confirmed leakage. See "
        "near_duplicate_partition_review.csv and near_duplicate_review.html.", "",
        "Groups are connected components of exact byte/pixel matches plus any "
        "explicitly supplied verified source relationships. Generated group IDs "
        "are bookkeeping only. Group isolation takes priority over exact sizes "
        "or class proportions. When groups exist, 1,000 seeded group-shuffle "
        "candidates are ranked first by test-size error, then class-count error; "
        "no model outcomes are considered. Realized counts/proportions appear "
        "in stage1_validation.json. No image is dropped to improve balance.", "",
        "## Validation", "",
        *[f"- {key}: {value}" for key, value in summary["checks"].items()],
        "- Primary file membership and SHA-256 contents unchanged: True.", "",
        "## Limitations and required review", "",
        *[f"- {warning}" for warning in summary["warnings"]], "",
        "Obtain acquisition records or supplier confirmation for source/part "
        "relationships. Record substantiated group assignments with evidence, "
        "then regenerate both holdout and CV assignments. Do not infer shared "
        "parts solely from similar impeller structure, hash distance, or numeric "
        "filenames. This stage does not verify visual class labels.", "",
        "## Reproduction", "",
        "Run `python -m src.prepare_splits` to regenerate and "
        "`python -m src.prepare_splits --validate-only` to re-audit and verify "
        "saved manifest/fold bytes against deterministic regeneration.", "",
        "References: [StratifiedKFold](https://scikit-learn.org/stable/modules/"
        "generated/sklearn.model_selection.StratifiedKFold.html), "
        "[StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/"
        "sklearn.model_selection.StratifiedGroupKFold.html).", "",
    ]
    (tables / "split_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--verified-groups", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = prepare_dataset(
            seed=args.seed, near_threshold=args.near_threshold,
            verified_groups=args.verified_groups, validate_only=args.validate_only,
        )
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError) as error:
        write_json(PROJECT_ROOT / "results/tables/stage1_validation.json", {
            "validation_passed": False, "error": str(error),
        })
        parser.exit(1, f"Stage 1 preparation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
