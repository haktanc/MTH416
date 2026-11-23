"""Check Stage 4B isolation, search space, weighting, and persisted results."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

import src.tuning_experiment as tuning
from src.baseline_experiment import METRICS
from src.baseline_scaling import BatchedStandardScaler, weight_glcm


def test_fixed_candidate_plan_and_controlled_weight_pairs():
    plan = tuning.search_plan()
    assert plan == tuning.search_plan()
    assert len({r["candidate"] for r in plan}) == len(plan) == 38
    counts = Counter((r["feature_set"], r["model"]) for r in plan)
    assert counts["hog16_glcm", "svm"] == 8
    assert sorted(counts.values()) == [6, 6, 6, 6, 6, 8]
    baseline = [r for r in plan if r["baseline"]]
    assert len(baseline) == 6 and all(r["glcm_weight"] == 1 for r in baseline)
    for weighted in [r for r in plan if r["glcm_weight"] > 1]:
        assert weighted["glcm_weight"] ** 2 * 48 == pytest.approx(34596)
        assert any(r["feature_set"] == weighted["feature_set"] and r["model"] == weighted["model"]
                   and r["parameters"] == weighted["parameters"] and r["glcm_weight"] == 1 for r in plan)


def test_fixed_weights_apply_after_fold_local_scaling():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(64, 60))
    validation = rng.normal(loc=100, size=(6, 60))
    scaler = BatchedStandardScaler().fit(train.copy())
    original = scaler.transform(validation)
    weighted = weight_glcm(original.copy(), 8)
    np.testing.assert_array_equal(weighted[:, :-48], original[:, :-48])
    np.testing.assert_array_equal(weighted[:, -48:], original[:, -48:] * 8)
    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))


def test_training_loader_preserves_saved_folds_and_never_opens_holdout(tmp_path, monkeypatch):
    tables = tmp_path / "results/tables"
    caches = tmp_path / "results/cache/features"
    tables.mkdir(parents=True)
    caches.mkdir(parents=True)
    rows = [
        {"image_id": f"img{i}", "image_path": f"train/{i}.jpg",
         "feature_row": i, "split": "train", "class_name": "def_front" if i >= 415 else "ok_front",
         "label": int(i >= 415), "validation_fold": i % 5 + 1, "group_id": f"g{i}"}
        for i in range(1040)
    ]
    for method in ("hog16", "glcm"):
        np.save(caches / f"train_{method}.npy", np.zeros((1040, 4), dtype=np.float32))
    np.save(caches / "train_labels.npy", np.array([r["label"] for r in rows]))
    (tables / "feature_run_summary.json").write_text(json.dumps({
        "provenance": {"configuration": {"image_shape": [512, 512]},
                       "source_code_sha256": {}, "versions": {}},
        "caches": [{"split": split, "method": method, "labels_sha256": "fixture"} for split in (
            "train", "test") for method in ("hog16", "glcm")],
    }))
    (tables / "baseline_cv_run.json").write_text(json.dumps({
        "completed_combinations": 10, "scope": "training-only cross-validation",
        "training_rows": 1040, "versions": {}, "provenance": {
            **dict.fromkeys(("train_manifest_sha256", "cv_folds_sha256",
                             "train_index_sha256", "stage3_summary_sha256"), "fixture"),
            "training_caches": {
                m: {"path": f"results/cache/features/train_{m}.npy", "shape": [1040, 4],
                    "dtype": "float32", "sha256": "fixture"} for m in ("hog16", "glcm")
            },
        },
    }))

    def reject_holdout(path):
        path = Path(path)
        assert path.name != "test.csv" and not path.name.startswith(("test_", "feature_test_"))
        return path

    monkeypatch.setattr(tuning, "file_sha256", lambda p: "fixture" if reject_holdout(p) else None)
    monkeypatch.setattr(tuning, "read_csv", lambda p: rows if reject_holdout(p) else [])

    def manifest(p, split, root):
        assert reject_holdout(p).name == "train.csv" and split == "train"
        return rows

    load = np.load
    monkeypatch.setattr(tuning, "read_manifest", manifest)
    monkeypatch.setattr(tuning.np, "load", lambda path, **kw: load(reject_holdout(path), **kw))
    layouts, folds, reference, _ = tuning.load_training(tmp_path)
    assert set(layouts) == {"hog16", "glcm"} and reference == rows
    for existing, _, validation in folds:
        assert len(validation) == 208 and all(rows[i]["validation_fold"] == existing for i in validation)


def test_checkpoint_summary_and_selection_are_round_trip_safe(tmp_path):
    plan = tuning.search_plan()
    folds = {}
    for candidate in plan:
        name = candidate["candidate"]
        folder = tmp_path / name
        folder.mkdir()
        score = 0.8 if candidate["baseline"] else 0.79
        for number in range(1, 6):
            tuning.checkpoint(folder / f"{number}.json", {
                **dict.fromkeys(METRICS, score), "f1": score, "fold": number,
                "false_negatives": 8, "false_positives": 9,
                "true_positives": 117, "true_negatives": 74,
            })
        folds[name] = tuning.read_completed(tmp_path, candidate)
    records, summaries = tuning.report_rows(plan, folds)
    assert len(records) == 190 and len(summaries) == 38
    assert {r["source"] for r in tuning.select_candidates(summaries)} == {"stage4a"}
    assert all(r["false_negatives_total"] == 40 for r in summaries)
