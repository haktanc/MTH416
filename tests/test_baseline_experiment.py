"""Check saved-fold baselines on small fixtures without project test data."""

import json

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.baseline_experiment import (
    METRICS, align_rows, estimated_peak_mib, evaluate_fold, fold_matrix, saved_folds, smoke_subset,
    summarize, svm_predictions,
)
from src.evaluation import evaluate_predictions
from src.model_training import build_baseline_classifier
from src.reporting import write_csv, write_json


def training_rows():
    return [
        {"image_id": f"image-{i}", "image_path": f"{i}.jpeg", "label": i % 2,
         "class_name": "ok_front" if i % 2 == 0 else "def_front",
         "feature_row": i, "split": "train", "validation_fold": 5 - i // 2,
         "group_id": f"group-{i}"}
        for i in range(10)
    ]


def test_identifier_join_reorders_features_and_rejects_misaligned_labels():
    rows = training_rows()
    candidate = [{**row, "feature_row": i} for i, row in enumerate(reversed(rows))]
    labels = [row["label"] for row in candidate]
    np.testing.assert_array_equal(align_rows(rows, candidate, labels), np.arange(9, -1, -1))
    for change in (
        {**candidate[0], "image_id": candidate[1]["image_id"]},
        {**candidate[0], "image_path": "unrelated.jpeg"},
        {**candidate[0], "split": "test"},
    ):
        with pytest.raises(ValueError):
            align_rows(rows, [change, *candidate[1:]], labels)
    with pytest.raises(ValueError, match="Cached label"):
        align_rows(rows, candidate, labels[::-1])


def test_saved_folds_keep_exact_membership_and_reject_contamination():
    rows = training_rows()
    saved = [{**row} for row in reversed(rows)]
    folds = saved_folds(rows, saved)
    for number, fit, valid in folds:
        np.testing.assert_array_equal(valid, [2 * (5 - number), 2 * (5 - number) + 1])
        assert len(fit) == 8 and not set(fit) & set(valid)
    saved[0]["split"] = "test"
    with pytest.raises(ValueError, match="non-training"):
        saved_folds(rows, saved)
    saved[0]["split"] = "train"
    saved[0]["validation_fold"] = 2
    with pytest.raises(ValueError, match="disagree"):
        saved_folds(rows, saved)


def test_combined_design_uses_identifier_order_and_only_training_paths(tmp_path):
    cache = tmp_path / "results/cache"
    cache.mkdir(parents=True)
    np.save(cache / "train_hog.npy", [[1, 2], [3, 4], [5, 6]])
    np.save(cache / "train_glcm.npy", [[61], [21], [41]])
    components = [
        {"path": "results/cache/train_hog.npy", "shape": [3, 2], "order": [0, 1, 2]},
        {"path": "results/cache/train_glcm.npy", "shape": [3, 1], "order": [1, 2, 0]},
    ]
    result = fold_matrix(tmp_path, components, np.array([2, 0]), np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[5, 6, 61], [1, 2, 21]])
    components[0]["path"] = "results/cache/test_hog.npy"
    with pytest.raises(ValueError, match="training caches"):
        fold_matrix(tmp_path, components, np.array([0]), np.float32)


def test_fixed_models_and_fold_local_standardization():
    svm = build_baseline_classifier("svm")
    scaler, classifier = svm.named_steps.values()
    assert isinstance(scaler, StandardScaler)
    params = classifier.get_params()
    assert {key: params[key] for key in ("kernel", "C", "gamma", "class_weight", "random_state")} == {
        "kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None, "random_state": 42,
    }
    train = np.arange(222, dtype=float).reshape(74, 3)
    train[:, 1] = 3
    expected = StandardScaler().fit(train)
    svm.fit(train.copy(), np.arange(len(train)) % 2)
    np.testing.assert_allclose(scaler.mean_, expected.mean_)
    np.testing.assert_allclose(scaler.var_, expected.var_)
    np.testing.assert_allclose(scaler.scale_, expected.scale_)
    svm.predict(np.full((4, 3), 90000.0))
    np.testing.assert_allclose(scaler.mean_, expected.mean_)
    rf = build_baseline_classifier("random_forest")
    assert list(rf.named_steps) == ["classifier"]
    params = rf.named_steps["classifier"].get_params()
    expected = dict(n_estimators=100, max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, class_weight=None, n_jobs=1, random_state=42)
    assert {key: params[key] for key in expected} == expected


def test_fold_prediction_and_scores_use_only_one_scaling_pass(tmp_path):
    cache = tmp_path / "results/cache"
    cache.mkdir(parents=True)
    values = np.array([[-9], [-8], [-7], [-6], [3], [4], [5], [6], [-3], [2]], dtype=np.float32)
    np.save(cache / "train_feature.npy", values)
    labels = np.array([0] * 4 + [1] * 4 + [0, 1])
    job = {"model": "svm", "fit": list(range(8)), "valid": [8, 9], "labels": labels.tolist(),
           "components": [{"path": "results/cache/train_feature.npy", "shape": [10, 1],
                           "order": list(range(10))}]}
    recorded = evaluate_fold(tmp_path, job)
    reference = build_baseline_classifier("svm")
    reference.fit(values[:8].astype(float), labels[:8])
    preds = reference.predict(values[8:].astype(float))
    scores = reference.decision_function(values[8:].astype(float))
    metrics, _ = evaluate_predictions(labels[8:], preds, scores)
    assert {key: recorded[key] for key in metrics} == metrics
    with pytest.raises(ValueError, match="overlap"):
        evaluate_fold(tmp_path, {**job, "valid": [7, 8]})


def test_batched_rbf_validation_agrees_with_svc_on_all_fixture_rows():
    rng = np.random.default_rng(42)
    train, valid = rng.normal(size=(60, 70)), rng.normal(size=(33, 70))
    model = build_baseline_classifier("svm").fit(train, np.arange(len(train)) % 2)
    classifier = model.named_steps["classifier"]
    scaled = model.named_steps["scaler"].transform(valid)
    labels, scores = svm_predictions(classifier, scaled)
    np.testing.assert_array_equal(labels, classifier.predict(scaled))
    np.testing.assert_allclose(scores, classifier.decision_function(scaled), rtol=1e-9, atol=1e-10)


def test_metrics_false_negatives_and_result_serialization(tmp_path):
    metrics, confusion = evaluate_predictions([0, 0, 1, 1, 1], [0, 0, 0, 1, 1],
                                               [0.1, 0.2, 0.3, 0.8, 0.9])
    assert metrics["accuracy"] == 0.8
    assert metrics["balanced_accuracy"] == pytest.approx((1 + 2 / 3) / 2)
    assert metrics["precision"] == 1
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == 0.8 and metrics["roc_auc"] == 1
    np.testing.assert_array_equal(confusion, [[2, 0], [1, 2]])
    rows = [{**dict.fromkeys(METRICS, 0.5), "feature_set": "glcm", "model": "svm",
             "fold": fold, "f1": fold / 10, "true_negatives": 2, "false_positives": 0,
             "true_positives": 2, "false_negatives": 1, "minimum_available_mib": 500}
            for fold in range(1, 6)]
    result, = summarize(rows, [("glcm", "svm"), ("hog8", "random_forest")])
    assert result["f1_mean"] == pytest.approx(0.3)
    assert result["f1_sd"] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1))
    assert result["false_negatives_total"] == 5
    write_csv(tmp_path / "baseline.csv", [result], list(result))
    write_json(tmp_path / "run.json", {"completed_combinations": 1})
    assert (tmp_path / "baseline.csv").read_text().splitlines()[1].startswith("glcm,svm,5,")
    assert json.loads((tmp_path / "run.json").read_text())["completed_combinations"] == 1


def test_smoke_sampling_is_reproducible_and_keeps_existing_partition():
    labels = np.arange(100) % 2
    eligible = np.arange(60, 100)
    selected = smoke_subset(eligible, labels, per_class=8)
    np.testing.assert_array_equal(selected, smoke_subset(eligible, labels, per_class=8))
    assert set(selected) <= set(eligible)
    np.testing.assert_array_equal(np.bincount(labels[selected]), [8, 8])


def test_high_dimensional_preflight_accounts_for_resident_cache_pages():
    # Stage 4A recorded about 1,043 MiB for HOG 8x8 Random Forest workers.
    assert estimated_peak_mib("random_forest", 832, 208, 142884) > 1043
