"""Validate the evaluation and plots on synthetic data before holdout access."""

import csv
from math import sqrt

import numpy as np
import pytest

from src import holdout_experiment as holdout
from src.feature_pipeline import file_sha256
from src.holdout_figures import plot_holdout


def test_combined_svm_applies_fixed_weight_after_train_scaling():
    train = np.arange(8 * 50, dtype=np.float64).reshape(8, 50)
    test = train[[0, 5]].copy() + 200
    labels = [0] * 4 + [1] * 4
    weight = sqrt(34596 / 48)
    pipeline = holdout.construct_pipeline("svm", "hog16_glcm", {"C": 3, "gamma": "scale"}, weight)
    assert [step["step"] for step in holdout.describe_pipeline(pipeline)] == [
        "scaler", "glcm_weight", "classifier",
    ]
    pipeline.fit(train.copy(), labels)
    z = (test - train.mean(axis=0)) / train.std(axis=0)
    z[:, -48:] *= weight
    np.testing.assert_allclose(pipeline[:-1].transform(test.copy()), z)
    np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, train.mean(axis=0))
    labels, scores, _, _ = holdout.infer(pipeline, test.copy(), "svm")
    np.testing.assert_array_equal(labels, pipeline.predict(test.copy()))
    np.testing.assert_allclose(scores, pipeline.decision_function(test.copy()))
    np.testing.assert_array_equal(labels, scores > 0)


def test_forest_scores_are_defective_probabilities():
    forest = holdout.construct_pipeline(
        "random_forest", "glcm",
        {"n_estimators": 5, "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None},
        1.0,
    )
    assert list(forest.named_steps) == ["classifier"]
    training = np.arange(16, dtype=np.float32).reshape(8, 2)
    forest.fit(training, [0, 0, 0, 0, 1, 1, 1, 1])
    labels, scores, _, _ = holdout.infer(forest, training, "random_forest")
    np.testing.assert_array_equal(labels, forest.predict(training))
    np.testing.assert_allclose(scores, forest.predict_proba(training)[:, 1])
    assert scores[-1] > scores[0]


@pytest.mark.parametrize("model,feature,parameters,weight", [
    ("svm", "glcm", {"C": 3}, 1.0),
    ("svm", "glcm", {"C": 3, "gamma": "scale"}, 2.0),
    ("random_forest", "glcm", {"n_estimators": 60}, 1.0),
])
def test_incomplete_or_inconsistent_pipeline_fails(model, feature, parameters, weight):
    with pytest.raises(ValueError):
        holdout.construct_pipeline(model, feature, parameters, weight)


def test_metrics_counts_and_uncertainty():
    metrics, matrix = holdout.score_holdout([0, 0, 1, 1], [0, 0, 0, 1], [0.1, 0.3, 0.2, 0.9])
    np.testing.assert_array_equal(matrix, [[2, 0], [1, 1]])
    assert metrics["specificity"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["roc_auc"] == 0.75
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["recall_ci95_low"] < metrics["recall"] < metrics["recall_ci95_high"]
    assert holdout.wilson(0, 156)[0] == 0
    assert holdout.wilson(156, 156)[1] == pytest.approx(1, abs=1e-15)


def test_guard_precedes_training_or_holdout_access(tmp_path, monkeypatch):
    output = tmp_path / holdout.LOG
    output.parent.mkdir(parents=True)
    output.write_text("Started an existing evaluation.", encoding="utf-8")
    monkeypatch.setattr(holdout, "protocol_from_training", lambda *_: pytest.fail("Accessed inputs"))
    monkeypatch.setattr(holdout, "validate_holdout", lambda *_: pytest.fail("Opened holdout"))
    with pytest.raises(FileExistsError, match=output.name):
        holdout.freeze(tmp_path)
    with pytest.raises(FileExistsError, match=output.name):
        holdout.run(tmp_path)
    assert output.read_text(encoding="utf-8") == "Started an existing evaluation."


def test_mismatched_protocol_cannot_open_holdout(tmp_path, monkeypatch):
    protocol = tmp_path / holdout.PROTOCOL
    protocol.parent.mkdir(parents=True)
    protocol.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(holdout, "protocol_from_training", lambda *_: ({"primary": "other"}, {}, []))
    monkeypatch.setattr(holdout, "validate_holdout", lambda *_: pytest.fail("Opened holdout"))
    with pytest.raises(ValueError, match="differs from the frozen protocol"):
        holdout.run(tmp_path)
    assert not (tmp_path / holdout.LOG).exists()


def test_synthetic_figures_and_error_rows(tmp_path):
    from PIL import Image

    (tmp_path / "results/tables").mkdir(parents=True)
    path = "sample.png"
    Image.fromarray(np.full((512, 512), 65, dtype=np.uint8)).save(tmp_path / path)
    with (tmp_path / "results/tables/dataset_audit.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "sha256"])
        writer.writerow([path, file_sha256(tmp_path / path)])
    test = [{"label": label, "image_id": str(i), "image_path": path}
            for i, label in enumerate([0, 0, 1, 1])]
    models, predictions, metrics = [], [], []
    for feature, name in holdout.FEATURE_NAMES.items():
        for model, classifier in holdout.MODEL_NAMES.items():
            primary = (feature, model) == ("hog16_glcm", "svm")
            candidate = holdout.PRIMARY if primary else f"{feature}_{model}"
            row = {"feature_set": feature, "model": model, "candidate": candidate,
                   "name": f"{name} + {classifier}", "primary": primary}
            models.append(row)
            truth = [0, 0, 1, 1]
            labels = [0, 1, 0, 1]
            scores = [0.1, 0.8, 0.4, 0.9]
            values, _ = holdout.score_holdout(truth, labels, scores)
            metrics.append({**row, **values, "fit_seconds": 0.1, "inference_seconds": 0.02})
            predictions.extend({"candidate": candidate, "image_id": item["image_id"],
                                "image_path": item["image_path"], "true_label": item["label"],
                                "predicted_label": label, "defective_score": score}
                               for item, label, score in zip(test, labels, scores))
    shared, examples = holdout.error_tables(test, predictions, models)
    assert len(shared) == 2
    assert {row["error"] for row in shared} == {"FP", "FN"}
    assert all(row["models_in_error"] == 6 for row in shared)
    assert len(examples) == 4
    assert holdout.error_tables(test, predictions, models) == (shared, examples)
    paths = plot_holdout(tmp_path, models, metrics, predictions, examples)
    assert len(paths) == len(set(paths)) == 11
    for path in paths:
        with Image.open(tmp_path / path) as image:
            assert image.info["dpi"][0] == pytest.approx(300, abs=0.1)
            image.verify()
