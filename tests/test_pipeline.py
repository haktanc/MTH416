"""Exercise data safety, mathematical features, and outputs without training."""

import importlib

import cv2
import numpy as np
import pytest

from src.data_loader import (check_integrity, collect_records, find_dataset,
                             image_counts, load_grayscale)
from src.evaluation import evaluate_predictions, save_comparison
from src.feature_extraction import (FeatureConfig, extract_features, extract_glcm,
                                    extract_hog, glcm_properties)
from src.model_training import build_classifier, load_model, save_model
from src.preprocessing import min_max_normalize, prepare_image


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "casting_data" / "casting_data"
    for split in ("train", "test"):
        for name in ("ok_front", "def_front"):
            directory = root / split / name
            directory.mkdir(parents=True)
            cv2.imwrite(str(directory / "sample.png"),
                        np.arange(256, dtype=np.uint8).reshape(16, 16))
    return root


def test_all_modules_import():
    for name in ("data_loader", "preprocessing", "feature_extraction",
                 "model_training", "evaluation", "visualization"):
        importlib.import_module(f"src.{name}")
    importlib.import_module("main")


def test_dataset_discovery_and_integrity(dataset):
    assert find_dataset(dataset.parent.parent) == dataset
    records = collect_records(dataset)
    assert len(records) == 4
    assert {r.label for r in records if r.class_name == "def_front"} == {1}
    assert sum(row["image_count"] for row in image_counts(records)) == 4
    assert check_integrity(records) == []
    records[0].path.write_bytes(b"invalid image")
    assert len(check_integrity(records)) == 1


def test_missing_and_ambiguous_datasets(dataset, tmp_path):
    with pytest.raises(FileNotFoundError):
        find_dataset(tmp_path / "missing")
    second = tmp_path / "second"
    for split in ("train", "test"):
        for name in ("ok_front", "def_front"):
            (second / split / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="found 2"):
        find_dataset(tmp_path)


def test_grayscale_and_preprocessing(dataset):
    image = load_grayscale(dataset / "train" / "ok_front" / "sample.png")
    processed = prepare_image(image, (32, 48))
    assert processed.shape == (32, 48)
    assert processed.dtype == np.float32
    assert processed.min() == 0 and processed.max() == 1
    np.testing.assert_array_equal(min_max_normalize(np.ones((4, 4))), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        prepare_image(np.zeros((2, 2, 3)))
    with pytest.raises(ValueError):
        min_max_normalize(np.array([np.nan]))


def test_feature_dimensions_and_glcm_values():
    image = np.zeros((32, 32), dtype=np.float32)
    config = FeatureConfig(distances=(1,), angles=(0.0,))
    properties = glcm_properties(image, config)
    assert properties["contrast"][0, 0] == 0
    assert properties["homogeneity"][0, 0] == 1
    assert properties["energy"][0, 0] == 1
    hog_vector = extract_hog(image, config)
    glcm_vector = extract_glcm(image, config)
    assert hog_vector.size == 3 * 3 * 2 * 2 * 9
    assert glcm_vector.size == 6
    np.testing.assert_array_equal(
        extract_features(image, config), np.concatenate((hog_vector, glcm_vector)))
    with pytest.raises(ValueError):
        extract_features(image, method="unknown")
    with pytest.raises(ValueError):
        extract_features(np.full((32, 32), 255.0))


def test_metrics_and_csv(tmp_path):
    # Synthetic labels test metric arithmetic; these are not experiment results.
    metrics, matrix = evaluate_predictions([0, 0, 1, 1], [0, 1, 0, 1],
                                           [0.1, 0.8, 0.4, 0.9])
    for name in ("accuracy", "precision", "recall", "f1"):
        assert metrics[name] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    np.testing.assert_array_equal(matrix, [[1, 1], [1, 1]])
    path = tmp_path / "comparison.csv"
    save_comparison({"synthetic_test": metrics}, path)
    assert path.is_file()
    single_class, _ = evaluate_predictions([0, 0], [0, 0], [0.1, 0.2])
    assert np.isnan(single_class["roc_auc"])


def test_model_configuration_and_serialization_without_fit(tmp_path):
    svm = build_classifier("svm")
    assert "scaler" in svm.named_steps
    assert not hasattr(svm.named_steps["classifier"], "support_")
    forest = build_classifier("random_forest")
    assert forest.named_steps["classifier"].n_jobs == 1
    assert not hasattr(forest.named_steps["classifier"], "estimators_")
    path = tmp_path / "unfitted.joblib"
    save_model(svm, path, metadata={"image_size": (128, 128)})
    assert load_model(path)["metadata"]["image_size"] == (128, 128)


def test_preview_and_csv_outputs(dataset, tmp_path):
    from main import inspect_dataset

    output = tmp_path / "results"
    inspect_dataset(dataset, output, verify_images=True, preview=True)
    assert (output / "tables" / "dataset_counts.csv").is_file()
    assert (output / "tables" / "integrity_failures.csv").is_file()
    for name in ("samples", "intensity_histogram", "hog", "glcm"):
        assert (output / "figures" / f"{name}.png").read_bytes().startswith(b"\x89PNG")
    assert not (output / "models").exists()
