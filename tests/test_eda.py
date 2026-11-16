"""Verify EDA statistics, training scope, and read-only input handling."""

import csv
import hashlib

import numpy as np
from PIL import Image
import pytest

from src.data_loader import CLASS_LABELS, MANIFEST_FIELDS, image_identifier
import src.eda as eda
from src.reporting import write_csv
from src import visualization


def test_known_pixel_statistics_use_population_sd():
    stats = eda.image_statistics(np.array([[0, 0], [0, 4]], dtype=np.uint8))
    assert stats["mean"] == 1.0
    assert stats["std"] == pytest.approx(np.sqrt(3))
    assert (stats["minimum"], stats["maximum"], stats["pixel_count"]) == (0, 4, 4)


@pytest.mark.parametrize("normalized", [False, True])
def test_histograms_include_both_endpoints_and_sum_to_one(normalized):
    maximum = 1 if normalized else 255
    counts, edges = eda.intensity_histogram(
        np.array([[0.0, maximum]]), normalized=normalized
    )
    assert counts.sum() == 2
    assert counts[0] == 1 and counts[-1] == 1
    assert (counts / counts.sum()).sum() == 1.0
    assert len(edges) == 257


def test_histogram_does_not_silently_drop_out_of_range_pixels():
    with pytest.raises(ValueError, match="range"):
        eda.intensity_histogram(np.array([[-1.0, 5.0]]))
    with pytest.raises(ValueError, match="range"):
        eda.intensity_histogram(np.array([[0.0, 1.01]]), normalized=True)


@pytest.mark.parametrize("a,b,expected", [
    ([3, 4], [1, 2], 1.0), ([1, 2], [3, 4], -1.0),
    ([1, 1], [1, 1], 0.0), ([1, 2], [1, 3], -0.25),
])
def test_cliffs_delta_handles_order_and_ties(a, b, expected):
    assert eda.cliffs_delta(a, b) == expected


def test_pooled_sd_is_distinct_from_mean_within_image_sd():
    rows = []
    for value in (0, 2):
        rows.append({
            "split": "train", "analysis_role": "training_analysis",
            "stage": "original", "class_name": "ok_front",
            **eda.image_statistics(np.full((2, 2), value)),
        })
    result = eda.summarize_classes(rows)[0]
    assert result["pooled_pixel_mean"] == 1.0
    assert result["pooled_pixel_std"] == 1.0
    assert result["mean_within_image_std"] == 0.0
    assert result["sd_image_mean_ddof1"] == pytest.approx(np.sqrt(2))


def test_sampling_is_seeded_sorted_and_training_only():
    rows = [{"split": "train", "class_name": name, "image_path": f"{name}/{i:03}"}
            for name in CLASS_LABELS for i in range(20)]
    first = eda.select_samples(rows)
    assert first == eda.select_samples(list(reversed(rows)))
    assert first != eda.select_samples(rows, seed=43)
    assert len({row["image_path"] for row in first}) == 6
    with pytest.raises(ValueError, match="training"):
        eda.select_samples([{**rows[0], "split": "test"}])


def test_test_rows_cannot_enter_comparisons_or_analysis_plots(tmp_path):
    with pytest.raises(ValueError, match="training"):
        eda.compare_classes([{"split": "test"}])
    with pytest.raises(ValueError, match="training"):
        visualization.plot_class_histograms([{"split": "test"}], tmp_path / "bad.png")


@pytest.fixture
def project(tmp_path):
    manifests = {"train": [], "test": []}
    for name, label in CLASS_LABELS.items():
        directory = tmp_path / "data/raw/casting_512x512" / name
        directory.mkdir(parents=True)
        intensities = [30, 60, 90, 0] if label == 0 else [100, 120, 140, 255]
        for index, value in enumerate(intensities):
            path = directory / f"fixture_{index}.jpeg"
            Image.fromarray(np.full((512, 512), value, dtype=np.uint8)).save(path)
            relative = path.relative_to(tmp_path).as_posix()
            split = "test" if index == 3 else "train"
            manifests[split].append({
                "image_path": relative, "image_id": image_identifier(relative),
                "class_name": name, "label": label, "split": split,
            })
    splits = tmp_path / "data/splits"
    splits.mkdir()
    for split, rows in manifests.items():
        write_csv(
            splits / f"{split}.csv",
            sorted(rows, key=lambda row: row["image_path"]), MANIFEST_FIELDS,
        )
    return tmp_path


@pytest.fixture
def compact_figures(monkeypatch):
    # Integration tests exercise real plotting at fixture resolution; real
    # Stage 2 exports retain their separate, verified 300-DPI save profile.
    def save_fixture(figure, path):
        visualization._save(figure, path, dpi=35)
    monkeypatch.setattr(visualization, "_save_eda", save_fixture)


def run_fixture(project, **kwargs):
    return eda.run_eda(project, expected_counts={"ok_front": 4, "def_front": 4}, **kwargs)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_end_to_end_preserves_inputs_and_excludes_test_decoding(
    project, compact_figures, monkeypatch,
):
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (project / "data").rglob("*") if path.is_file()}
    real_loader = eda.load_grayscale
    def training_loader(path):
        if path.name == "fixture_3.jpeg":
            raise AssertionError("Default EDA must not decode test images")
        return real_loader(path)
    monkeypatch.setattr(eda, "load_grayscale", training_loader)
    result = run_fixture(project, size=(17, 23))
    assert result["processed_images"] == 6
    assert result["test_images_described"] == 0
    assert result["preprocessing_validation_passed"]
    assert result["constant_resized_images"] == 6
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (project / "data").rglob("*") if path.is_file()}
    rows = read_csv(project / "results/tables/eda_preprocessing_validation.csv")
    assert len(rows) == 6 and {row["output_dtype"] for row in rows} == {"float32"}
    assert {row["normalized_maximum"] for row in rows} == {"0.0"}
    for name in eda.FIGURE_NAMES:
        with Image.open(project / "results/figures" / name) as image:
            assert image.format == "PNG"
    assert not (project / "results/models").exists()


def test_optional_test_descriptors_do_not_change_training_comparisons(
    project, compact_figures,
):
    run_fixture(project)
    tables = project / "results/tables"
    before_comparisons = (tables / "eda_class_comparisons.csv").read_bytes()
    before_samples = (tables / "eda_selected_samples.csv").read_bytes()
    training = [
        row for row in read_csv(tables / "eda_class_statistics.csv")
        if row["split"] == "train"
    ]
    result = run_fixture(project, include_test_descriptive=True)
    assert (result["processing_height"], result["processing_width"]) == (512, 512)
    assert not result["resizing_applied"]
    assert result["processed_images"] == 8
    assert result["test_images_described"] == 2
    assert not result["test_pixel_statistics_used_in_comparisons_or_figures"]
    assert before_comparisons == (tables / "eda_class_comparisons.csv").read_bytes()
    assert before_samples == (tables / "eda_selected_samples.csv").read_bytes()
    assert training == [
        row for row in read_csv(tables / "eda_class_statistics.csv")
        if row["split"] == "train"
    ]
    test_rows = [
        row for row in read_csv(tables / "eda_image_statistics.csv")
        if row["split"] == "test"
    ]
    assert test_rows and {row["analysis_role"] for row in test_rows} == {"test_descriptive_only"}


def test_raw_change_during_processing_is_detected(project, monkeypatch):
    original_loader = eda.load_grayscale
    changed = False
    def mutate_fixture(path):
        nonlocal changed
        image = original_loader(path)
        if not changed:
            path.write_bytes(path.read_bytes() + b"fixture-only change")
            changed = True
        return image
    monkeypatch.setattr(eda, "load_grayscale", mutate_fixture)
    with pytest.raises(ValueError, match="changed during EDA"):
        run_fixture(project)


def test_output_cannot_target_raw_data_or_splits(project):
    for target in (project / "data/raw", project / "data/splits"):
        with pytest.raises(ValueError, match="results directory"):
            run_fixture(project, results_directory=target)


def test_comparison_plot_uses_fixed_intensity_display_ranges(monkeypatch, tmp_path):
    samples = [{
        "split": "train", "class_name": name,
        "original": np.full((4, 4), 100, dtype=np.uint8),
        "resized": np.full((2, 2), 100, dtype=np.float32),
        "normalized": np.zeros((2, 2), dtype=np.float32),
    } for name in CLASS_LABELS]
    captured = []
    def inspect(figure, path):
        captured.extend(axis.images[0].get_clim() for axis in figure.axes)
        visualization._pyplot().close(figure)
    monkeypatch.setattr(visualization, "_save_eda", inspect)
    visualization.plot_preprocessing_comparison(samples, tmp_path / "comparison.png")
    assert captured == [(0, 255), (0, 255), (0, 1)] * 2


def test_publication_save_profile_is_300_dpi(tmp_path):
    plt = visualization._pyplot()
    figure, axis = plt.subplots(figsize=(2, 2))
    axis.plot([0, 1], [0, 1])
    path = tmp_path / "publication.png"
    visualization._save_eda(figure, path)
    with Image.open(path) as image:
        assert image.info["dpi"][0] == pytest.approx(300, abs=0.1)


def test_cli_defaults_to_native_512_without_reading_real_data(monkeypatch):
    captured = {}

    def capture_run(**kwargs):
        captured.update(kwargs)
        return {"configuration_checked": True}

    monkeypatch.setattr(eda, "run_eda", capture_run)
    assert eda.main([]) == 0
    assert captured["size"] == (512, 512)
    assert eda.main(["--size", "384", "400"]) == 0
    assert captured["size"] == (384, 400)
