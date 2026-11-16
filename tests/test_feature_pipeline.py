"""Verify training-only resource decisions and split-aligned, tamper-evident caches."""

import json

import numpy as np
from PIL import Image
import pytest

from src.data_loader import MANIFEST_FIELDS, image_identifier
from src.feature_pipeline import (
    benchmark_method, benchmark_samples, cache_split, configuration, feasibility,
    file_sha256, load_cached_features,
)
from src.reporting import write_csv, write_json


def sample_rows(count=100):
    return [{"image_path": f"{index:03}.jpeg", "class_name": "def_front" if index < 60 else "ok_front",
             "split": "train"} for index in range(count)]


def test_benchmark_sampling_is_seeded_order_independent_and_training_only():
    rows = sample_rows()
    selected = benchmark_samples(rows)
    assert selected == benchmark_samples(list(reversed(rows)))
    assert selected != benchmark_samples(rows, seed=43)
    assert len({r["image_path"] for r in selected}) == 50
    assert sum(r["class_name"] == "def_front" for r in selected) == 30
    with pytest.raises(ValueError, match="training"):
        benchmark_samples([{**r, "split": "test"} for r in rows])
    with pytest.raises(ValueError, match="test"):
        benchmark_method(None, [{"split": "test"}], "hog8")


def test_feasibility_uses_only_measured_resources_and_budget():
    benchmark = {"method": "hog8", "images": 50, "vector_bytes": 571536,
                 "end_to_end_seconds": 2, "peak_rss_bytes": 100 * 1024**2}
    resources = {"disk_free_bytes": 10 * 1024**3, "available_ram_bytes": 2 * 1024**3}
    assert feasibility([benchmark], 1300, resources)[0]["feasible"]
    assert not feasibility([benchmark], 1300, resources, max_projected_seconds=1)[0]["feasible"]
    assert not feasibility([benchmark], 1300, {**resources, "disk_free_bytes": 100})[0]["feasible"]
    assert not feasibility([benchmark], 1300, {**resources, "available_ram_bytes": 100})[0]["feasible"]


@pytest.fixture
def cached_project(tmp_path):
    tables = tmp_path / "results/tables"
    tables.mkdir(parents=True)
    splits = tmp_path / "data/splits"
    splits.mkdir(parents=True)
    manifests = {}
    for split in ("train", "test"):
        rows = []
        for name, label in (("def_front", 1), ("ok_front", 0)):
            path = tmp_path / f"data/raw/casting_512x512/{name}/{split}.jpeg"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.tile(np.arange(512, dtype=np.uint16) % 256, (512, 1)).astype(np.uint8)
            Image.fromarray(pixels).save(path)
            relative = path.relative_to(tmp_path).as_posix()
            rows.append({"image_path": relative, "image_id": image_identifier(relative),
                         "class_name": name, "label": label, "split": split})
        manifests[split] = rows
        write_csv(splits / f"{split}.csv", rows, MANIFEST_FIELDS)
    fold_map = {r["image_path"]: index + 1 for index, r in enumerate(manifests["train"])}
    write_csv(tables / "cv_folds.csv", [{"image_path": path, "validation_fold": fold} for path, fold in fold_map.items()],
              ["image_path", "validation_fold"])
    cache = tmp_path / "results/cache/features"
    before = {str(p): p.read_bytes() for p in (tmp_path / "data").rglob("*") if p.is_file()}
    entries = []
    for split, rows in manifests.items():
        entries.extend(cache_split(tmp_path, split, rows, ["hog8", "hog16", "glcm"], fold_map, cache))
    assert before == {str(p): p.read_bytes() for p in (tmp_path / "data").rglob("*") if p.is_file()}
    fingerprint_paths = ["data/splits/train.csv", "data/splits/test.csv", "results/tables/cv_folds.csv"]
    write_json(tables / "feature_run_summary.json", {
        "status": "complete", "caches": entries,
        "provenance": {"configuration": json.loads(json.dumps(configuration())),
                       "input_sha256": {p: file_sha256(tmp_path / p) for p in fingerprint_paths}},
    })
    return tmp_path, manifests, entries


def test_cache_shapes_float32_readonly_and_partition_row_alignment(cached_project):
    root, manifests, entries = cached_project
    for split in manifests:
        for method, columns in (("hog8", 142884), ("hog16", 34596), ("glcm", 48)):
            matrix, labels, index = load_cached_features(root, method, split)
            assert isinstance(matrix, np.memmap) and not matrix.flags.writeable
            assert matrix.shape == (2, columns) and matrix.dtype == np.float32
            assert np.isfinite(matrix).all()
            np.testing.assert_array_equal(labels, [1, 0])
            assert [r["image_id"] for r in index] == [r["image_id"] for r in manifests[split]]
            assert all(r["validation_fold"] == "" for r in index) if split == "test" else all(r["validation_fold"] for r in index)
            del matrix
    assert all(r["file_bytes"] > r["payload_bytes"] for r in entries)


@pytest.mark.parametrize("target", ["cache", "index", "labels", "manifest", "cv"])
def test_cache_loader_rejects_stale_or_tampered_content(cached_project, target):
    root, _, _ = cached_project
    paths = {"cache": "results/cache/features/train_hog8.npy", "index": "results/tables/feature_train_index.csv",
             "labels": "results/cache/features/train_labels.npy", "manifest": "data/splits/train.csv",
             "cv": "results/tables/cv_folds.csv"}
    path = root / paths[target]
    with path.open("ab") as stream:
        stream.write(b"fixture-only tampering")
    with pytest.raises(ValueError, match="checksum|stale"):
        load_cached_features(root, "hog8", "train")


def test_cache_writer_rejects_mixed_splits_and_overwriting(cached_project):
    root, manifests, _ = cached_project
    cache = root / "results/cache/features"
    with pytest.raises(ValueError, match="partition"):
        cache_split(root, "train", manifests["train"] + manifests["test"], ["glcm"], {}, cache)
    with pytest.raises(FileExistsError):
        cache_split(root, "train", manifests["train"], ["glcm"], {}, cache)


def test_training_feature_figures_have_correct_labels_and_separated_panels(monkeypatch, tmp_path):
    from src import visualization

    checked = []
    def inspect_figure(figure, path):
        figure.tight_layout(rect=(0, 0.035, 1, 0.95))
        figure.canvas.draw()
        if path.name == "feature_hog_comparison.png":
            assert "px/cell" in figure.axes[1].get_title()
            assert "px/cell" in figure.axes[2].get_title()
            renderer = figure.canvas.get_renderer()
            for column in range(3):
                upper = figure.axes[column].get_window_extent(renderer)
                lower_title = figure.axes[column + 3].title.get_window_extent(renderer)
                assert lower_title.y1 < upper.y0
            assert all(axis.images[0].get_array().shape == (512, 512) for axis in figure.axes)
        if path.name == "feature_glcm_properties.png":
            assert len(figure.legends) == 1
            assert all(axis.get_legend() is None for axis in figure.axes)
        checked.append(path.name)
        visualization._pyplot().close(figure)
    monkeypatch.setattr(visualization, "_save_eda", inspect_figure)
    samples = [{"image_path": name + ".jpeg", "class_name": name, "split": "train",
                "image": np.tile(np.linspace(0, 1, 512, dtype=np.float32), (512, 1))}
               for name in ("ok_front", "def_front")]
    with visualization.publication_style():
        rows = visualization.plot_texture_features(samples, tmp_path)
    assert len(rows) == 96
    assert len(checked) == 3
    with pytest.raises(ValueError, match="512"):
        visualization.plot_texture_features([{**samples[0], "image": np.zeros((32, 32))}], tmp_path)
