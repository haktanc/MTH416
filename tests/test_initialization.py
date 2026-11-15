"""Verify initialization using temporary images, without experiments."""

import importlib
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import main
from src.data_loader import (
    collect_primary_records,
    load_grayscale,
    primary_image_counts,
)


@pytest.fixture
def primary_dataset(tmp_path):
    """Build a tiny unsplit fixture outside the real raw dataset."""
    root = tmp_path / "unicode_\u00f6" / "casting_512x512"
    for name in ("def_front", "ok_front"):
        directory = root / name
        directory.mkdir(parents=True)
        image = np.full((512, 512), 127, dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        assert success
        (directory / "sample.png").write_bytes(encoded.tobytes())
    return root


def test_primary_records_preserve_unsplit_labels(primary_dataset):
    records = collect_primary_records(primary_dataset)
    assert primary_image_counts(records) == {"ok_front": 1, "def_front": 1}
    assert {record.split for record in records} == {"unsplit"}
    assert {(r.class_name, r.label) for r in records} == {
        ("ok_front", 0), ("def_front", 1),
    }
    assert all(record.path.is_absolute() for record in records)
    assert records == collect_primary_records(primary_dataset)


def test_startup_reads_samples_without_writing(
    primary_dataset, monkeypatch, capsys,
):
    monkeypatch.setattr(main, "EXPECTED_PRIMARY_COUNTS", {
        "def_front": 1, "ok_front": 1,
    })
    before = {
        path.relative_to(primary_dataset): path.read_bytes()
        for path in primary_dataset.rglob("*") if path.is_file()
    }
    main.initialize_project(primary_dataset)
    after = {
        path.relative_to(primary_dataset): path.read_bytes()
        for path in primary_dataset.rglob("*") if path.is_file()
    }
    assert before == after
    assert {p.name for p in primary_dataset.iterdir()} == {
        "def_front", "ok_front",
    }
    assert "Total primary images: 2" in capsys.readouterr().out


def test_startup_rejects_wrong_count(primary_dataset):
    with pytest.raises(ValueError, match="Unexpected image count"):
        main.initialize_project(primary_dataset)


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        collect_primary_records(tmp_path / "missing")


def test_missing_class(tmp_path):
    (tmp_path / "ok_front").mkdir()
    with pytest.raises(ValueError, match="Missing class directory"):
        collect_primary_records(tmp_path)


@pytest.mark.parametrize("entry", ["unknown_class", "train"])
def test_unexpected_root_directory(primary_dataset, entry):
    (primary_dataset / entry).mkdir()
    with pytest.raises(ValueError, match="Unexpected dataset entries"):
        collect_primary_records(primary_dataset)


@pytest.mark.parametrize("entry", ["nested", "notes.txt"])
def test_unexpected_class_entry(primary_dataset, entry):
    path = primary_dataset / "ok_front" / entry
    if entry == "nested":
        path.mkdir()
    else:
        path.write_text("Not an image.", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected entries"):
        collect_primary_records(primary_dataset)


def test_empty_class(primary_dataset):
    # This removes only a temporary fixture, never a real dataset image.
    (primary_dataset / "def_front" / "sample.png").unlink()
    with pytest.raises(ValueError, match="No images found"):
        collect_primary_records(primary_dataset)


@pytest.mark.parametrize("content", [b"", b"not a valid image"])
def test_unreadable_image(tmp_path, content):
    path = tmp_path / "invalid.jpeg"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Cannot decode"):
        load_grayscale(path)


def test_wrong_sample_dimensions(primary_dataset, monkeypatch):
    monkeypatch.setattr(main, "EXPECTED_PRIMARY_COUNTS", {
        "def_front": 1, "ok_front": 1,
    })
    monkeypatch.setattr(main, "load_grayscale", lambda _: np.zeros((32, 32)))
    with pytest.raises(ValueError, match="Unexpected sample dimensions"):
        main.initialize_project(primary_dataset)


def test_cli_failure_has_nonzero_exit(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main.main(["--data-dir", str(tmp_path / "missing")])
    assert error.value.code == 1
    assert "Dataset directory does not exist" in capsys.readouterr().err


def test_module_imports_do_not_load_optional_analysis_stack():
    script = """
import importlib
import sys
for name in (
    'data_loader', 'preprocessing', 'feature_extraction',
    'model_training', 'evaluation', 'visualization',
):
    importlib.import_module('src.' + name)
importlib.import_module('main')
unexpected = set(sys.modules).intersection({
    'pandas', 'matplotlib', 'seaborn', 'skimage', 'sklearn', 'joblib',
    'IPython', 'jupyter',
})
assert not unexpected, unexpected
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
