"""Benchmark and cache native-resolution CPU features without model fitting."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from src.data_loader import PROJECT_ROOT, collect_primary_records, load_grayscale, read_manifest
from src.feature_extraction import (
    FeatureConfig, GLCM_PROPERTIES, extract_features, hog_feature_length,
)
from src.preprocessing import DEFAULT_IMAGE_SIZE, prepare_image
from src.reporting import inventory_digest, write_csv, write_json


CONFIGS = {
    "hog8": FeatureConfig(pixels_per_cell=(8, 8)),
    "hog16": FeatureConfig(pixels_per_cell=(16, 16)),
    "glcm": FeatureConfig(),
}
INDEX_FIELDS = ["feature_row", "image_path", "image_id", "class_name", "label", "split", "validation_fold"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    import csv

    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def process_memory() -> dict:
    """Measure native process working set and lifetime peak, including extensions."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "pagefile", "peak_pagefile", "private",
                )
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not function(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("Could not measure process working-set memory.")
        return {"rss_bytes": counters.rss, "peak_rss_bytes": counters.peak_rss}
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"rss_bytes": None, "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}


def system_resources(root: Path) -> dict:
    result = {"disk_free_bytes": shutil.disk_usage(root).free,
              "platform": platform.platform(), "cpu": platform.processor(),
              "logical_cpus": os.cpu_count()}
    if os.name == "nt":
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total", "available", "total_page", "available_page",
                    "total_virtual", "available_virtual", "extended",
                )
            ]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("Could not measure available physical memory.")
        result.update(total_ram_bytes=memory.total, available_ram_bytes=memory.available)
    else:
        result.update(total_ram_bytes=os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
                      available_ram_bytes=os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    return result


def configuration() -> dict:
    return {"image_shape": list(DEFAULT_IMAGE_SIZE), "dtype": "float32",
            "normalization": "Per-image Min-Max; constants become zeros; no resizing",
            "grayscale": "OpenCV IMREAD_GRAYSCALE", "hog_transform_sqrt": False,
            "glcm_symmetric": True, "glcm_normed": True,
            "glcm_quantization": "min(floor(normalized_pixel * 32), 31)",
            "glcm_feature_order": "property, distance, angle (C order)",
            "glcm_properties": list(GLCM_PROPERTIES),
            "angles_degrees": [0, 45, 90, 135],
            "methods": {name: asdict(config) for name, config in CONFIGS.items()}}


def inputs(root: Path) -> tuple[dict, dict, dict]:
    manifests = {split: read_manifest(root / f"data/splits/{split}.csv", split, root)
                 for split in ("train", "test")}
    primary = collect_primary_records(root / "data/raw/casting_512x512")
    all_rows = manifests["train"] + manifests["test"]
    paths = [r["image_path"] for r in all_rows]
    if len(set(paths)) != len(paths) or set(paths) != {r.path.relative_to(root).as_posix() for r in primary}:
        raise ValueError("Manifests must cover the primary collection exactly once.")
    hashes = {r.path.relative_to(root).as_posix(): file_sha256(r.path) for r in primary}
    audit_path = root / "results/tables/dataset_audit.csv"
    if audit_path.exists():
        audited = {r["image_path"]: r["sha256"] for r in read_csv(audit_path)}
        if audited != hashes:
            raise ValueError("Primary raw bytes differ from the saved Stage 1 inventory.")
    fingerprints = {f"data/splits/{split}.csv": file_sha256(root / f"data/splits/{split}.csv")
                    for split in manifests}
    cv_path = root / "results/tables/cv_folds.csv"
    fingerprints["results/tables/cv_folds.csv"] = file_sha256(cv_path)
    folds = read_csv(cv_path)
    train_by_path = {r["image_path"]: r for r in manifests["train"]}
    if len(folds) != len(train_by_path) or {r["image_path"] for r in folds} != set(train_by_path):
        raise ValueError("Saved CV assignments must cover only training images exactly once.")
    for row in folds:
        original = train_by_path[row["image_path"]]
        if row["image_id"] != original["image_id"] or int(row["label"]) != original["label"]:
            raise ValueError("CV image identities or labels differ from the manifest.")
        if int(row["validation_fold"]) not in range(1, 6):
            raise ValueError("Expected saved five-fold assignments.")
    source_names = ("feature_pipeline.py", "feature_extraction.py", "preprocessing.py", "data_loader.py", "visualization.py")
    stamp = {"input_sha256": fingerprints,
             "source_inventory_sha256": inventory_digest([
                 {"image_path": path, "sha256": digest} for path, digest in hashes.items()]),
             "source_code_sha256": {f"src/{name}": file_sha256(root / "src" / name) for name in source_names},
             "configuration": configuration(),
             "versions": {name: version(name) for name in ("numpy", "scikit-image", "opencv-python", "scipy")}}
    # Convert tuples to JSON lists before comparing stamps loaded from disk.
    return manifests, {r["image_path"]: int(r["validation_fold"]) for r in folds}, json.loads(json.dumps(stamp))


def load_native(root: Path, row: dict) -> np.ndarray:
    image = load_grayscale(root / row["image_path"])
    if image.shape != DEFAULT_IMAGE_SIZE or image.dtype != np.uint8:
        raise ValueError("Feature extraction requires native 512 x 512 uint8 decoding.")
    return prepare_image(image)


def benchmark_samples(train: list[dict], count: int = 50, seed: int = 42) -> list[dict]:
    """Proportional class allocation without changing the dataset partitions."""
    if count < 2 or count > len(train) or any(r["split"] != "train" for r in train):
        raise ValueError("Benchmark size must be 2..training size; training rows only.")
    classes = sorted({r["class_name"] for r in train})
    if len(classes) != 2:
        raise ValueError("Both training classes are required.")
    proportions = np.array([sum(r["class_name"] == name for r in train) for name in classes]) * count / len(train)
    allocated = np.floor(proportions).astype(int)
    for index in np.argsort(-(proportions - allocated), kind="stable")[:count - allocated.sum()]:
        allocated[index] += 1
    if min(allocated) < 1:
        raise ValueError("Benchmark sample must include both classes.")
    rng, selected = np.random.default_rng(seed), []
    for name, amount in zip(classes, allocated):
        rows = sorted((r for r in train if r["class_name"] == name), key=lambda r: r["image_path"])
        selected.extend(rows[int(i)] for i in rng.choice(len(rows), int(amount), replace=False))
    return sorted(selected, key=lambda r: r["image_path"])


def feature_vector(image: np.ndarray, method: str) -> np.ndarray:
    if method not in CONFIGS:
        raise ValueError("Unknown feature method.")
    return extract_features(image, CONFIGS[method], "glcm" if method == "glcm" else "hog")


def benchmark_method(root: Path, rows: list[dict], method: str) -> dict:
    """One warm-up then a sequential pass; descriptor timing excludes decoding."""
    if not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("Benchmarks must not use test images.")
    image = load_native(root, rows[0])
    feature_vector(image, method)
    baseline = process_memory()
    io_seconds = feature_seconds = 0.0
    durations = []
    started, cpu_started = time.perf_counter(), time.process_time()
    for row in rows:
        tick = time.perf_counter()
        image = load_native(root, row)
        io_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        vector = feature_vector(image, method)
        duration = time.perf_counter() - tick
        feature_seconds += duration
        durations.append(duration)
        if vector.dtype != np.float32 or not np.isfinite(vector).all():
            raise ValueError("Benchmark produced invalid features.")
    return {"method": method, "images": len(rows), "feature_dimension": int(vector.size),
            "feature_dtype": str(vector.dtype), "vector_bytes": vector.nbytes,
            "decode_normalize_seconds": io_seconds, "feature_seconds": feature_seconds,
            "feature_ms_per_image": feature_seconds * 1000 / len(rows),
            "median_feature_ms": float(np.median(durations)) * 1000,
            "end_to_end_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "baseline_rss_bytes": baseline["rss_bytes"], **process_memory()}


def run_benchmarks(root: Path, rows: list[dict]) -> list[dict]:
    results = []
    for method in CONFIGS:
        print(f"Benchmarking {method} on {len(rows)} training images...", flush=True)
        command = [sys.executable, "-B", "-m", "src.feature_pipeline", "--worker", method,
                   "--project-root", str(root)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        result = subprocess.run(command, input=json.dumps(rows), capture_output=True,
                                text=True, check=True, env=env, cwd=PROJECT_ROOT)
        results.append(json.loads(result.stdout))
    return results


def feasibility(benchmarks: list[dict], count: int, resources: dict,
                max_projected_seconds: float = 600) -> list[dict]:
    """Apply resource gates only, never class/test performance criteria."""
    decisions, allocated_disk = [], 0
    for benchmark in benchmarks:
        required = count * benchmark["vector_bytes"] + 4096
        estimated = count / benchmark["images"] * benchmark["end_to_end_seconds"]
        reasons = []
        if allocated_disk + required > resources["disk_free_bytes"] * 0.9:
            reasons.append("Insufficient disk headroom for selected caches.")
        if resources["available_ram_bytes"] < max(512 * 1024 ** 2, 2 * benchmark["peak_rss_bytes"]):
            reasons.append("Insufficient RAM headroom relative to measured process peak.")
        if estimated > max_projected_seconds:
            reasons.append("Projected extraction exceeds configured time budget.")
        if not reasons:
            allocated_disk += required
        decisions.append({"method": benchmark["method"], "feasible": not reasons,
                          "projected_payload_bytes": count * benchmark["vector_bytes"],
                          "projected_extraction_seconds": estimated, "reasons": reasons})
    return decisions


def cache_split(root: Path, split: str, rows: list[dict], methods: list[str],
                fold_map: dict, cache: Path) -> list[dict]:
    """Stream .npy rows directly; never allocate a complete feature matrix."""
    if split not in ("train", "test") or any(row["split"] != split for row in rows):
        raise ValueError("Cache rows must belong to exactly one saved partition.")
    dimensions = {name: (48 if name == "glcm" else hog_feature_length(DEFAULT_IMAGE_SIZE, CONFIGS[name]))
                  for name in methods}
    targets = {name: cache / f"{split}_{name}.npy" for name in methods}
    if any(path.exists() or path.with_suffix(".npy.partial").exists() for path in targets.values()):
        raise FileExistsError("Feature cache exists; use a new --cache-dir for a fresh run.")
    cache.mkdir(parents=True, exist_ok=True)
    elapsed = {name: 0.0 for name in methods}
    io_seconds, started = 0.0, time.perf_counter()
    with ExitStack() as stack:
        streams = {}
        for name, path in targets.items():
            streams[name] = stack.enter_context(path.with_suffix(".npy.partial").open("wb"))
            np.lib.format.write_array_header_2_0(streams[name], {
                "descr": "<f4", "fortran_order": False,
                "shape": (len(rows), dimensions[name]),
            })
        for index, row in enumerate(rows):
            tick = time.perf_counter()
            image = load_native(root, row)
            io_seconds += time.perf_counter() - tick
            for name in methods:
                tick = time.perf_counter()
                vector = feature_vector(image, name)
                elapsed[name] += time.perf_counter() - tick
                if vector.shape != (dimensions[name],) or not np.isfinite(vector).all():
                    raise ValueError("Unexpected feature shape or nonfinite feature value.")
                streams[name].write(vector.astype("<f4", copy=False).tobytes())
            if (index + 1) % 100 == 0:
                print(f"Cached {split}: {index + 1}/{len(rows)} images", flush=True)
    extraction_seconds = time.perf_counter() - started
    entries = []
    for name, path in targets.items():
        path.with_suffix(".npy.partial").replace(path)
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        for offset in range(0, len(rows), 16):
            if not np.isfinite(matrix[offset:offset + 16]).all():
                raise ValueError("Saved cache contains invalid values.")
        entries.append({"split": split, "method": name, "rows": len(rows),
                        "columns": dimensions[name], "dtype": str(matrix.dtype),
                        "path": path.relative_to(root).as_posix(),
                        "payload_bytes": matrix.nbytes, "file_bytes": path.stat().st_size,
                        "sha256": file_sha256(path), "feature_seconds": elapsed[name],
                        "split_decode_normalize_seconds": io_seconds,
                        "split_extraction_wall_seconds": extraction_seconds})
        del matrix
    label_path = cache / f"{split}_labels.npy"
    np.save(label_path, np.array([r["label"] for r in rows], dtype=np.int8), allow_pickle=False)
    index_rows = [{"feature_row": i, **row,
                   "validation_fold": fold_map[row["image_path"]] if split == "train" else ""}
                  for i, row in enumerate(rows)]
    index_path = root / f"results/tables/feature_{split}_index.csv"
    write_csv(index_path, index_rows, INDEX_FIELDS)
    for entry in entries:
        entry.update(index_sha256=file_sha256(index_path), labels_sha256=file_sha256(label_path))
    return entries


def load_cached_features(root: Path, method: str, split: str, verify_hash: bool = True):
    """Return a read-only memory map, labels and index after provenance checks."""
    root = Path(root).resolve()
    if method not in CONFIGS or split not in ("train", "test"):
        raise ValueError("Unknown method or split.")
    summary = json.loads((root / "results/tables/feature_run_summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("Feature cache is not complete.")
    if summary["provenance"]["configuration"] != json.loads(json.dumps(configuration())):
        raise ValueError("Feature configuration changed; cache is stale.")
    for relative, digest in summary["provenance"].get("source_code_sha256", {}).items():
        if file_sha256(root / relative) != digest:
            raise ValueError("Feature/preprocessing source changed; cache is stale.")
    for name, expected_version in summary["provenance"].get("versions", {}).items():
        if version(name) != expected_version:
            raise ValueError("Dependency versions changed; cache is stale.")
    for relative, digest in summary["provenance"]["input_sha256"].items():
        if file_sha256(root / relative) != digest:
            raise ValueError("Saved manifests or CV assignments changed; cache is stale.")
    entry = next((r for r in summary["caches"] if r["method"] == method and r["split"] == split), None)
    if entry is None:
        raise ValueError("Requested cache was not extracted.")
    path = (root / entry["path"]).resolve()
    allowed = (root / "results/cache").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError("Feature cache path must stay under results/cache.")
    if verify_hash and file_sha256(path) != entry["sha256"]:
        raise ValueError("Feature cache checksum mismatch.")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (entry["rows"], entry["columns"]) or matrix.dtype != np.float32:
        raise ValueError("Feature cache shape or dtype mismatch.")
    index_path = root / f"results/tables/feature_{split}_index.csv"
    label_path = path.parent / f"{split}_labels.npy"
    if file_sha256(index_path) != entry["index_sha256"] or file_sha256(label_path) != entry["labels_sha256"]:
        raise ValueError("Feature index or label checksum mismatch.")
    rows = read_csv(index_path)
    manifest = read_manifest(root / f"data/splits/{split}.csv", split, root)
    if len(rows) != len(manifest):
        raise ValueError("Feature index length differs from the manifest.")
    for index, (row, source) in enumerate(zip(rows, manifest)):
        if int(row["feature_row"]) != index or any(str(row[k]) != str(source[k]) for k in source):
            raise ValueError("Feature row mapping differs from the manifest.")
    labels = np.load(label_path, allow_pickle=False)
    if not np.array_equal(labels, [r["label"] for r in manifest]):
        raise ValueError("Cached labels differ from the manifest.")
    return matrix, labels, rows


def run_stage3(root: Path = PROJECT_ROOT, benchmark_only: bool = False,
               reuse_benchmark: bool = False, cache_directory: Path | None = None,
               max_projected_seconds: float = 600) -> dict:
    from src import visualization

    started = time.perf_counter()
    root = Path(root).resolve()
    manifests, fold_map, stamp = inputs(root)
    if Counter(r["class_name"] for r in manifests["train"] + manifests["test"]) != {"def_front": 781, "ok_front": 519}:
        raise ValueError("Expected the accepted 1,300-image collection.")
    cache = Path(cache_directory or root / "results/cache/features").resolve()
    if not cache.is_relative_to((root / "results/cache").resolve()):
        raise ValueError("Large caches must stay under ignored results/cache.")
    tables, figures = root / "results/tables", root / "results/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    selected = benchmark_samples(manifests["train"])
    sample_rows = [{"benchmark_row": i, **row, "seed": 42} for i, row in enumerate(selected)]
    benchmark_path = tables / "feature_benchmark_summary.json"
    if reuse_benchmark:
        previous = json.loads(benchmark_path.read_text())
        if previous["provenance"] != stamp or previous["samples"] != sample_rows:
            raise ValueError("Benchmark provenance or sample selection changed; rebenchmark first.")
        benchmarks = previous["benchmarks"]
    else:
        benchmarks = run_benchmarks(root, selected)
        write_json(benchmark_path, {"provenance": stamp, "samples": sample_rows, "benchmarks": benchmarks})
    write_csv(tables / "feature_benchmark_samples.csv", sample_rows, list(sample_rows[0]))
    write_csv(tables / "feature_benchmark.csv", benchmarks, list(benchmarks[0]))
    resources = system_resources(root)
    decisions = feasibility(benchmarks, 1300, resources, max_projected_seconds)
    with visualization.publication_style():
        visualization.plot_feature_benchmark(benchmarks, decisions, figures / "feature_benchmark.png")
    if benchmark_only:
        print(json.dumps({"benchmarks": benchmarks, "resources": resources, "feasibility": decisions}, indent=2))
        return {"benchmarks": benchmarks, "resources": resources, "feasibility": decisions}
    methods = [row["method"] for row in decisions if row["feasible"]]
    if not methods:
        raise ValueError("No configuration passed the measured resource gates.")
    cache_names = [cache / f"{split}_{name}.npy" for split in manifests for name in methods]
    if any(p.exists() or p.with_suffix(".npy.partial").exists() for p in cache_names):
        raise FileExistsError("Cache exists; choose a fresh directory under results/cache.")
    entries = []
    extraction_started = time.perf_counter()
    for split, rows in manifests.items():
        entries.extend(cache_split(root, split, rows, methods, fold_map, cache))
    extraction_seconds = time.perf_counter() - extraction_started
    # Reuse the reviewed first training sample of each class from Stage 2.
    candidates = read_csv(tables / "eda_selected_samples.csv")
    examples = [r for r in candidates if int(r["sample_rank"]) == 1]
    if len(examples) != 2 or {r["class_name"] for r in examples} != {"def_front", "ok_front"}:
        raise ValueError("Expected one reviewed training example per class.")
    train_by_path = {r["image_path"]: r for r in manifests["train"]}
    for row in examples:
        if row["image_path"] not in train_by_path or row["image_id"] != train_by_path[row["image_path"]]["image_id"]:
            raise ValueError("Visualization sample must match the saved training manifest.")
    samples = [{**train_by_path[r["image_path"]], "image": load_native(root, r)} for r in examples]
    write_csv(tables / "feature_visualization_samples.csv", examples, list(examples[0]))
    with visualization.publication_style():
        sample_properties = visualization.plot_texture_features(samples, figures)
    write_csv(tables / "feature_sample_glcm.csv", sample_properties, list(sample_properties[0]))
    _, _, after = inputs(root)
    if after != stamp:
        raise ValueError("Raw images, manifests, CV, dependencies or source code changed during extraction.")
    summary = {"stage": 3, "status": "complete", "backend": "CPU", "seed": 42,
               "provenance": stamp, "benchmarks": benchmarks, "resources_before_extraction": resources,
               "feasibility": decisions, "caches": entries,
               "methods_extracted": methods, "train_images": len(manifests["train"]),
               "test_images": len(manifests["test"]), "benchmark_training_images": len(selected),
               "extraction_wall_seconds_including_writes_and_cache_validation": extraction_seconds,
               "workflow_wall_seconds": time.perf_counter() - started,
               "process_memory_after_extraction": process_memory(),
               "raw_images_unchanged": True, "manifests_and_cv_unchanged": True,
               "test_used_for_configuration_selection": False, "classifier_training_run": False,
               "warnings": ["Resource measurements are single runs, not performance benchmarks across machines.",
                            "Physical-part independence is unverified; perceptual candidates are advisory.",
                            "Stage 4 must compare configurations using the existing training-only CV assignments."]}
    write_csv(tables / "feature_cache_inventory.csv", entries, list(entries[0]))
    write_json(tables / "feature_run_summary.json", summary)
    for split in manifests:
        for method in methods:
            matrix, _, _ = load_cached_features(root, method, split)
            del matrix
    report = ["# Stage 3: Native-resolution HOG and GLCM", "",
              "CPU execution; all 1,300 images remain 512 x 512. Per-image Min-Max normalization; no fitted preprocessing.", "",
              "## Training-only benchmark", "",
              "50 deterministic training images (seed 42), shared by all methods; one untimed warm-up per isolated worker.",
              "Descriptor time excludes decoding/normalization. Peak RSS is the worker's lifetime OS working-set high-water mark, including native libraries.", "",
              "| Method | Dimensions | Feature seconds | ms/image | End-to-end seconds | Peak RSS MiB |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    report += [f"| {r['method']} | {r['feature_dimension']} | {r['feature_seconds']:.6f} | {r['feature_ms_per_image']:.4f} | {r['end_to_end_seconds']:.6f} | {r['peak_rss_bytes'] / 1024**2:.2f} |" for r in benchmarks]
    report += ["", "## Cached features", "", "| Split | Method | Shape | Payload MiB |", "| --- | --- | --- | ---: |"]
    report += [f"| {r['split']} | {r['method']} | {r['rows']} x {r['columns']} | {r['payload_bytes'] / 1024**2:.4f} |" for r in entries]
    report += ["", f"Extraction, writes and cache checks: {extraction_seconds:.4f} seconds.",
               f"Measured workflow through preservation checks: {summary['workflow_wall_seconds']:.4f} seconds.", "",
               "The HOG comparison shows local gradient structure at two cell scales; renderings are not feature vectors or classifier explanations.",
               "GLCM matrix panels show distance-1 probabilities for all four angles. Property panels include both distances and all six statistics.",
               "Caches are streamed float32 .npy files under ignored results/cache; indexes retain exact manifest row order and saved training validation folds.",
               "No combined matrices are duplicated on disk. Load read-only memory maps with load_cached_features; fit future scalers inside training CV only.",
               "File SHA-256, shape, dtype, index and label checks passed. Raw fingerprints, manifests and CV were unchanged.", "", "## Limitations", ""]
    report += ["- " + warning for warning in summary["warnings"]]
    (tables / "feature_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--reuse-benchmark", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-projected-seconds", type=float, default=600)
    parser.add_argument("--worker", choices=list(CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        import cv2
        cv2.setNumThreads(1)
        print(json.dumps(benchmark_method(args.project_root, json.load(sys.stdin), args.worker)))
    else:
        import cv2
        cv2.setNumThreads(1)
        run_stage3(args.project_root, args.benchmark_only, args.reuse_benchmark,
                   args.cache_dir, args.max_projected_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
