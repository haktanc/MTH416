"""Training-focused EDA and stateless preprocessing using existing manifests."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

from src.data_loader import (
    CLASS_LABELS,
    EXPECTED_PRIMARY_COUNTS,
    PROJECT_ROOT,
    collect_primary_records,
    load_grayscale,
    read_manifest,
)
from src.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    min_max_normalize,
    resize_grayscale,
    validate_grayscale,
    validate_size,
)
from src.reporting import inventory_digest, output_directory, write_csv, write_json
from src import visualization


STAGES = ("original", "resized", "normalized")
METRICS = ("mean", "std", "minimum", "maximum")
IMAGE_FIELDS = [
    "image_path", "image_id", "class_name", "label", "split", "analysis_role",
    "stage", "height", "width", "dtype", "pixel_count", *METRICS,
]
VALIDATION_FIELDS = [
    "image_path", "image_id", "class_name", "split", "analysis_role",
    "output_height", "output_width", "output_dtype", "finite", "bounds_valid",
    "normalized_minimum", "normalized_maximum", "original_constant",
    "resized_constant", "passed",
]
FIGURE_NAMES = [
    "eda_class_distribution.png", "eda_training_samples.png",
    "eda_training_intensity_histograms.png", "eda_training_image_statistics.png",
    "eda_preprocessing_comparison.png", "eda_preprocessing_histograms.png",
    "eda_training_class_comparisons.png",
]


def image_statistics(image: np.ndarray) -> dict:
    """Compute within-image population statistics in float64 arithmetic."""
    values = validate_grayscale(image)
    return {
        "height": values.shape[0], "width": values.shape[1],
        "dtype": str(values.dtype), "pixel_count": values.size,
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64, ddof=0)),
        "minimum": float(values.min()), "maximum": float(values.max()),
    }


def intensity_histogram(
    image: np.ndarray, *, normalized: bool = False, bins: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Count every pixel; original/resized units and normalized units differ."""
    values = validate_grayscale(image)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("Histogram bins must be an integer of at least two.")
    upper = 1.0 if normalized else 255.0
    if values.min() < 0 or values.max() > upper:
        raise ValueError("Histogram input is outside its documented intensity range.")
    edges = np.linspace(0.0, 1.0 if normalized else 256.0, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    if int(counts.sum()) != values.size:
        raise ValueError("Histogram binning lost pixel values.")
    return counts, edges


def cliffs_delta(defective: np.ndarray, ok: np.ndarray) -> float:
    """Return observed P(Defective > OK) - P(Defective < OK), including ties."""
    a, b = np.asarray(defective, dtype=float), np.asarray(ok, dtype=float)
    if (
        a.ndim != 1 or b.ndim != 1 or not a.size or not b.size
        or not np.isfinite(a).all() or not np.isfinite(b).all()
    ):
        raise ValueError("Class comparisons require finite nonempty vectors.")
    ordered = np.sort(b)
    greater = np.searchsorted(ordered, a, side="left").sum()
    less = (b.size - np.searchsorted(ordered, a, side="right")).sum()
    return float((greater - less) / (a.size * b.size))


def class_distribution(train: list[dict], test: list[dict]) -> list[dict]:
    """Describe labels from both manifests without inspecting test pixels."""
    result = []
    for split, rows in (("all", train + test), ("train", train), ("test", test)):
        counts = Counter(row["class_name"] for row in rows)
        for name, label in CLASS_LABELS.items():
            result.append({
                "split": split, "class_name": name, "label": label,
                "image_count": counts[name], "subset_total": len(rows),
                "percentage": 100.0 * counts[name] / len(rows),
            })
    return result


def select_samples(
    train: list[dict], samples_per_class: int = 3, seed: int = 42,
) -> list[dict]:
    """Select training entries from sorted class lists using a local RNG."""
    if not train or any(row["split"] != "train" for row in train):
        raise ValueError("Representative selection requires training rows only.")
    if not 1 <= samples_per_class <= 6:
        raise ValueError("Select between one and six samples per class.")
    random = np.random.default_rng(seed)
    selected = []
    for name in CLASS_LABELS:
        candidates = sorted(
            (row for row in train if row["class_name"] == name),
            key=lambda row: row["image_path"],
        )
        if len(candidates) < samples_per_class:
            raise ValueError(f"Too few training samples in {name}.")
        positions = random.choice(len(candidates), samples_per_class, replace=False)
        chosen = sorted(
            (candidates[int(index)] for index in positions),
            key=lambda row: row["image_path"],
        )
        selected.extend({**row, "sample_rank": rank, "seed": seed}
                        for rank, row in enumerate(chosen, 1))
    return selected


def summarize_classes(image_rows: list[dict]) -> list[dict]:
    """Distinguish pooled pixel SD from variability between image means."""
    result = []
    for split in ("train", "test"):
        for stage in STAGES:
            for name, label in CLASS_LABELS.items():
                group = [row for row in image_rows if (
                    row["split"] == split and row["stage"] == stage
                    and row["class_name"] == name
                )]
                if not group:
                    continue
                means = np.array([row["mean"] for row in group])
                stds = np.array([row["std"] for row in group])
                weights = np.array([row["pixel_count"] for row in group], dtype=float)
                pooled_mean = float(np.average(means, weights=weights))
                variance = float(np.average(
                    stds ** 2 + (means - pooled_mean) ** 2, weights=weights
                ))
                result.append({
                    "split": split, "analysis_role": group[0]["analysis_role"],
                    "stage": stage, "class_name": name, "label": label,
                    "image_count": len(group), "pixel_count": int(weights.sum()),
                    "pooled_pixel_mean": pooled_mean,
                    "pooled_pixel_std": float(np.sqrt(max(variance, 0.0))),
                    "minimum": min(row["minimum"] for row in group),
                    "maximum": max(row["maximum"] for row in group),
                    "mean_image_mean": float(means.mean()),
                    "sd_image_mean_ddof1": float(means.std(ddof=1)) if len(group) > 1 else None,
                    "median_image_mean": float(np.median(means)),
                    "iqr_image_mean": float(np.percentile(means, 75) - np.percentile(means, 25)),
                    "mean_within_image_std": float(stds.mean()),
                })
    return result


def compare_classes(training_rows: list[dict]) -> list[dict]:
    """Make descriptive image-level comparisons using training data only."""
    if not training_rows or any(row["split"] != "train" for row in training_rows):
        raise ValueError("Class comparisons require training data only.")
    result = []
    for stage in STAGES:
        for metric in METRICS:
            values = {
                name: np.array([row[metric] for row in training_rows if (
                    row["stage"] == stage and row["class_name"] == name
                )]) for name in CLASS_LABELS
            }
            a, b = values["def_front"], values["ok_front"]
            if not a.size or not b.size:
                raise ValueError("Both training classes are required at each stage.")
            result.append({
                "split": "train", "analysis_role": "training_analysis",
                "stage": stage, "metric": metric,
                "defective_count": a.size, "ok_count": b.size,
                "defective_mean": float(a.mean()), "ok_mean": float(b.mean()),
                "mean_difference_defective_minus_ok": float(a.mean() - b.mean()),
                "median_difference_defective_minus_ok": float(np.median(a) - np.median(b)),
                "cliffs_delta": cliffs_delta(a, b),
            })
    return result


def _inputs(root: Path, expected_counts: dict) -> tuple[list, list, list]:
    train = read_manifest(root / "data/splits/train.csv", "train", root)
    test = read_manifest(root / "data/splits/test.csv", "test", root)
    primary = collect_primary_records(root / "data/raw/casting_512x512")
    train_paths = {row["image_path"] for row in train}
    test_paths = {row["image_path"] for row in test}
    primary_paths = {record.path.relative_to(root).as_posix() for record in primary}
    if train_paths & test_paths or train_paths | test_paths != primary_paths:
        raise ValueError("Saved manifests overlap or do not cover the primary set.")
    if {row["image_id"] for row in train} & {row["image_id"] for row in test}:
        raise ValueError("Training and test identities overlap.")
    if dict(Counter(row["class_name"] for row in train + test)) != expected_counts:
        raise ValueError("Primary class counts differ from the expected dataset.")
    return train, test, primary


def _snapshot(root: Path, primary: list) -> tuple[dict, dict]:
    hashes = {
        record.path.relative_to(root).as_posix(): hashlib.sha256(
            record.path.read_bytes()
        ).hexdigest() for record in primary
    }
    manifests = {
        name: hashlib.sha256((root / "data/splits" / name).read_bytes()).hexdigest()
        for name in ("train.csv", "test.csv")
    }
    return hashes, manifests


def run_eda(
    project_root: Path = PROJECT_ROOT,
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    seed: int = 42, samples_per_class: int = 3,
    include_test_descriptive: bool = False, results_directory: Path | None = None,
    expected_counts: dict | None = None,
) -> dict:
    """Produce real EDA tables and figures without fitting or changing inputs."""
    height, width = validate_size(size)
    root = Path(project_root).resolve()
    destination = Path(results_directory or root / "results").resolve()
    if not destination.is_relative_to((root / "results").resolve()):
        raise ValueError("EDA outputs must stay under the project's results directory.")
    expected = EXPECTED_PRIMARY_COUNTS if expected_counts is None else expected_counts
    train, test, primary = _inputs(root, expected)
    before = _snapshot(root, primary)
    selected = select_samples(train, samples_per_class, seed)
    selected_paths = {row["image_path"] for row in selected}
    images, validation, sample_arrays, histogram_totals = [], [], {}, {}
    processing = train + test if include_test_descriptive else train
    for record in processing:
        original = load_grayscale(root / record["image_path"])
        if original.shape != DEFAULT_IMAGE_SIZE or original.dtype != np.uint8:
            raise ValueError(f"Unexpected primary image structure: {record['image_path']}")
        resized = resize_grayscale(original, (height, width))
        normalized = min_max_normalize(resized)
        role = "training_analysis" if record["split"] == "train" else "test_descriptive_only"
        arrays = dict(zip(STAGES, (original, resized, normalized)))
        sample = {**record, **arrays, "statistics": {}, "histograms": {}}
        for stage, image in arrays.items():
            stats = image_statistics(image)
            images.append({**record, "analysis_role": role, "stage": stage, **stats})
            counts, edges = intensity_histogram(image, normalized=stage == "normalized")
            key = (record["split"], record["class_name"], stage)
            if key not in histogram_totals:
                histogram_totals[key] = (counts.copy(), edges)
            else:
                histogram_totals[key][0][:] += counts
            sample["statistics"][stage] = stats
            sample["histograms"][stage] = (counts, edges)
        finite = bool(np.isfinite(normalized).all())
        minimum, maximum = float(normalized.min()), float(normalized.max())
        constant = bool(resized.min() == resized.max())
        bounds = 0.0 <= minimum <= maximum <= 1.0
        passed = bool(
            finite and bounds and normalized.dtype == np.float32
            and normalized.shape == (height, width) and minimum == 0.0
            and maximum == (0.0 if constant else 1.0)
        )
        validation.append({
            "image_path": record["image_path"], "image_id": record["image_id"],
            "class_name": record["class_name"], "split": record["split"],
            "analysis_role": role, "output_height": height, "output_width": width,
            "output_dtype": str(normalized.dtype), "finite": finite,
            "bounds_valid": bounds, "normalized_minimum": minimum,
            "normalized_maximum": maximum,
            "original_constant": bool(original.min() == original.max()),
            "resized_constant": constant, "passed": passed,
        })
        if not passed:
            raise ValueError(f"Preprocessing validation failed: {record['image_path']}")
        if record["image_path"] in selected_paths:
            sample_arrays[record["image_path"]] = sample

    histogram_rows = []
    for (split, name, stage), (counts, edges) in sorted(histogram_totals.items()):
        for index, count in enumerate(counts):
            histogram_rows.append({
                "split": split,
                "analysis_role": (
                    "training_analysis" if split == "train"
                    else "test_descriptive_only"
                ),
                "class_name": name, "label": CLASS_LABELS[name], "stage": stage,
                "bin_index": index, "bin_left": float(edges[index]),
                "bin_right": float(edges[index + 1]), "pixel_count": int(count),
                "probability": float(count / counts.sum()),
            })
    training_stats = [row for row in images if row["split"] == "train"]
    comparisons = compare_classes(training_stats)
    class_stats = summarize_classes(images)
    distribution = class_distribution(train, test)
    samples = [
        {**sample_arrays[row["image_path"]], "sample_rank": row["sample_rank"]}
        for row in selected
    ]
    current_primary = collect_primary_records(root / "data/raw/casting_512x512")
    if _snapshot(root, current_primary) != before:
        raise ValueError("Raw images or saved manifests changed during EDA.")

    figures = output_directory(destination / "figures", root)
    tables = output_directory(destination / "tables", root)
    with visualization.publication_style():
        visualization.plot_class_distribution(distribution, figures / FIGURE_NAMES[0])
        visualization.plot_training_samples(samples, figures / FIGURE_NAMES[1], seed)
        visualization.plot_class_histograms(
            [row for row in histogram_rows if row["split"] == "train"],
            figures / FIGURE_NAMES[2],
        )
        visualization.plot_image_statistics(training_stats, figures / FIGURE_NAMES[3])
        visualization.plot_preprocessing_comparison(samples, figures / FIGURE_NAMES[4])
        visualization.plot_preprocessing_histograms(samples, figures / FIGURE_NAMES[5])
        visualization.plot_class_comparisons(comparisons, figures / FIGURE_NAMES[6])

    write_csv(tables / "eda_class_distribution.csv", distribution, list(distribution[0]))
    write_csv(tables / "eda_image_statistics.csv", images, IMAGE_FIELDS)
    write_csv(tables / "eda_class_statistics.csv", class_stats, list(class_stats[0]))
    write_csv(tables / "eda_class_comparisons.csv", comparisons, list(comparisons[0]))
    write_csv(tables / "eda_intensity_histograms.csv", histogram_rows, list(histogram_rows[0]))
    write_csv(tables / "eda_selected_samples.csv", selected, [
        "image_path", "image_id", "class_name", "label", "split", "sample_rank", "seed",
    ])
    write_csv(tables / "eda_preprocessing_validation.csv", validation, VALIDATION_FIELDS)
    summary = {
        "stage": 2, "seed": seed, "resize_height": height, "resize_width": width,
        "processing_height": height, "processing_width": width,
        "native_height": DEFAULT_IMAGE_SIZE[0],
        "native_width": DEFAULT_IMAGE_SIZE[1],
        "resizing_applied": (height, width) != DEFAULT_IMAGE_SIZE,
        "samples_per_class": samples_per_class, "histogram_bins": 256,
        "normalization": (
            "Per-image Min-Max; native dimensions preserved unless an "
            "explicit different size is requested; constants map to zero"
        ),
        "output_dtype": "float32", "output_range": [0.0, 1.0],
        "interpolation": (
            "None; native dimensions preserved"
            if (height, width) == DEFAULT_IMAGE_SIZE
            else "Area for downsampling both dimensions; otherwise bilinear"
        ),
        "primary_images": len(primary), "train_images": len(train), "test_images": len(test),
        "processed_images": len(processing), "training_images_analyzed": len(train),
        "test_images_described": len(test) if include_test_descriptive else 0,
        "test_pixel_statistics_used_in_comparisons_or_figures": False,
        "preprocessing_parameters_fitted_across_images": False,
        "preprocessing_validation_passed": all(row["passed"] for row in validation),
        "constant_original_images": sum(row["original_constant"] for row in validation),
        "constant_resized_images": sum(row["resized_constant"] for row in validation),
        "raw_images_unchanged": True, "manifests_unchanged": True,
        "source_inventory_sha256": inventory_digest([
            {"image_path": path, "sha256": digest} for path, digest in before[0].items()
        ]),
        "manifest_sha256": before[1],
        "figure_paths": [(figures / name).relative_to(root).as_posix() for name in FIGURE_NAMES],
        "selected_training_images": [row["image_path"] for row in selected],
        "python_version": sys.version.split()[0],
        "versions": {name: version(name) for name in ("numpy", "opencv-python", "matplotlib")},
        "warnings": [
            "Class comparisons are descriptive and use image-level "
            "observations, not independent pixels.",
            "Unknown physical-part relationships and Stage 1 "
            "provisional-split limitations remain.",
            "Default processing preserves native 512 x 512 dimensions; "
            "a different processing size must be requested explicitly.",
            "Test descriptors, if requested, must not inform feature or model selection.",
            "No HOG, GLCM, model training, or hyperparameter optimization was run.",
        ],
    }
    write_json(tables / "eda_run_summary.json", summary)
    _write_report(tables / "eda_report.md", summary, distribution, class_stats)
    return summary


def _write_report(path, summary, distribution, class_stats):
    training = [row for row in class_stats if row["split"] == "train"]
    lines = [
        "# Stage 2: Exploratory Data Analysis and Preprocessing", "",
        f"Primary images: {summary['primary_images']}; training: {summary['train_images']}; "
        f"test: {summary['test_images']}.",
        f"Images processed: {summary['processed_images']}; test descriptive images: "
        f"{summary['test_images_described']}.", "",
        "## Scope and methodology", "",
        "All image examples, intensity comparison figures, and class effect sizes "
        "use training images only. The class-count plot uses manifest labels. "
        "Optional test pixel summaries are marked test_descriptive_only and "
        "never enter class comparisons or training figures. No preprocessing "
        "configuration was selected using test descriptors or model outcomes.", "",
        "Grayscale decoding produces uint8 single-channel arrays in [0, 255]; "
        "it is not intensity normalization. The float32 processing input keeps "
        "native 512 x 512 dimensions by default and is not resampled. Only an "
        "explicit different target invokes resizing: area interpolation when "
        "both dimensions shrink, otherwise bilinear interpolation. Constant "
        "inputs are preserved exactly to avoid amplifying interpolation noise. "
        "The legacy CSV stage key 'resized' denotes this processing input, "
        "even when no spatial resampling is applied.", "",
        f"Configured size: {summary['resize_height']} x {summary['resize_width']}. "
        f"Spatial resizing applied: {summary['resizing_applied']}. "
        "For each processing input r, m = min(r), M = max(r), and "
        "z = (r - m) / (M - m) when M > m; otherwise z = 0. "
        "Normalization preserves r's dimensions. The output is float32 in "
        "[0, 1]. These extrema belong to that image "
        "alone; there is no dataset-fitted scaler or transfer of test statistics.", "",
        "Per-image pixel SD uses ddof=0. Pooled class SD includes within-image "
        "and between-image variation and is distinct from the average of "
        "within-image SDs. Between-image mean SD uses ddof=1. Histograms use "
        "256 bins and probabilities summing to one; images contribute equal "
        "weight within each stage because stage dimensions are fixed.", "",
        "Class mean/median differences and Cliff's delta use one observation "
        "per image. Delta = P(Defective > OK) - P(Defective < OK); ties contribute "
        "zero. Values are descriptive, with no p-values, confidence intervals, "
        "or claims of independent physical components.", "",
        "## Manifest class distribution", "",
        "| Subset | Class | Images | Percentage |", "| --- | --- | ---: | ---: |",
        *[
            f"| {row['split']} | {row['class_name']} | "
            f"{row['image_count']} | {row['percentage']:.4f}% |"
            for row in distribution
        ], "",
        "## Training pixel summaries", "",
        "| Stage | Class | Images | Pooled mean | Pooled SD | Minimum | Maximum |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *[f"| {row['stage']} | {row['class_name']} | {row['image_count']} | "
          f"{row['pooled_pixel_mean']:.6f} | {row['pooled_pixel_std']:.6f} | "
          f"{row['minimum']:.6f} | {row['maximum']:.6f} |" for row in training], "",
        "See eda_class_comparisons.csv for image-level differences and effects; "
        "eda_image_statistics.csv and eda_intensity_histograms.csv contain the "
        "underlying computed summaries. Test rows, when present, are separate.", "",
        "## Reproducibility and validation", "",
        f"Sample seed: {summary['seed']}; samples per class: {summary['samples_per_class']}. "
        "Selection uses sorted training lists and a local NumPy generator; "
        "sample identities are stored in eda_selected_samples.csv.",
        f"All {summary['processed_images']} preprocessing checks passed: "
        f"{summary['preprocessing_validation_passed']}. "
        f"Constant original images: {summary['constant_original_images']}; "
        f"constant resized images: {summary['constant_resized_images']}.",
        "Raw membership/content and manifest SHA-256 fingerprints were checked "
        "before/after processing. The saved split files were not regenerated. "
        "PNG figures are exported at 300 DPI with fixed display ranges for "
        "original/resized [0, 255] and normalized [0, 1] images.", "",
        "## Remaining limitations", "",
        *[f"- {warning}" for warning in summary["warnings"]], "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", nargs=2, type=int, default=DEFAULT_IMAGE_SIZE,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional target; default 512 x 512 preserves native dimensions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--include-test-descriptive", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args(argv)
    try:
        summary = run_eda(
            size=tuple(args.size), seed=args.seed,
            samples_per_class=args.samples_per_class,
            include_test_descriptive=args.include_test_descriptive,
            results_directory=args.output_dir,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"Stage 2 EDA failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
