"""Run the fixed Stage 4A baselines on the saved training folds."""

import argparse
from collections import Counter
import json
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.data_loader import PROJECT_ROOT
from src.evaluation import evaluate_predictions
from src.feature_pipeline import (
    file_sha256, load_cached_features, process_memory, read_csv, system_resources,
)
from src.model_training import build_baseline_classifier
from src.reporting import output_directory, write_csv, write_json


FEATURES = {
    "hog8": ("hog8",),
    "hog16": ("hog16",),
    "glcm": ("glcm",),
    "hog8_glcm": ("hog8", "glcm"),
    "hog16_glcm": ("hog16", "glcm"),
}
DIMENSIONS = {"hog8": 142884, "hog16": 34596, "glcm": 48}
MODELS = ("svm", "random_forest")
METRICS = (
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
    "false_negatives", "fit_seconds", "predict_seconds", "score_seconds",
    "peak_rss_mib",
)
MIB = 1024 ** 2
RESERVE_MIB = 384


def align_rows(reference, candidate, labels):
    """Join by unique image ID and verify identity, split, label, and path."""
    if len(candidate) != len(labels):
        raise ValueError("Cache labels and index have different lengths.")
    lookup = {row["image_id"]: (index, row) for index, row in enumerate(candidate)}
    ids = {row["image_id"] for row in reference}
    if len(ids) != len(reference) or len(lookup) != len(candidate) or ids != lookup.keys():
        raise ValueError("Feature indexes must contain the same unique image IDs.")
    order = []
    for row in reference:
        index, actual = lookup[row["image_id"]]
        if row["split"] != "train" or actual["split"] != "train":
            raise ValueError("Stage 4A requires training rows only.")
        if any(str(row[key]) != str(actual[key]) for key in ("image_path", "class_name", "label")):
            raise ValueError("Feature identity or label differs across caches.")
        if int(actual["label"]) != labels[index]:
            raise ValueError("Cached label differs from the feature index.")
        if int(actual["feature_row"]) != index:
            raise ValueError("Feature index row position is invalid.")
        order.append(index)
    return np.asarray(order, dtype=np.intp)


def saved_folds(rows, assignments):
    """Use the existing fold IDs without constructing new split assignments."""
    lookup = {row["image_id"]: row for row in assignments}
    if len(lookup) != len(assignments) or set(lookup) != {row["image_id"] for row in rows}:
        raise ValueError("Saved CV must cover exactly the training IDs.")
    fold_ids = []
    for row in rows:
        match = lookup[row["image_id"]]
        if row["split"] != "train" or match.get("split", "train") != "train":
            raise ValueError("Saved CV contains a non-training row.")
        if any(str(row[key]) != str(match[key]) for key in ("image_path", "class_name", "label")):
            raise ValueError("Saved CV identity or label differs from the training index.")
        fold = int(match["validation_fold"])
        if fold != int(row["validation_fold"]):
            raise ValueError("Feature index and saved CV fold IDs disagree.")
        fold_ids.append(fold)
    if set(fold_ids) != set(range(1, 6)):
        raise ValueError("Expected exactly five saved folds.")
    groups = {}
    for fold, row in zip(fold_ids, rows):
        group = lookup[row["image_id"]]["group_id"]
        if not group or groups.setdefault(group, fold) != fold:
            raise ValueError("A saved independence group spans validation folds.")
    labels = np.asarray([int(row["label"]) for row in rows])
    split = []
    for fold in range(1, 6):
        fit = np.flatnonzero(np.asarray(fold_ids) != fold)
        valid = np.flatnonzero(np.asarray(fold_ids) == fold)
        if set(labels[fit]) != {0, 1} or set(labels[valid]) != {0, 1}:
            raise ValueError("Every fit and validation fold must contain both classes.")
        split.append((fold, fit, valid))
    return split


def load_training_inputs(root):
    """Check Stage 3 provenance and open only training feature matrices."""
    summary = json.loads((root / "results/tables/feature_run_summary.json").read_text())
    if summary["provenance"]["configuration"]["image_shape"] != [512, 512]:
        raise ValueError("Stage 4A requires native-resolution caches.")
    reference = read_csv(root / "results/tables/feature_train_index.csv")
    if Counter(int(row["label"]) for row in reference) != {0: 415, 1: 625}:
        raise ValueError("Expected the saved 1,040-image training set.")
    fold_path = root / "results/tables/cv_folds.csv"
    folds = saved_folds(reference, read_csv(fold_path))
    layout = {}
    for method, dimension in DIMENSIONS.items():
        matrix, labels, rows = load_cached_features(root, method, "train")
        if matrix.shape != (len(reference), dimension) or matrix.dtype != np.float32:
            raise ValueError(f"Unexpected {method} training cache geometry.")
        entry, = (r for r in summary["caches"] if r["split"] == "train" and r["method"] == method)
        layout[method] = {
            "path": entry["path"], "sha256": entry["sha256"],
            "shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "order": align_rows(reference, rows, labels).tolist(),
        }
        del matrix
    provenance = {
        "train_manifest_sha256": file_sha256(root / "data/splits/train.csv"),
        "cv_folds_sha256": file_sha256(fold_path),
        "train_index_sha256": file_sha256(root / "results/tables/feature_train_index.csv"),
        "stage3_summary_sha256": file_sha256(root / "results/tables/feature_run_summary.json"),
        "source_inventory_sha256": summary["provenance"]["source_inventory_sha256"],
        "training_caches": {key: {k: v for k, v in layout[key].items() if k != "order"}
                            for key in layout},
    }
    return layout, folds, [int(row["label"]) for row in reference], provenance


def fold_matrix(root, components, indices, dtype):
    """Gather only requested rows, allocating one contiguous design per phase."""
    columns = sum(entry["shape"][1] for entry in components)
    result = np.empty((len(indices), columns), dtype=dtype)
    column = 0
    for entry in components:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to((root / "results/cache").resolve()) or not path.name.startswith("train_"):
            raise ValueError("Workers may only open training caches.")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        width = matrix.shape[1]
        rows = np.asarray(entry["order"], dtype=np.intp)[indices]
        for offset in range(0, len(rows), 16):
            batch = rows[offset:offset + 16]
            result[offset:offset + len(batch), column:column + width] = matrix[batch]
        column += width
        del matrix
    return result


def svm_predictions(classifier, scaled_validation):
    """Evaluate the fitted binary RBF decision function in one batched pass."""
    from sklearn.metrics.pairwise import rbf_kernel

    # _gamma is libsvm's resolved numeric value for gamma="scale".
    kernel = rbf_kernel(scaled_validation, classifier.support_vectors_, gamma=classifier._gamma)
    scores = kernel @ classifier.dual_coef_[0] + classifier.intercept_[0]
    labels = np.where(scores > 0, classifier.classes_[1], classifier.classes_[0])
    near_zero = np.flatnonzero(np.abs(scores) < 1e-8)
    if len(near_zero):
        rows = scaled_validation[near_zero]
        scores[near_zero] = classifier.decision_function(rows)
        labels[near_zero] = classifier.predict(rows)
    return labels, scores


def evaluate_fold(root, job):
    """Fit one fresh pipeline and evaluate only its saved validation rows."""
    from threadpoolctl import threadpool_limits

    fit = np.asarray(job["fit"], dtype=np.intp)
    valid = np.asarray(job["valid"], dtype=np.intp)
    if len(np.intersect1d(fit, valid)):
        raise ValueError("Fit and validation rows overlap.")
    labels = np.asarray(job["labels"], dtype=np.int8)
    model = job["model"]
    dtype = np.float64 if model == "svm" else np.float32
    estimator = build_baseline_classifier(model)
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        x_fit = fold_matrix(root, job["components"], fit, dtype)
        load_fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        estimator.fit(x_fit, labels[fit])
        fit_seconds = time.perf_counter() - started
        del x_fit
        started = time.perf_counter()
        x_valid = fold_matrix(root, job["components"], valid, dtype)
        load_valid_seconds = time.perf_counter() - started
        classifier = estimator.named_steps["classifier"]
        started = time.perf_counter()
        if model == "svm":
            x_valid = estimator.named_steps["scaler"].transform(x_valid)
            prediction, scores = svm_predictions(classifier, x_valid)
        else:
            prediction = classifier.predict(x_valid)
        predict_seconds = time.perf_counter() - started
        score_seconds = check_seconds = 0.0
        if model == "svm":
            started = time.perf_counter()
            sample = np.unique([0, len(x_valid) // 2, len(x_valid) - 1])
            np.testing.assert_allclose(
                scores[sample], classifier.decision_function(x_valid[sample]),
                rtol=1e-7, atol=1e-8,
            )
            np.testing.assert_array_equal(prediction[sample], classifier.predict(x_valid[sample]))
            check_seconds = time.perf_counter() - started
        else:
            started = time.perf_counter()
            scores = classifier.predict_proba(x_valid)[:, list(classifier.classes_).index(1)]
            score_seconds = time.perf_counter() - started
        metrics, confusion = evaluate_predictions(labels[valid], prediction, scores)
    tn, fp, fn, tp = map(int, confusion.ravel())
    return {
        **metrics, "true_negatives": tn, "false_positives": fp,
        "false_negatives": fn, "true_positives": tp,
        "fit_rows": len(fit), "validation_rows": len(valid),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "score_seconds": score_seconds, "load_fit_seconds": load_fit_seconds,
        "prediction_check_seconds": check_seconds,
        "load_validation_seconds": load_valid_seconds,
        "peak_rss_mib": process_memory()["peak_rss_bytes"] / MIB,
    }


def estimated_peak_mib(model, fit_rows, valid_rows, columns):
    """Budget data copies, fitted support vectors, and native overhead."""
    if model == "svm":
        # Dense float64 input + a worst-case support-vector copy, bounded scaler
        # scratch, the kernel cache, and Python/native library overhead.
        return max(2 * fit_rows, fit_rows + valid_rows) * columns * 8 / MIB + 384
    # During gathering the mapped source pages can be resident alongside the
    # dense float32 fit matrix. Allow the full cache plus the fit rows.
    return (2 * fit_rows + valid_rows) * columns * 4 / MIB + 256


def run_worker(root, job):
    """Isolate one fit and interrupt it if the system runs low on physical RAM."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    command = [sys.executable, "-B", "-m", "src.baseline_experiment", "--worker"]
    started = time.perf_counter()
    with subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        request = json.dumps(job)
        low_checks = 0
        available_mib = system_resources(root)["available_ram_bytes"] / MIB
        try:
            while True:
                try:
                    output, stderr = child.communicate(request, timeout=0.6)
                    break
                except subprocess.TimeoutExpired:
                    request = None
                    available = system_resources(root)["available_ram_bytes"] / MIB
                    available_mib = min(available_mib, available)
                    low_checks = low_checks + 1 if available < RESERVE_MIB else 0
                    if low_checks >= 2:
                        raise MemoryError(f"Physical memory fell below the {RESERVE_MIB} MiB reserve.")
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
    if child.returncode != 0:
        if "MemoryError" in stderr or "Unable to allocate" in stderr:
            raise MemoryError(stderr.strip().splitlines()[-1])
        raise RuntimeError(stderr.strip() or f"Worker exited with status {child.returncode}.")
    return {**json.loads(output), "worker_wall_seconds": time.perf_counter() - started,
            "minimum_available_mib": available_mib}


def summarize(fold_rows, combinations):
    summary = []
    for feature, model in combinations:
        rows = [row for row in fold_rows if (row["feature_set"], row["model"]) == (feature, model)]
        if len(rows) != 5:
            continue
        if {row["fold"] for row in rows} != set(range(1, 6)):
            raise ValueError("Expected one result for each saved fold.")
        result = {"feature_set": feature, "model": model, "folds": len(rows)}
        for metric in METRICS:
            values = np.array([row[metric] for row in rows])
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(finite.mean()) if finite.size else None
            result[f"{metric}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else None
        for key in ("true_negatives", "false_positives", "false_negatives", "true_positives"):
            result[f"{key}_total"] = sum(row[key] for row in rows)
        result["peak_rss_mib_max"] = max(row["peak_rss_mib"] for row in rows)
        result["minimum_available_mib"] = min(row["minimum_available_mib"] for row in rows)
        summary.append(result)
    return summary


def smoke_subset(indices, labels, per_class):
    """Use a small reproducible subset of an existing fold for resource checks."""
    rng = np.random.default_rng(42)
    return np.sort(np.concatenate([
        rng.choice(indices[labels[indices] == label], per_class, replace=False)
        for label in (0, 1)
    ]))


def run(root=PROJECT_ROOT):
    root = Path(root).resolve()
    started = time.perf_counter()
    layout, folds, labels, provenance = load_training_inputs(root)
    tables = output_directory(root / "results/tables", root)
    figures = output_directory(root / "results/figures", root)
    combinations = [(feature, model) for feature in FEATURES for model in MODELS]
    resources = system_resources(root)
    print(f"Training caches verified. Initial available RAM: {resources['available_ram_bytes'] / MIB:.0f} MiB.", flush=True)
    smoke_rows, results, stopped = [], [], []
    _, fit, valid = folds[0]
    smoke_fit = smoke_subset(fit, np.array(labels), 48)
    smoke_valid = smoke_subset(valid, np.array(labels), 16)
    smoke_fields = ["feature_set", "model", "columns", "estimated_full_peak_mib", "fit_rows",
                    "validation_rows", "fit_seconds", "predict_seconds", "worker_wall_seconds",
                    "peak_rss_mib", "minimum_available_mib"]
    for feature, model in combinations:
        components = [layout[method] for method in FEATURES[feature]]
        columns = sum(entry["shape"][1] for entry in components)
        estimate = estimated_peak_mib(model, len(fit), len(valid), columns)
        base = {"feature_set": feature, "model": model, "columns": columns,
                "estimated_full_peak_mib": estimate}
        payload = {"model": model, "labels": labels, "components": components}
        try:
            print(f"Smoke test {feature} / {model}", flush=True)
            smoke = run_worker(root, {**payload, "fit": smoke_fit.tolist(), "valid": smoke_valid.tolist()})
            smoke_rows.append({key: ({**base, **smoke})[key] for key in smoke_fields})
            write_csv(tables / "baseline_cv_smoke.csv", smoke_rows, smoke_fields)
            for fold, fit_indices, valid_indices in folds:
                available = system_resources(root)["available_ram_bytes"] / MIB
                if available < estimate + RESERVE_MIB:
                    raise MemoryError(f"Estimated peak {estimate:.0f} MiB plus {RESERVE_MIB} MiB reserve "
                                      f"exceeds currently available {available:.0f} MiB.")
                print(f"Fit {feature} / {model} / fold {fold}", flush=True)
                row = run_worker(root, {**payload, "fit": fit_indices.tolist(),
                                        "valid": valid_indices.tolist()})
                results.append({**base, "fold": fold, **row})
                write_csv(tables / "baseline_cv_folds.csv", results, list(results[0]))
                print(f"  fit {row['fit_seconds']:.2f}s; peak {row['peak_rss_mib']:.0f} MiB", flush=True)
        except MemoryError as error:
            stopped.append({**base, "reason": str(error)})
            print(f"Resource stop: {feature} / {model}: {error}", flush=True)
    _, _, _, after = load_training_inputs(root)
    if after != provenance:
        raise ValueError("Saved training inputs changed during the run.")
    summary = summarize(results, combinations)
    if not summary:
        raise RuntimeError(f"No complete five-fold configuration: {stopped}")
    write_csv(tables / "baseline_cv_summary.csv", summary, list(summary[0]))
    from src.baseline_figures import plot_baselines

    figure_paths = plot_baselines(summary, figures)
    metadata = {
        "stage": "4A", "scope": "training-only cross-validation", "backend": "CPU",
        "seed": 42, "training_rows": len(labels), "fold_rows": len(results),
        "completed_combinations": len(summary), "skipped_combinations": stopped,
        "folds": [{"fold": fold, "fit_rows": len(fit), "validation_rows": len(valid)}
                  for fold, fit, valid in folds],
        "feature_sets": FEATURES, "positive_label": 1, "sample_sd_ddof": 1,
        "models": {
            "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": None,
                    "scaler": "StandardScaler partial_fit batches of 16 within each fit fold",
                    "random_state": 42,
                    "prediction": "Batched fitted RBF decision function for labels and AUC scores",
                    "reference_check_rows_per_fold": 3},
            "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_split": 2,
                              "min_samples_leaf": 1, "class_weight": None, "random_state": 42,
                              "n_jobs": 1},
        },
        "resource_guard": {"reserve_mib": RESERVE_MIB, "parallel_fits": 1,
                           "numerical_threads": 1, "smoke_fit_rows": len(smoke_fit),
                           "smoke_validation_rows": len(smoke_valid)},
        "initial_resources": resources, "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance, "python": sys.version.split()[0],
        "versions": {name: version(name) for name in ("numpy", "scikit-learn", "matplotlib", "threadpoolctl")},
        "figures": [path.relative_to(root).as_posix() for path in figure_paths],
    }
    write_json(tables / "baseline_cv_run.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(evaluate_fold(PROJECT_ROOT, json.load(sys.stdin))))
    else:
        run()


if __name__ == "__main__":
    main()
