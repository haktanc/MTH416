"""Freeze the six Stage 4B pipelines and evaluate the saved holdout once."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from math import sqrt
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from src.baseline_experiment import (
    DIMENSIONS, FEATURES, MIB, MODELS, RESERVE_MIB, estimated_peak_mib,
    fold_matrix, run_worker,
)
from src.baseline_scaling import weight_glcm
from src.data_loader import PROJECT_ROOT, collect_primary_records, read_manifest
from src.evaluation import evaluate_predictions
from src.feature_pipeline import (
    configuration, file_sha256, load_cached_features, process_memory, read_csv,
    system_resources,
)
from src.model_training import build_baseline_classifier
from src.reporting import csv_bytes, write_csv, write_json
from src.tuning_experiment import FEATURE_SETS, load_training, select_candidates, verify_inputs


PRIMARY = "hog16_glcm_svm_07"
PROTOCOL = "results/tables/holdout_protocol.json"
LOG = "results/tables/holdout_execution.txt"
CACHE = "results/cache/features"
MODEL_NAMES = {"svm": "SVM", "random_forest": "Random Forest"}
FEATURE_NAMES = {"hog16": "HOG 16×16", "glcm": "GLCM", "hog16_glcm": "HOG 16×16 + GLCM"}
SEED = 42
COUNTS = {"train": {0: 415, 1: 625}, "test": {0: 104, 1: 156}}
CV_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")
PREDICTION_FIELDS = [
    "candidate", "feature_set", "model", "image_id", "image_path",
    "true_label", "predicted_label", "defective_score", "score_type",
]


def construct_pipeline(model, feature_set, parameters, weight):
    """Recreate the Stage 4B estimator and its ordered transformations."""
    from sklearn.preprocessing import FunctionTransformer

    required = {"C", "gamma"} if model == "svm" else {
        "max_depth", "max_features", "min_samples_leaf", "n_estimators",
    }
    if parameters.keys() != required:
        raise ValueError(f"Incomplete or unexpected tuned parameters for {model}.")
    pipeline = build_baseline_classifier(model, random_state=SEED)
    pipeline.named_steps["classifier"].set_params(**parameters)
    if weight != 1.0:
        if (model, feature_set) != ("svm", "hog16_glcm") or weight != sqrt(34596 / 48):
            raise ValueError("Unexpected Stage 4B feature weight.")
        pipeline.steps.insert(-1, ("glcm_weight", FunctionTransformer(
            weight_glcm, kw_args={"factor": weight},
        )))
    return pipeline


def describe_pipeline(pipeline):
    """Serialize every resolved classifier/scaler parameter and transform order."""
    steps = []
    for name, estimator in pipeline.steps:
        parameters = estimator.get_params(deep=False)
        if name == "glcm_weight":
            parameters["func"] = "src.baseline_scaling.weight_glcm"
        steps.append({"step": name, "implementation": type(estimator).__name__,
                      "parameters": parameters})
    return steps


def selected_configurations(root, tuning):
    """Reconcile the six saved selections with the plan and five-fold records."""
    tables = root / "results/tables"
    plan = {row["candidate"]: row for row in tuning["plan"]}
    candidates = read_csv(tables / "tuning_cv_candidates.csv")
    folds = read_csv(tables / "tuning_cv_folds.csv")
    if (len(plan) != 38 or Counter(row["candidate"] for row in candidates) != Counter(plan.keys())
            or Counter((row["candidate"], int(row["fold"])) for row in folds)
            != Counter((candidate, fold) for candidate in plan for fold in range(1, 6))):
        raise ValueError("Stage 4B plan and records must cover 38 five-fold candidates.")
    for row in candidates:
        row["f1_mean"] = float(row["f1_mean"])
        row["false_negatives_total"] = int(row["false_negatives_total"])
    winners = {row["candidate"] for row in select_candidates(candidates)}
    saved = read_csv(tables / "tuning_cv_selected.csv")
    if len(saved) != 6 or {row["candidate"] for row in saved} != winners:
        raise ValueError("Stage 4B selected records disagree with the saved candidate rankings.")
    baseline = {(r["feature_set"], r["model"]): r
                for r in read_csv(tables / "baseline_cv_summary.csv")}
    candidate_lookup = {row["candidate"]: row for row in candidates}
    configurations = {}
    for row in saved:
        candidate, feature, model = row["candidate"], row["feature_set"], row["model"]
        parameters = json.loads(row["parameters"])
        weight = float(row["glcm_weight"])
        columns = sum(DIMENSIONS[method] for method in FEATURES[feature])
        source = row["source"]
        expected = {"candidate": candidate, "feature_set": feature, "model": model,
                    "parameters": parameters, "glcm_weight": weight, "columns": columns,
                    "baseline": source == "stage4a"}
        if plan[candidate] != expected or source not in {"stage4a", "stage4b"}:
            raise ValueError(f"Saved plan disagrees with selection {candidate}.")
        records = [candidate_lookup[candidate], *(r for r in folds if r["candidate"] == candidate)]
        for record in records:
            if (any(record[k] != row[k] for k in ("feature_set", "model", "source"))
                    or json.loads(record["parameters"]) != parameters
                    or float(record["glcm_weight"]) != weight
                    or int(record["columns"]) != columns):
                raise ValueError(f"Stage 4B records disagree on {candidate}.")
        cv = {}
        for metric in CV_METRICS:
            mean = f"{metric}_mean"
            if float(row[mean]) != float(candidate_lookup[candidate][mean]):
                raise ValueError(f"Stage 4B selected mean {mean} disagrees for {candidate}.")
            cv[f"stage4b_{metric}"] = float(row[mean])
            baseline_value = float(baseline[feature, model][mean])
            if baseline_value != float(row[f"baseline_{mean}"]):
                raise ValueError(f"Stage 4A baseline mean {mean} disagrees for {candidate}.")
            cv[f"stage4a_{metric}"] = baseline_value
        pipeline = construct_pipeline(model, feature, parameters, weight)
        configurations[feature, model] = {
            "candidate": candidate, "name": f"{FEATURE_NAMES[feature]} + {MODEL_NAMES[model]}",
            "feature_set": feature, "model": model, "columns": columns,
            "feature_order": list(FEATURES[feature]),
            "input_dtype": "float64" if model == "svm" else "float32",
            "selected_parameters": parameters, "glcm_weight": weight,
            "selected_source": source, "pipeline": describe_pipeline(pipeline),
            "training_cv": cv,
        }
    models = [configurations[feature, model] for feature in FEATURE_SETS for model in MODELS]
    primary, = (row for row in models if row["candidate"] == PRIMARY)
    if (primary["feature_set"], primary["model"], primary["selected_parameters"],
            primary["glcm_weight"], primary["pipeline"][-1]["parameters"]["kernel"]) != (
            "hog16_glcm", "svm", {"C": 3, "gamma": "scale"}, sqrt(34596 / 48), "rbf"):
        raise ValueError("The saved primary model differs from the predetermined protocol.")
    return models


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_from_training(root):
    """Build an auditable protocol using only source code and saved training inputs."""
    tables = root / "results/tables"
    tuning = read_json(tables / "tuning_cv_run.json")
    feature = read_json(tables / "feature_run_summary.json")
    stage1 = read_json(tables / "stage1_validation.json")
    if (tuning["failures"] or tuning["scope"] != "training-only tuning on saved folds"
            or (tuning["seed"], tuning["training_rows"], tuning["completed_candidates"],
                tuning["fold_records"]) != (SEED, 1040, 38, 190)
            or tuning["python"] != platform.python_version()
            or any(version(package) != saved for package, saved in tuning["versions"].items())):
        raise ValueError("Stage 4B completion, train-only scope, or runtime provenance differs.")
    if (feature["status"] != "complete"
            or feature["provenance"]["configuration"] != json.loads(json.dumps(configuration()))
            or not stage1["validation_passed"] or not stage1["checks"]["no_exact_content_crosses_holdout"]):
        raise ValueError("Saved feature or split validation does not match the protocol.")
    cache_manifest_hashes = feature["provenance"]["input_sha256"]
    for name, digest in stage1["manifest_sha256"].items():
        if cache_manifest_hashes[f"data/splits/{name}"] != digest:
            raise ValueError("Stage 1 and Stage 3 disagree on a manifest fingerprint.")
    verify_inputs(root, tuning["implementation"])
    layout, _, train_rows, hashes = load_training(root)
    if hashes != tuning["training_hashes"]:
        raise ValueError("Training files differ from the Stage 4B inputs.")
    models = selected_configurations(root, tuning)
    stage5_sources = ("src/holdout_experiment.py", "src/holdout_figures.py")
    files = [*(f"results/tables/tuning_cv_{stem}" for stem in
               ("run.json", "selected.csv", "candidates.csv", "folds.csv")),
             "results/tables/stage1_validation.json", *stage5_sources]
    frozen = {
        "stage": 5, "source": "Saved Stage 4B training-only CV selections",
        "dataset": "data/raw/casting_512x512", "label_encoding": {"OK": 0, "Defective": 1},
        "partition_rows": {"train": 1040, "test": 260}, "seed": SEED,
        "primary": PRIMARY, "models": models, "feature_configuration": configuration(),
        "expected_manifest_sha256": cache_manifest_hashes,
        "training_sha256": hashes,
        "implementation_sha256": tuning["implementation"],
        "frozen_sha256": {path: file_sha256(root / path) for path in files},
        "python": tuning["python"], "versions": tuning["versions"],
        "execution": {"backend": "CPU", "parallel_fits": 1, "numerical_threads": 1},
        "score_methods": {"svm": "decision_function", "random_forest": "predict_proba[:, class 1]"},
        "decision_rule": "Estimator predict with the default decision rule",
        "uncertainty": "Nominal 95% Wilson intervals for accuracy, recall and specificity",
        "error_examples": "Up to three primary-model examples per outcome, seed 42",
    }
    return json.loads(json.dumps(frozen)), layout, train_rows


def require_unused(root):
    """Refuse to open holdout data when a Stage 5 attempt or result exists."""
    previous = sorted(path for subdir in ("tables", "figures", "cache", "models")
                      for path in (root / "results" / subdir).glob("holdout_*")
                      if path != root / PROTOCOL)
    if previous:
        raise FileExistsError(
            "Existing Stage 5 artifacts require provenance review before another run: "
            + ", ".join(path.name for path in previous)
        )


def freeze(root):
    require_unused(root)
    frozen, _, _ = protocol_from_training(root)
    with (root / PROTOCOL).open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return frozen


def validate_holdout(root, frozen, train):
    """Verify the unchanged holdout cache, split coverage, and label alignment."""
    verify_inputs(root, frozen["expected_manifest_sha256"])
    test = read_manifest(root / "data/splits/test.csv", "test", root)
    if Counter(row["label"] for row in test) != COUNTS["test"]:
        raise ValueError("Unexpected holdout class counts.")
    all_rows = train + test
    paths = {r["image_path"] for r in all_rows}
    primary = collect_primary_records(root / frozen["dataset"])
    if (len(paths) != len(all_rows) or len({r["image_id"] for r in all_rows}) != len(all_rows)
            or paths != {r.path.relative_to(root).as_posix() for r in primary}):
        raise ValueError("Train and holdout must be disjoint and cover the primary set.")
    stage3 = read_json(root / "results/tables/feature_run_summary.json")
    hashes = {}
    for method in ("hog16", "glcm"):
        item, = (row for row in stage3["caches"] if row["split"] == "test" and row["method"] == method)
        expected_path = f"{CACHE}/test_{method}.npy"
        if item["path"] != expected_path or (root / expected_path).resolve() != root / expected_path:
            raise ValueError(f"Unexpected holdout cache path for {method}.")
        matrix, labels, rows = load_cached_features(root, method, "test")
        if (matrix.shape != (260, DIMENSIONS[method]) or not np.isfinite(matrix).all()
                or not np.array_equal(labels, [row["label"] for row in test])
                or any(row["validation_fold"].strip() for row in rows)):
            raise ValueError(f"Unexpected holdout cache data for {method}.")
        del matrix
        hashes.update({expected_path: item["sha256"],
                       f"{CACHE}/test_labels.npy": item["labels_sha256"],
                       "results/tables/feature_test_index.csv": item["index_sha256"]})
    return test, hashes


def holdout_matrix(root, methods, count, dtype):
    """Materialize the holdout once in the frozen component order."""
    matrix = np.empty((count, sum(DIMENSIONS[method] for method in methods)), dtype=dtype)
    offset = 0
    for method in methods:
        cache = np.load(root / CACHE / f"test_{method}.npy", mmap_mode="r", allow_pickle=False)
        width = DIMENSIONS[method]
        matrix[:, offset:offset + width] = cache
        offset += width
    return matrix


def infer(pipeline, features, model):
    """Use the estimator's default labels and its actual class-1 score API."""
    if list(pipeline.classes_) != [0, 1]:
        raise ValueError("Expected OK and Defective in classifier class order.")
    classifier = pipeline.named_steps["classifier"]
    started = perf_counter()
    if model == "svm":
        # StandardScaler(copy=False) may mutate the supplied matrix.
        features = pipeline[:-1].transform(features)
    predictions = classifier.predict(features)
    predict_seconds = perf_counter() - started
    started = perf_counter()
    if model == "svm":
        scores = classifier.decision_function(features)
    else:
        scores = classifier.predict_proba(features)[:, list(classifier.classes_).index(1)]
    return predictions, scores, predict_seconds, perf_counter() - started


def evaluate_model(root, job):
    """Fit in a fresh worker with no holdout labels in its request."""
    from threadpoolctl import threadpool_limits

    model = job["configuration"]
    train_labels = np.asarray(job["train_labels"], dtype=np.int8)
    estimator = construct_pipeline(model["model"], model["feature_set"],
                                   model["selected_parameters"], model["glcm_weight"])
    if describe_pipeline(estimator) != model["pipeline"]:
        raise ValueError("Worker pipeline differs from the frozen configuration.")
    dtype = np.dtype(model["input_dtype"])
    with threadpool_limits(limits=1):
        train = fold_matrix(root, job["training_components"], np.arange(len(train_labels)), dtype)
        started = perf_counter()
        estimator.fit(train, train_labels)
        fit_seconds = perf_counter() - started
        del train
        test = holdout_matrix(root, model["feature_order"], job["test_count"], dtype)
        predictions, scores, predict_seconds, score_seconds = infer(estimator, test, model["model"])
    return {
        "predicted_labels": predictions.tolist(), "scores": scores.tolist(),
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
        "score_seconds": score_seconds, "inference_seconds": predict_seconds + score_seconds,
        "peak_rss_mib": process_memory()["peak_rss_bytes"] / MIB,
    }


def wilson(successes, total):
    """Nominal 95% binomial interval for a proportion."""
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    radius = z * sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def score_holdout(truth, predictions, scores):
    from sklearn.metrics import average_precision_score

    metrics, confusion = evaluate_predictions(truth, predictions, scores)
    tn, fp, fn, tp = map(int, confusion.ravel())
    metrics.update(tn=tn, fp=fp, fn=fn, tp=tp, specificity=tn / (tn + fp),
                   average_precision=average_precision_score(truth, scores, pos_label=1))
    for metric, numerator, denominator in (
        ("accuracy", tn + tp, len(truth)), ("recall", tp, tp + fn), ("specificity", tn, tn + fp),
    ):
        low, high = wilson(numerator, denominator)
        metrics.update({f"{metric}_ci95_low": low, f"{metric}_ci95_high": high})
    return metrics, confusion


def error_tables(test, predictions, configurations):
    """Record overlapping errors and reproducible primary-model sample IDs."""
    rng = np.random.default_rng(SEED)
    matrix = np.asarray([row["predicted_label"] for row in predictions]).reshape(-1, len(test))
    truth = np.asarray([row["label"] for row in test])
    errors = matrix != truth
    shared = []
    for index, row in enumerate(test):
        wrong = np.flatnonzero(errors[:, index])
        if len(wrong) >= 2:
            shared.append({"image_id": row["image_id"], "image_path": row["image_path"],
                           "true_label": row["label"], "error": "FN" if row["label"] else "FP",
                           "models_in_error": len(wrong),
                           "candidates_in_error": "|".join(configurations[i]["candidate"] for i in wrong)})
    shared.sort(key=lambda row: (-row["models_in_error"], row["image_id"]))
    primary, = (i for i, row in enumerate(configurations) if row["candidate"] == PRIMARY)
    primary_rows = predictions[primary * len(test):(primary + 1) * len(test)]
    examples = []
    for category, actual, predicted in (("False negatives", 1, 0), ("False positives", 0, 1),
                                        ("Correct defective", 1, 1), ("Correct OK", 0, 0)):
        candidates = np.flatnonzero((truth == actual) & (matrix[primary] == predicted))
        for index in rng.choice(candidates, size=min(3, len(candidates)), replace=False):
            examples.append({"outcome": category, **primary_rows[index]})
    return shared, examples


def run(root):
    from src.holdout_figures import plot_holdout

    require_unused(root)
    frozen = read_json(root / PROTOCOL)
    confirmed, layout, train = protocol_from_training(root)
    if frozen != confirmed:
        raise ValueError("Current training provenance or model configuration differs from the frozen protocol.")
    models = frozen["models"]
    resources = system_resources(root)
    budget = max(estimated_peak_mib(row["model"], 1040, 260, row["columns"]) for row in models)
    if resources["available_ram_bytes"] / MIB < budget + RESERVE_MIB:
        raise MemoryError("Insufficient available physical RAM for the frozen evaluation.")

    started = perf_counter()
    with (root / LOG).open("x", encoding="utf-8") as log:
        def record(message):
            line = f"{datetime.now(timezone.utc).isoformat()} {message}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        record(f"Started frozen protocol SHA-256 {file_sha256(root / PROTOCOL)}")
        try:
            test, test_hashes = validate_holdout(root, frozen, train)
            record("Verified split identity, disjoint membership, and both holdout cache alignments.")
            truth = [row["label"] for row in test]
            predictions, metrics, counts = [], [], []
            with (root / "results/tables/holdout_predictions.csv").open("xb") as stream:
                stream.write(csv_bytes([], PREDICTION_FIELDS))
                header_size = stream.tell()
                for model in models:
                    candidate = model["candidate"]
                    record(f"Fitting {candidate} on all 1,040 training rows.")
                    result = run_worker(root, {
                        "configuration": model, "train_labels": [row["label"] for row in train],
                        "training_components": [layout[method] for method in model["feature_order"]],
                        "test_count": len(test),
                    }, module="src.holdout_experiment")
                    labels, scores = result.pop("predicted_labels"), result.pop("scores")
                    values, confusion = score_holdout(truth, labels, scores)
                    identity = {k: model[k] for k in ("candidate", "feature_set", "model", "name", "columns")}
                    metrics.append({**identity, "primary": candidate == PRIMARY,
                                    **values, **result, **model["training_cv"]})
                    counts.extend({**identity, "actual_label": actual, "predicted_label": predicted,
                                   "count": int(confusion[actual, predicted]),
                                   "actual_row_percent": 100 * confusion[actual, predicted] / sum(confusion[actual])}
                                  for actual in (0, 1) for predicted in (0, 1))
                    score_type = frozen["score_methods"][model["model"]]
                    batch = [{"candidate": candidate, "feature_set": model["feature_set"],
                              "model": model["model"], "image_id": row["image_id"],
                              "image_path": row["image_path"], "true_label": row["label"],
                              "predicted_label": label, "defective_score": score, "score_type": score_type}
                             for row, label, score in zip(test, labels, scores, strict=True)]
                    stream.write(csv_bytes(batch, PREDICTION_FIELDS)[header_size:])
                    stream.flush()
                    predictions.extend(batch)
                    record(f"Saved {len(batch)} predictions for {candidate}.")
            shared, examples = error_tables(test, predictions, models)
            tables = root / "results/tables"
            write_csv(tables / "holdout_metrics.csv", metrics, list(metrics[0]))
            write_csv(tables / "holdout_confusion.csv", counts, list(counts[0]))
            write_csv(tables / "holdout_shared_errors.csv", shared,
                      ["image_id", "image_path", "true_label", "error", "models_in_error", "candidates_in_error"])
            write_csv(tables / "holdout_examples.csv", examples, ["outcome", *PREDICTION_FIELDS])
            figures = plot_holdout(root, models, metrics, predictions, examples)
            verify_inputs(root, frozen["expected_manifest_sha256"] | frozen["training_sha256"]
                          | test_hashes | frozen["implementation_sha256"] | frozen["frozen_sha256"])
            write_json(tables / "holdout_run.json", {
                "status": "complete", "stage": 5, "protocol_sha256": file_sha256(root / PROTOCOL),
                "scope": "Independent image-level holdout evaluation of six frozen configurations",
                "primary": PRIMARY, "seed": SEED, "resources": resources,
                "train_rows": len(train), "test_rows": len(test), "predictions": len(predictions),
                "holdout_sha256": test_hashes, "figures": figures,
                "wall_seconds": perf_counter() - started,
            })
            record("Completed the six-model evaluation and verified saved inputs remained unchanged.")
        except Exception as error:
            record(f"Run stopped: {type(error).__name__}: {error}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"), nargs="?")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(evaluate_model(PROJECT_ROOT, json.load(sys.stdin)), allow_nan=False))
    elif args.action == "freeze":
        frozen = freeze(PROJECT_ROOT)
        print(f"Frozen {len(frozen['models'])} configurations in {PROTOCOL}.")
    elif args.action == "run":
        run(PROJECT_ROOT)
    else:
        parser.error("Select freeze or run.")


if __name__ == "__main__":
    main()
