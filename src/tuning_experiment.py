"""Stage 4B search on the saved training folds, with resumable fold results."""

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numpy as np

from src.baseline_experiment import (
    FEATURES, METRICS, MIB, MODELS, RESERVE_MIB, align_rows, estimated_peak_mib,
    evaluate_fold, run_worker, saved_folds, smoke_subset, summarize,
)
from src.data_loader import PROJECT_ROOT, read_manifest
from src.feature_pipeline import file_sha256, read_csv, system_resources
from src.reporting import output_directory, write_csv, write_json


FEATURE_SETS = ("hog16", "glcm", "hog16_glcm")
BASELINE_FILES = ("baseline_cv_run.json", "baseline_cv_folds.csv", "baseline_cv_summary.csv")
FOLD_FIELDS = ["feature_set", "model", "candidate", "fold", "columns",
               "parameters", "glcm_weight", "source", "fit_rows", "validation_rows",
               *METRICS, "true_negatives", "false_positives", "true_positives",
               "load_fit_seconds", "load_validation_seconds", "prediction_check_seconds"]


def search_plan():
    """Six candidates per pair, with two controlled combined-SVM weight trials."""
    plan = []
    for feature in FEATURE_SETS:
        columns = 34596 * ("hog16" in FEATURES[feature]) + 48 * ("glcm" in FEATURES[feature])
        weights = [(1.0, 1.0, "scale"), (1.0, 0.3, "scale"), (1.0, 3.0, "scale"),
                   (1.0, 1.0, 0.25 / columns), (1.0, 1.0, 4.0 / columns),
                   (1.0, 3.0, 0.25 / columns)]
        if feature == "hog16_glcm":
            balanced = (34596 / 48) ** 0.5
            weights += [(balanced, 1.0, "scale"), (balanced, 3.0, "scale")]
        svm = [(dict(C=c, gamma=gamma), weight) for weight, c, gamma in weights]
        rf = [(dict(n_estimators=n, max_depth=depth, min_samples_leaf=leaf,
                    max_features=max_features), 1.0)
              for n, depth, leaf, max_features in [
                  (100, None, 1, "sqrt"), (60, None, 1, "sqrt"), (140, None, 1, "sqrt"),
                  (80, 12, 1, "sqrt"), (80, None, 3, "sqrt"), (100, None, 1, "log2"),
              ]]
        for model, options in zip(MODELS, (svm, rf)):
            for index, (parameters, weight) in enumerate(options):
                plan.append(dict(feature_set=feature, model=model, columns=columns,
                                 candidate=f"{feature}_{model}_{index:02d}",
                                 parameters=parameters, glcm_weight=weight, baseline=index == 0))
    return plan


def load_training(root):
    """Authenticate only training inputs, avoiding even holdout manifest reads."""
    tables = root / "results/tables"
    meta = json.loads((tables / BASELINE_FILES[0]).read_text())
    stage3 = json.loads((tables / "feature_run_summary.json").read_text())
    old = meta["provenance"]
    if (meta["completed_combinations"] != 10 or meta["training_rows"] != 1040
            or meta["scope"] != "training-only cross-validation"
            or stage3["provenance"]["configuration"]["image_shape"] != [512, 512]):
        raise ValueError("Expected complete Stage 4A training-only results and native features.")
    for relative, expected in stage3["provenance"]["source_code_sha256"].items():
        if file_sha256(root / relative) != expected:
            raise ValueError("Stage 3 feature source no longer matches the cache provenance.")
    for name, expected in (stage3["provenance"]["versions"] | meta["versions"]).items():
        if version(name) != expected:
            raise ValueError(f"Dependency version differs from the saved experiment: {name}")
    protected = {
        "data/splits/train.csv": old["train_manifest_sha256"],
        "results/tables/cv_folds.csv": old["cv_folds_sha256"],
        "results/tables/feature_train_index.csv": old["train_index_sha256"],
        "results/tables/feature_run_summary.json": old["stage3_summary_sha256"],
    }
    manifest = read_manifest(root / "data/splits/train.csv", "train", root)
    reference = read_csv(tables / "feature_train_index.csv")
    labels = [int(r["label"]) for r in reference]
    align_rows(manifest, reference, labels)
    if Counter(labels) != {0: 415, 1: 625}:
        raise ValueError("Unexpected training-set class counts.")
    folds = saved_folds(reference, read_csv(tables / "cv_folds.csv"))
    layout = {}
    for method in ("hog16", "glcm"):
        item = old["training_caches"][method]
        expected_path = f"results/cache/features/train_{method}.npy"
        path = (root / item["path"]).resolve()
        if item["path"] != expected_path or path != root / expected_path:
            raise ValueError("Stage 4B may only load the canonical training caches.")
        protected[expected_path] = item["sha256"]
        entry, = (r for r in stage3["caches"] if r["split"] == "train" and r["method"] == method)
        protected["results/cache/features/train_labels.npy"] = entry["labels_sha256"]
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(matrix.shape) != item["shape"] or str(matrix.dtype) != item["dtype"]:
            raise ValueError(f"Cache geometry differs for {method}.")
        y = np.load(path.parent / "train_labels.npy", allow_pickle=False)
        layout[method] = {**item, "order": align_rows(reference, reference, y).tolist()}
        del matrix
    verify_inputs(root, protected)
    protected.update({f"results/tables/{name}": file_sha256(tables / name) for name in BASELINE_FILES})
    return layout, folds, reference, protected


def verify_inputs(root, hashes):
    for path, digest in hashes.items():
        if file_sha256(root / path) != digest:
            raise ValueError(f"Saved training input changed: {path}")


def baseline_records(root, plan):
    raw = read_csv(root / "results/tables/baseline_cv_folds.csv")
    expected = Counter((feature, model, fold)
                       for feature in FEATURES for model in MODELS for fold in range(1, 6))
    if Counter((r["feature_set"], r["model"], int(r["fold"])) for r in raw) != expected:
        raise ValueError("Stage 4A must retain all ten complete baseline results.")
    out = {}
    for candidate in (r for r in plan if r["baseline"]):
        matched = [r for r in raw if all(r[k] == candidate[k] for k in ("feature_set", "model"))]
        out[candidate["candidate"]] = [
            {k: r[k] if k in ("feature_set", "model") else int(r[k]) if k in (
                "columns", "fold", "fit_rows", "validation_rows", "true_negatives",
                "true_positives", "false_negatives", "false_positives",
            ) else float(r[k]) for k in r if k in FOLD_FIELDS}
            for r in matched
        ]
    return out


def checkpoint(path, value):
    temporary = path.with_suffix(".partial")
    write_json(temporary, value)
    temporary.replace(path)


def evaluate_job(root, job):
    """Commit each finished fold before moving to the next one."""
    if "tasks" not in job:
        return evaluate_fold(root, job)
    folder = (root / job["checkpoint"]).resolve()
    if not folder.is_relative_to((root / "results/cache/tuning_cv").resolve()):
        raise ValueError("Checkpoint must stay inside the Stage 4B cache.")
    for task in job["tasks"]:
        started = time.perf_counter()
        result = evaluate_fold(root, {**job, **task, "capture_predictions": True})
        checkpoint(folder / f"{task['fold']}.json",
                   {**result, "fold": task["fold"], "fold_wall_seconds": time.perf_counter() - started})
    return {"folds_finished": len(job["tasks"])}


def read_completed(directory, candidate):
    folder = directory / candidate["candidate"]
    return [json.loads(p.read_text()) for p in sorted(folder.glob("[1-5].json"))]


def report_rows(plan, fold_records):
    folds, summaries = [], []
    for candidate in plan:
        candidate_id = candidate["candidate"]
        records = fold_records.get(candidate_id, [])
        base = {k: candidate[k] for k in ("feature_set", "model", "columns", "candidate", "glcm_weight")}
        base.update(parameters=json.dumps(candidate["parameters"], sort_keys=True),
                    source="stage4a" if candidate["baseline"] else "stage4b")
        rows = [{**base, **record} for record in records]
        folds.extend(rows)
        if len(rows) == 5:
            # Baseline summary infrastructure expects the resource field too.
            payload = [{"minimum_available_mib": 0.0, **r} for r in rows]
            row, = summarize(payload, [(base["feature_set"], base["model"])])
            row.pop("minimum_available_mib")
            row["false_positives_mean"] = float(np.mean([r["false_positives"] for r in rows]))
            row["false_positives_sd"] = float(np.std([r["false_positives"] for r in rows], ddof=1))
            summaries.append({**base, **row})
    return folds, summaries


def select_candidates(summaries):
    """Rank by mean defective F1, then fewer FNs, then candidate ID."""
    best = []
    for feature in FEATURE_SETS:
        for model in MODELS:
            group = [r for r in summaries if (r["feature_set"], r["model"]) == (feature, model)]
            baseline = next(r for r in group if r["source"] == "stage4a")
            winner = min(group, key=lambda r: (-r["f1_mean"], r["false_negatives_total"], r["candidate"]))
            best.append({**winner, **{
                f"baseline_{k}": baseline[k] for k in (
                    "accuracy_mean", "balanced_accuracy_mean", "precision_mean", "recall_mean",
                    "f1_mean", "roc_auc_mean", "false_negatives_total", "false_positives_total",
                    "fit_seconds_mean", "predict_seconds_mean",
                )
            }, "f1_gain": winner["f1_mean"] - baseline["f1_mean"]})
    return best


def run(root=PROJECT_ROOT):
    root = Path(root).resolve()
    started = time.perf_counter()
    layout, folds, reference, protected = load_training(root)
    labels = np.array([int(r["label"]) for r in reference])
    plan = search_plan()
    tables = output_directory(root / "results/tables", root)
    cache = output_directory(root / "results/cache/tuning_cv", root)
    stamp = {"plan": plan, "training_hashes": protected,
             "implementation": {f"src/{name}.py": file_sha256(root / f"src/{name}.py")
                                for name in ("tuning_experiment", "baseline_experiment", "baseline_scaling",
                                             "model_training", "evaluation")},
             "python": sys.version.split()[0],
             "versions": {name: version(name) for name in ("numpy", "scikit-learn", "threadpoolctl")}}
    key = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()[:16]
    directory = output_directory(cache / key, root)
    fold_records = baseline_records(root, plan)
    errors, resources = [], []
    fit_set, val_set = folds[0][1:]
    for model in MODELS:
        smoke_file = directory / f"smoke_{model}.json"
        if smoke_file.exists():
            resources.append(json.loads(smoke_file.read_text()))
            continue
        smoke = run_worker(root, {
            "labels": labels.tolist(), "model": model,
            "components": [layout[name] for name in FEATURES["hog16_glcm"]],
            "fit": smoke_subset(fit_set, labels, 48).tolist(),
            "valid": smoke_subset(val_set, labels, 16).tolist(),
            "parameters": {"n_estimators": 140} if model == "random_forest" else {"C": 3.0},
            "glcm_weight": (34596 / 48) ** 0.5 if model == "svm" else 1.0,
        }, module=__name__ if __name__ != "__main__" else "src.tuning_experiment")
        report = {"model": model, **smoke}
        checkpoint(smoke_file, report)
        resources.append(report)
        print(f"Smoke {model}: {smoke['fit_seconds']:.2f}s, peak {smoke['peak_rss_mib']:.0f} MiB", flush=True)

    module = "src.tuning_experiment"
    for candidate in plan:
        if candidate["baseline"]:
            continue
        name, feature, model = (candidate[k] for k in ("candidate", "feature_set", "model"))
        folder = output_directory(directory / name, root)
        finished = {r["fold"] for r in read_completed(directory, candidate)}
        tasks = [dict(fold=k, fit=fit.tolist(), valid=val.tolist())
                 for k, fit, val in folds if k not in finished]
        if tasks:
            estimate = estimated_peak_mib(model, len(fit_set), len(val_set), candidate["columns"])
            available = system_resources(root)["available_ram_bytes"] / MIB
            print(f"{name}: {len(tasks)} folds, {available:.0f} MiB available", flush=True)
            try:
                if available < estimate + RESERVE_MIB:
                    raise MemoryError(f"Need {estimate + RESERVE_MIB:.0f} MiB, only {available:.0f} MiB available.")
                telemetry = run_worker(root, {
                    **candidate, "tasks": tasks, "labels": labels.tolist(),
                    "components": [layout[method] for method in FEATURES[feature]],
                    "checkpoint": folder.relative_to(root).as_posix(),
                }, module=module)
                checkpoint(folder / "worker.json", telemetry)
            except (MemoryError, RuntimeError) as error:
                failure = {"candidate": name, "reason": str(error)}
                errors.append(failure)
                write_json(tables / "tuning_cv_failures.json", errors)
                print(f"{name}: {error}", flush=True)
        fold_records[name] = read_completed(directory, candidate)
        rows, summaries = report_rows(plan, fold_records)
        write_csv(tables / "tuning_cv_folds.csv", rows, FOLD_FIELDS)
        write_csv(tables / "tuning_cv_candidates.csv", summaries, list(summaries[0]))
        print(f"{len(summaries)}/{len(plan)} complete candidates", flush=True)
    verify_inputs(root, protected)
    best = select_candidates(summaries)
    write_csv(tables / "tuning_cv_selected.csv", best, list(best[0]))
    oof_fields = ["candidate", "fold", "image_id", "truth", "prediction", "score"]
    oof = []
    for selected in best:
        if selected["source"] == "stage4a":
            continue
        indices = []
        for row in fold_records[selected["candidate"]]:
            values = row["predictions"]
            indices.extend(values["indices"])
            oof.extend(dict(candidate=selected["candidate"], fold=row["fold"],
                            image_id=reference[i]["image_id"], truth=int(labels[i]),
                            prediction=pred, score=score)
                       for i, pred, score in zip(values["indices"], values["labels"], values["scores"], strict=True))
        if sorted(indices) != list(range(len(reference))):
            raise ValueError("Selected predictions do not cover training IDs exactly once.")
    write_csv(tables / "tuning_cv_selected_oof.csv", oof, oof_fields)
    write_json(tables / "tuning_cv_failures.json", errors)
    from src.tuning_figures import plot_tuning

    figures = plot_tuning(best, output_directory(root / "results/figures", root))
    write_json(tables / "tuning_cv_run.json", {
        **stamp, "scope": "training-only tuning on saved folds", "seed": 42,
        "training_rows": len(reference), "completed_candidates": len(summaries),
        "fold_records": len(rows), "smoke": resources, "checkpoint_key": key,
        "failures": errors, "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "summed_new_fit_seconds": sum(row["fit_seconds"] for row in rows if row["source"] == "stage4b"),
        "peak_worker_rss_mib": max(row["peak_rss_mib"] for row in rows if row["source"] == "stage4b"),
        "selection": "Highest mean defective F1, then fewer false negatives, then candidate ID",
        "figures": [p.relative_to(root).as_posix() for p in figures],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(evaluate_job(Path.cwd(), json.load(sys.stdin))))
    else:
        run()


if __name__ == "__main__":
    main()
