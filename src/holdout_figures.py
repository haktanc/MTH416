"""Plot the frozen models' Stage 5 scores and deterministic image examples."""

from pathlib import Path

import numpy as np

from src.data_loader import load_grayscale
from src.feature_pipeline import file_sha256, read_csv
from src.visualization import _pyplot, _save_eda, publication_style


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#5E3C99", "#806400")
CLASS_NAMES = {0: "OK", 1: "Defective"}
OUTCOMES = ("False negatives", "False positives", "Correct defective", "Correct OK")


def plot_holdout(root, models, metrics, predictions, examples):
    """Render 300-DPI PNGs consistently and return their relative paths."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    from src.holdout_experiment import FEATURE_NAMES, MODEL_NAMES

    plt = _pyplot()
    paths = []
    positions = np.arange(len(models))
    labels = [r["name"] for r in models]
    tick_labels = [f"{FEATURE_NAMES[r['feature_set']]}\n{MODEL_NAMES[r['model']]}" for r in models]
    lookup = {(r["feature_set"], r["model"]): r for r in metrics}
    feature_sets = list(FEATURE_NAMES)
    classifiers = list(MODEL_NAMES)

    def export(figure, key, title, note="Stage 5 · Independent image-level holdout · 260 images"):
        figure.suptitle(title, y=0.99)
        figure.text(0.5, 0.012, note, ha="center", fontsize=8)
        path = root / "results" / "figures" / f"holdout_{key}.png"
        _save_eda(figure, path)
        paths.append(path.relative_to(root).as_posix())

    def model_ticks(axis):
        axis.set(xticks=positions, xticklabels=tick_labels)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.23)

    def grouped_bars(axis, groups, names, series, *, score=False):
        centers = np.arange(len(groups))
        width = 0.78 / len(series)
        for i, (label, values, color) in enumerate(series):
            bars = axis.bar(centers + width * (i - (len(series) - 1) / 2), values,
                            width=width, color=color, label=label)
            axis.bar_label(bars, fmt="%.3f" if score else "%d", padding=3, fontsize=8)
        axis.set(xticks=centers, xticklabels=names)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.23)
        axis.legend(loc="lower left" if score else "upper left")
        if score:
            axis.set_ylim(0, 1.12)

    with publication_style():
        for field, title in (("f1", "Defective F1"), ("recall", "Defective recall")):
            fig, ax = plt.subplots(figsize=(11, 4.6))
            values = np.array([r[field] for r in metrics])
            errors = None
            if field == "recall":
                errors = np.array([values - [r["recall_ci95_low"] for r in metrics],
                                   [r["recall_ci95_high"] for r in metrics] - values])
            bars = ax.bar(positions, values, color=COLORS, yerr=errors, capsize=4)
            ax.bar_label(bars, fmt="%.3f", padding=4)
            ax.set(ylabel=title, ylim=(0, 1.13))
            model_ticks(ax)
            if field == "recall":
                export(fig, field, f"Holdout {title}",
                       "Stage 5 · Error bars show nominal 95% Wilson intervals for 156 Defective images.")
            else:
                export(fig, field, f"Holdout {title}")

        fig, ax = plt.subplots(figsize=(11, 4.6))
        grouped_bars(ax, positions, tick_labels, [
            ("Accuracy", [r["accuracy"] for r in metrics], COLORS[0]),
            ("Balanced accuracy", [r["balanced_accuracy"] for r in metrics], COLORS[1]),
        ], score=True)
        ax.set_ylabel("Score")
        export(fig, "accuracy", "Holdout accuracy and balanced accuracy")

        fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.7))
        for ax, row in zip(axes.flat, metrics, strict=True):
            matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for actual, predicted in np.ndindex(matrix.shape):
                rate = normalized[actual, predicted]
                ax.text(predicted, actual, f"{matrix[actual, predicted]}\n({rate:.1%})",
                        ha="center", va="center", color="white" if rate > 0.55 else "#163049")
            ax.set(title=row["name"], xticks=[0, 1], yticks=[0, 1],
                   xticklabels=CLASS_NAMES.values(), yticklabels=CLASS_NAMES.values(),
                   xlabel="Predicted class", ylabel="Actual class")
        export(fig, "confusion", "Holdout confusion matrices", "Counts and percentages within each actual class.")

        for kind in ("roc", "precision_recall"):
            fig, ax = plt.subplots(figsize=(8.7, 5.4))
            for row, color, label in zip(metrics, COLORS, labels, strict=True):
                series = [p for p in predictions if p["candidate"] == row["candidate"]]
                truth, scores = zip(*((p["true_label"], p["defective_score"]) for p in series))
                if kind == "roc":
                    x, y, _ = roc_curve(truth, scores, pos_label=1)
                    statistic = f"AUC {row['roc_auc']:.3f}"
                    ax.plot(x, y, color=color, linewidth=1.6, label=f"{label} ({statistic})")
                else:
                    y, x, _ = precision_recall_curve(truth, scores, pos_label=1)
                    statistic = f"AP {row['average_precision']:.3f}"
                    ax.step(x[::-1], y[::-1], where="pre", color=color, linewidth=1.6,
                            label=f"{label} ({statistic})")
            if kind == "roc":
                ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", label="Chance reference")
                ax.set(xlabel="False positive rate", ylabel="Defective true positive rate")
            else:
                ax.axhline(np.mean(truth), linestyle="--", color="0.55", label="Defective prevalence")
                ax.set(xlabel="Defective recall", ylabel="Defective precision")
            ax.set(xlim=(0, 1), ylim=(0, 1.03))
            ax.legend(loc="lower left", bbox_to_anchor=(1.01, 0), fontsize=9)
            ax.grid(alpha=0.2)
            export(fig, kind, "Holdout ROC curves" if kind == "roc" else "Holdout precision–recall curves")

        fig, ax = plt.subplots(figsize=(11, 4.6))
        grouped_bars(ax, positions, tick_labels, [
            ("False positives", [r["fp"] for r in metrics], COLORS[0]),
            ("False negatives", [r["fn"] for r in metrics], COLORS[1]),
        ])
        ax.set(ylabel="Number of images", ylim=(0, 1.25 * max(max(r["fp"], r["fn"]) for r in metrics)))
        export(fig, "errors", "Holdout false alarms and missed defects")

        for group_by, groups, series, names, key, title in (
            ("model", classifiers, feature_sets, MODEL_NAMES, "feature_comparison",
             "Feature representations by classifier"),
            ("feature_set", feature_sets, classifiers, FEATURE_NAMES, "classifier_comparison",
             "Classifiers by feature representation"),
        ):
            series_names = FEATURE_NAMES if group_by == "model" else MODEL_NAMES
            fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
            for ax, (metric, metric_name) in zip(axes, (("f1", "Defective F1"), ("recall", "Defective recall"))):
                values = []
                for j, item in enumerate(series):
                    pairs = [(item, group) if group_by == "model" else (group, item) for group in groups]
                    values.append((series_names[item], [lookup[pair][metric] for pair in pairs], COLORS[j]))
                grouped_bars(ax, groups, [names[g] for g in groups], values, score=True)
                ax.set(ylabel=metric_name)
            export(fig, key, f"Holdout {title.lower()}")

        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
        for ax, field, title in zip(axes, ("fit_seconds", "inference_seconds"),
                                    ("Fit 1,040 images", "Predict and score 260 images")):
            ax.barh(positions, [r[field] for r in metrics], color=COLORS)
            ax.set(xscale="log", xlabel="Seconds (log scale)", title=title)
            ax.grid(axis="x", alpha=0.23)
            ax.set_axisbelow(True)
        axes[0].set(yticks=positions, yticklabels=labels)
        axes[0].invert_yaxis()
        export(fig, "runtime", "Final training and inference time",
               "One CPU run per model, with one numerical thread. Matrix loading excluded.")

        audit = {r["image_path"]: r["sha256"] for r in read_csv(root / "results/tables/dataset_audit.csv")}
        primary_name = next(r["name"] for r in metrics if r["primary"])
        fig, axes = plt.subplots(4, 3, figsize=(10.5, 13.5))
        for row_axes, outcome in zip(axes, OUTCOMES, strict=True):
            samples = [r for r in examples if r["outcome"] == outcome]
            for ax in row_axes:
                ax.set_axis_off()
            if not samples:
                row_axes[1].text(0.5, 0.5, f"{outcome}: no examples", ha="center", va="center")
            for ax, row in zip(row_axes, samples):
                path = root / row["image_path"]
                if file_sha256(path) != audit[row["image_path"]]:
                    raise ValueError("Example image differs from the original Stage 1 inventory.")
                image = load_grayscale(path)
                if image.shape != (512, 512):
                    raise ValueError("Expected a native 512×512 holdout image.")
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{outcome} · {Path(row['image_path']).name}\n"
                             f"Actual {CLASS_NAMES[row['true_label']]} · "
                             f"Predicted {CLASS_NAMES[row['predicted_label']]}", fontsize=9)
        export(fig, "examples", f"Holdout image examples · {primary_name}",
               "Primary model for every panel. Filename identifiers and full IDs are recorded in holdout_examples.csv.")
    return paths
