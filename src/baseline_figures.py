"""Publication figures for the Stage 4A training cross-validation results."""

import numpy as np

from src.visualization import _pyplot, _save_eda, publication_style


FEATURE_LABELS = {
    "hog8": "HOG 8×8", "hog16": "HOG 16×16", "glcm": "GLCM",
    "hog8_glcm": "HOG 8×8 + GLCM", "hog16_glcm": "HOG 16×16 + GLCM",
}
MODEL_STYLES = {
    "svm": ("RBF SVM", "#0072B2", "o", -0.13),
    "random_forest": ("Random Forest", "#D55E00", "s", 0.13),
}


def plot_baselines(summary, directory):
    """Plot means and sample SDs, leaving incomplete combinations blank."""
    plt = _pyplot()
    paths = []
    lookup = {(row["feature_set"], row["model"]): row for row in summary}

    def plot(name, title, panels, features):
        figure, axes = plt.subplots(1, len(panels), squeeze=False,
                                    figsize=(6.8 * len(panels), 4.1))
        ticks = np.arange(len(features))
        for axis, (metric, label) in zip(axes[0], panels):
            for model, (model_label, color, marker, offset) in MODEL_STYLES.items():
                pairs = [(index, lookup[feature, model]) for index, feature in enumerate(features)
                         if (feature, model) in lookup]
                axis.errorbar(
                    [row[f"{metric}_mean"] for _, row in pairs],
                    [index + offset for index, _ in pairs],
                    xerr=[row[f"{metric}_sd"] for _, row in pairs],
                    fmt=marker, color=color, capsize=3, label=model_label,
                )
            axis.set_yticks(ticks, [FEATURE_LABELS[key] for key in features])
            axis.invert_yaxis()
            axis.set_xlabel(label)
            axis.grid(axis="x", linewidth=0.6, alpha=0.3)
            if metric == "fit_seconds":
                axis.set_xscale("log")
            else:
                axis.set_xlim(0, 1.02)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2,
                      bbox_to_anchor=(0.5, 0.94), frameon=False)
        figure.suptitle(f"Training-set cross-validation: {title}", y=0.99)
        figure.text(0.5, 0.012, "Means ± sample SD across five saved validation folds.",
                    ha="center", fontsize=9)
        path = directory / f"baseline_cv_{name}.png"
        _save_eda(figure, path)
        paths.append(path)

    all_features = list(FEATURE_LABELS)
    with publication_style():
        plot("f1", "Defective F1", [("f1", "Defective F1")], all_features)
        plot("recall", "Defective recall", [("recall", "Defective recall")], all_features)
        plot("accuracy", "Accuracy and balanced accuracy",
             [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy")], all_features)
        plot("training_time", "Training time",
             [("fit_seconds", "Fit time (seconds, log scale)")], all_features)
        plot("hog_comparison", "HOG cell-size comparison",
             [("f1", "Defective F1"), ("recall", "Defective recall")],
             [feature for feature in all_features if feature != "glcm"])
    return paths
