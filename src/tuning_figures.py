"""Export Stage 4B selections and their matching fixed baselines."""

import numpy as np

from src.baseline_figures import FEATURE_LABELS
from src.visualization import _pyplot, _save_eda, publication_style


def plot_tuning(selected, directory):
    plt = _pyplot()
    paths = []
    labels = [f"{FEATURE_LABELS[r['feature_set']]} · "
              f"{'SVM' if r['model'] == 'svm' else 'RF'}" for r in selected]
    positions = np.arange(len(selected))
    footer = "Training CV used for selection. Error bars show sample SD across the five saved folds."

    def export(figure, name, title, note=footer):
        figure.suptitle(title, y=0.99)
        figure.text(0.5, 0.012, note, ha="center", fontsize=8)
        path = directory / f"tuning_cv_{name}.png"
        _save_eda(figure, path)
        paths.append(path)

    with publication_style():
        for metric, label in (("f1", "Defective F1"), ("recall", "Defective recall")):
            figure, axis = plt.subplots(figsize=(8.8, 4.4))
            for y, row in enumerate(selected):
                axis.plot([row[f"baseline_{metric}_mean"], row[f"{metric}_mean"]],
                          [y, y], color="0.7", linewidth=1)
            axis.scatter([r[f"baseline_{metric}_mean"] for r in selected], positions,
                         marker="o", s=72, facecolors="none", edgecolors="#0072B2", label="Stage 4A baseline")
            axis.errorbar([r[f"{metric}_mean"] for r in selected], positions,
                          xerr=[r[f"{metric}_sd"] for r in selected], color="#D55E00",
                          fmt="s", capsize=3, label="Stage 4B selection")
            axis.set(yticks=positions, yticklabels=labels, xlabel=label, xlim=(0.65, 1.01))
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.25)
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=8)
            export(figure, f"baseline_{metric}", f"Training CV: baseline versus selected {label}",
                   "Selected error bars show sample SD across folds. Baseline points show means.")

        figure, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
        for axis, metric, label in zip(axes, ("f1", "balanced_accuracy"), ("Defective F1", "Balanced accuracy")):
            for model, marker, color, shift in (("svm", "o", "#0072B2", -0.14),
                                                 ("random_forest", "s", "#D55E00", 0.14)):
                rows = [r for r in selected if r["model"] == model]
                axis.errorbar([r[f"{metric}_mean"] for r in rows], np.arange(3) + shift,
                              xerr=[r[f"{metric}_sd"] for r in rows], fmt=marker,
                              capsize=3, color=color, label="SVM" if model == "svm" else "Random Forest")
            axis.set(xlabel=label, xlim=(0.65, 1.01))
            axis.grid(axis="x", alpha=0.25)
        axes[0].set(yticks=np.arange(3), yticklabels=[FEATURE_LABELS[r["feature_set"]]
                                                    for r in selected if r["model"] == "svm"])
        axes[0].invert_yaxis()
        handles, keys = axes[1].get_legend_handles_labels()
        figure.legend(handles, keys, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, fontsize=8)
        export(figure, "classifier_comparison", "Training CV: SVM versus Random Forest")

        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for axis, model in zip(axes, ("svm", "random_forest")):
            rows = [r for r in selected if r["model"] == model]
            y = np.arange(len(rows))
            axis.errorbar([r["recall_mean"] for r in rows], y - 0.14,
                          xerr=[r["recall_sd"] for r in rows], fmt="o", capsize=3,
                          label="Defective recall", color="#0072B2")
            axis.errorbar([r["f1_mean"] for r in rows], y + 0.14, xerr=[r["f1_sd"] for r in rows],
                          fmt="s", capsize=3, label="Defective F1", color="#D55E00")
            axis.set(title="SVM" if model == "svm" else "Random Forest",
                     xlabel="Selected CV score", xlim=(0.65, 1.01))
            axis.grid(axis="x", alpha=0.25)
        axes[0].set(yticks=np.arange(3), yticklabels=[FEATURE_LABELS[r["feature_set"]]
                                                    for r in selected if r["model"] == "svm"])
        axes[0].invert_yaxis()
        axes[1].legend(loc="lower left", fontsize=8)
        export(figure, "feature_comparison", "Training CV: HOG, GLCM, and combined features")

        figure, axes = plt.subplots(1, 2, figsize=(10.6, 4), sharey=True)
        axes[0].errorbar([r["fit_seconds_mean"] for r in selected], positions,
                         xerr=[r["fit_seconds_sd"] for r in selected], fmt="o", capsize=3)
        axes[1].scatter([r["columns"] for r in selected], positions, marker="s", color="#0072B2")
        axes[0].set(yticks=positions, yticklabels=labels)
        axes[0].invert_yaxis()
        for axis, label in zip(axes, ("Mean fit seconds per fold (log scale)", "Feature dimensions (log scale)")):
            axis.set(xlabel=label, xscale="log")
            axis.grid(axis="x", alpha=0.25)
        export(figure, "resources", "Training CV: feature dimensions and fit cost",
               "Fit timing includes SVM scaling. Timing bars show sample SD.")

        import json

        figure, axis = plt.subplots(figsize=(12, 4.7))
        axis.axis("off")
        cells = []
        for label, r in zip(labels, selected):
            params = json.loads(r["parameters"])
            if r["model"] == "svm":
                gamma = params["gamma"]
                gamma = gamma if isinstance(gamma, str) else f"{gamma:.3g}"
                detail = f"C={params['C']:g}, gamma={gamma}"
                if r["feature_set"] == "hog16_glcm":
                    detail += f", GLCM weight={r['glcm_weight']:.4g}"
            else:
                detail = (f"trees={params['n_estimators']}, depth={params['max_depth']}, "
                          f"leaf={params['min_samples_leaf']}, features={params['max_features']}")
            cells.append([label, detail, f"{r['f1_mean']:.4f}", str(r["false_negatives_total"])])
        table = axis.table(cellText=cells, colLabels=["Features · model", "Selected hyperparameters", "F1", "FN"],
                           colWidths=[0.25, 0.52, 0.12, 0.08], cellLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.75)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#dce3ea")
            cell.set_facecolor("#e1edf6" if row == 0 else "#f4f7fa" if row % 2 else "white")
        export(figure, "hyperparameters", "Training CV: best hyperparameter configurations",
               "Selected by mean Defective F1. FN is the total across five training validation folds.")
    return paths
