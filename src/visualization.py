"""Save English-labeled PNG figures without an interactive display."""

from pathlib import Path

import numpy as np

from src.feature_extraction import FeatureConfig, extract_hog, glcm_properties


def _pyplot():
    """Load the optional plotting stack only when a plot is requested."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(figure, path: Path, *, dpi: int = 150, rect=None) -> None:
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=rect)
    figure.savefig(
        path, dpi=dpi, bbox_inches="tight", facecolor="white",
        metadata={"Software": "MAT 416"},
    )
    plt.close(figure)


def plot_samples(images, titles, path: Path) -> None:
    if len(images) == 0 or len(images) != len(titles):
        raise ValueError("Provide at least one image and one title per image.")
    plt = _pyplot()
    figure, axes = plt.subplots(
        1, len(images), figsize=(4 * len(images), 4), squeeze=False
    )
    for axis, image, title in zip(axes.flat, images, titles):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    _save(figure, path)


def plot_histogram(image: np.ndarray, path: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots()
    axis.hist(image.ravel(), bins=64)
    axis.set(
        xlabel="Pixel intensity", ylabel="Pixel count",
        title="Intensity distribution",
    )
    _save(figure, path)


def plot_hog(image: np.ndarray, path: Path,
             config: FeatureConfig = FeatureConfig()) -> None:
    _, visualization = extract_hog(image, config, visualize=True)
    plot_samples(
        [image, visualization], ["Preprocessed image", "HOG descriptor"], path
    )


def plot_glcm(image: np.ndarray, path: Path,
              config: FeatureConfig = FeatureConfig()) -> None:
    plt = _pyplot()
    properties = glcm_properties(image, config)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (name, values) in zip(axes.flat, properties.items()):
        for distance, row in zip(config.distances, values):
            axis.plot(
                np.degrees(config.angles), row, marker="o",
                label=f"Distance {distance}",
            )
        axis.set(
            title=name.capitalize(), xlabel="Angle (degrees)",
            ylabel="Property value",
        )
        axis.legend()
    _save(figure, path)


def plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    plt = _pyplot()
    figure, axis = plt.subplots()
    ConfusionMatrixDisplay(
        matrix, display_labels=["Acceptable", "Defective"]
    ).plot(ax=axis, colorbar=False)
    axis.set_title("Confusion matrix")
    _save(figure, path)


def plot_model_comparison(table, path: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(9, 5))
    table[["accuracy", "precision", "recall", "f1"]].plot.bar(ax=axis)
    axis.set(
        title="Model performance comparison", xlabel="Model",
        ylabel="Score", ylim=(0, 1),
    )
    _save(figure, path)


CLASS_ORDER = ("ok_front", "def_front")
CLASS_TITLES = {"ok_front": "OK", "def_front": "Defective"}
CLASS_COLORS = {"ok_front": "#0072B2", "def_front": "#D55E00"}
STAGE_TITLES = {
    "original": "Original", "resized": "Processing input",
    "normalized": "Normalized",
}


def publication_style():
    """Return a scoped, consistent style without changing global defaults."""
    return _pyplot().rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.titlesize": 14,
    })


def _save_eda(figure, path: Path) -> None:
    """Export a 300-DPI PNG with space for a title and figure note."""
    _save(figure, path, dpi=300, rect=(0, 0.035, 1, 0.95))


def _training_only(rows: list[dict]) -> None:
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("EDA analysis figures require training rows only.")


def plot_class_distribution(rows: list[dict], path: Path) -> None:
    """Show manifest counts; this plot does not inspect test pixel values."""
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.7))
    for axis, split, title in zip(
        axes, ("all", "train", "test"), ("All images", "Training", "Test")
    ):
        selected = {
            row["class_name"]: row for row in rows if row["split"] == split
        }
        counts = [selected[name]["image_count"] for name in CLASS_ORDER]
        bars = axis.bar(
            [CLASS_TITLES[name] for name in CLASS_ORDER], counts,
            color=[CLASS_COLORS[name] for name in CLASS_ORDER], width=0.6,
        )
        axis.bar_label(bars, labels=[
            f"{selected[name]['image_count']}\n"
            f"({selected[name]['percentage']:.1f}%)"
            for name in CLASS_ORDER
        ], padding=4)
        axis.set(title=title, ylabel="Number of images")
        axis.set_ylim(0, max(counts) * 1.3)
    figure.suptitle("Class distribution from the saved manifests")
    _save_eda(figure, path)


def plot_training_samples(samples: list[dict], path: Path, seed: int) -> None:
    """Display deterministic random examples, without claiming part identity."""
    _training_only(samples)
    plt = _pyplot()
    columns = sum(row["class_name"] == "ok_front" for row in samples)
    figure, axes = plt.subplots(2, columns, figsize=(3 * columns, 6), squeeze=False)
    for row_index, name in enumerate(CLASS_ORDER):
        selected = [row for row in samples if row["class_name"] == name]
        for axis, sample in zip(axes[row_index], selected):
            axis.imshow(
                sample["original"], cmap="gray", vmin=0, vmax=255,
                interpolation="nearest",
            )
            axis.set_title(f"{CLASS_TITLES[name]} - sample {sample['sample_rank']}")
            axis.axis("off")
    figure.suptitle(f"Training images selected with random seed {seed}")
    figure.text(
        0.5, 0.012, "Sample paths are recorded in eda_selected_samples.csv.",
        ha="center", fontsize=9,
    )
    _save_eda(figure, path)


def plot_class_histograms(rows: list[dict], path: Path) -> None:
    """Compare class pixel probabilities with a common binning per stage."""
    _training_only(rows)
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for axis, stage in zip(axes, STAGE_TITLES):
        for name in CLASS_ORDER:
            selected = [
                row for row in rows
                if row["stage"] == stage and row["class_name"] == name
            ]
            edges = [row["bin_left"] for row in selected]
            edges.append(selected[-1]["bin_right"])
            axis.stairs(
                [row["probability"] for row in selected], edges,
                color=CLASS_COLORS[name], label=CLASS_TITLES[name], linewidth=1.4,
            )
        axis.set(
            title=STAGE_TITLES[stage],
            xlabel="Intensity [0, 1]" if stage == "normalized" else "Intensity [0, 255]",
            ylabel="Pixel probability per bin",
        )
        axis.set_xlim(0, 1 if stage == "normalized" else 256)
        axis.legend(fontsize=9)
    figure.suptitle("Training grayscale intensity distributions")
    figure.text(
        0.5, 0.012, "256 bins per stage; each class histogram sums to one.",
        ha="center", fontsize=9,
    )
    _save_eda(figure, path)


def plot_image_statistics(rows: list[dict], path: Path) -> None:
    """Compare per-image summaries; pixels are not treated as observations."""
    _training_only(rows)
    plt = _pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 6.8))
    panels = (
        ("original", "mean", "Original: image mean", "Intensity [0, 255]"),
        ("original", "std", "Original: within-image SD", "Intensity units"),
        ("normalized", "mean", "Normalized: image mean", "Intensity [0, 1]"),
        ("normalized", "std", "Normalized: within-image SD", "Normalized units"),
    )
    for axis, (stage, metric, title, ylabel) in zip(axes.flat, panels):
        values = [[
            row[metric] for row in rows
            if row["stage"] == stage and row["class_name"] == name
        ] for name in CLASS_ORDER]
        boxes = axis.boxplot(
            values, positions=[0, 1], widths=0.5, patch_artist=True,
            flierprops={"markersize": 2, "alpha": 0.35},
            medianprops={"color": "black", "linewidth": 1.3},
        )
        for patch, name in zip(boxes["boxes"], CLASS_ORDER):
            patch.set_facecolor(CLASS_COLORS[name])
            patch.set_alpha(0.65)
        axis.set_xticks([0, 1], [
            f"{CLASS_TITLES[name]}\n(n={len(value)})"
            for name, value in zip(CLASS_ORDER, values)
        ])
        axis.set(title=title, ylabel=ylabel)
    figure.suptitle("Training image-level statistical comparisons")
    figure.text(
        0.5, 0.012,
        "Boxes: median and IQR; whiskers: 1.5 IQR; points: outliers.",
        ha="center", fontsize=9,
    )
    _save_eda(figure, path)


def _first_samples(samples):
    _training_only(samples)
    return [
        next(sample for sample in samples if sample["class_name"] == name)
        for name in CLASS_ORDER
    ]


def plot_preprocessing_comparison(samples: list[dict], path: Path) -> None:
    """Use fixed display ranges so plotting does not silently normalize data."""
    plt = _pyplot()
    figure, axes = plt.subplots(2, 3, figsize=(9, 8))
    for row_index, sample in enumerate(_first_samples(samples)):
        for column, stage in enumerate(STAGE_TITLES):
            image = sample[stage]
            axis = axes[row_index, column]
            axis.imshow(
                image, cmap="gray", vmin=0,
                vmax=1 if stage == "normalized" else 255,
                interpolation="nearest",
            )
            axis.set_title(
                f"{CLASS_TITLES[sample['class_name']]}: {STAGE_TITLES[stage]}\n"
                f"{image.shape[0]} x {image.shape[1]}, {image.dtype}"
            )
            axis.axis("off")
    figure.suptitle("Original, processing input, and normalized training images")
    figure.text(
        0.5, 0.012,
        "Fixed display ranges: original/resized [0, 255]; normalized [0, 1].",
        ha="center", fontsize=9,
    )
    _save_eda(figure, path)


def plot_preprocessing_histograms(samples: list[dict], path: Path) -> None:
    """Separate resizing effects from intensity rescaling for the same images."""
    plt = _pyplot()
    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    for row_index, sample in enumerate(_first_samples(samples)):
        for column, stage in enumerate(STAGE_TITLES):
            counts, edges = sample["histograms"][stage]
            stats = sample["statistics"][stage]
            axis = axes[row_index, column]
            axis.stairs(
                counts / counts.sum(), edges,
                color=CLASS_COLORS[sample["class_name"]], linewidth=1.2,
            )
            axis.set(
                title=(
                    f"{CLASS_TITLES[sample['class_name']]}: {STAGE_TITLES[stage]}\n"
                    f"Mean = {stats['mean']:.3f}; SD = {stats['std']:.3f}"
                ),
                xlabel="Intensity [0, 1]" if stage == "normalized" else "Intensity [0, 255]",
                ylabel="Pixel probability per bin",
            )
            axis.set_xlim(0, 1 if stage == "normalized" else 256)
    figure.suptitle("Before-and-after histograms for the same training samples")
    _save_eda(figure, path)


def plot_class_comparisons(rows: list[dict], path: Path) -> None:
    """Plot observed dominance effects without inferential significance claims."""
    _training_only(rows)
    plt = _pyplot()
    selected = [row for row in rows if row["metric"] in {"mean", "std"}]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(selected))
    values = [row["cliffs_delta"] for row in selected]
    axis.axvline(0, color="0.5", linestyle="--", linewidth=1)
    axis.scatter(values, positions, s=48, color="#0072B2", zorder=3)
    axis.set_yticks(positions, [
        f"{STAGE_TITLES[row['stage']]}: image {row['metric']}" for row in selected
    ])
    axis.invert_yaxis()
    for value, position in zip(values, positions):
        axis.annotate(
            f"{value:+.3f}", (value, position), xytext=(7, 0),
            textcoords="offset points", va="center", fontsize=9,
        )
    axis.set(
        xlim=(-1.12, 1.12),
        xlabel="Observed Cliff's delta (Defective relative to OK)",
        title="Training-only descriptive class comparisons",
    )
    figure.text(
        0.5, 0.012,
        "Positive: larger values in Defective; image-level effects, no p-values.",
        ha="center", fontsize=9,
    )
    _save_eda(figure, path)
