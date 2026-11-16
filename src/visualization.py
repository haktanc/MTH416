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


def plot_feature_benchmark(benchmarks: list[dict], decisions: list[dict], path: Path) -> None:
    """Show measured training-only costs and explicitly projected cache size."""
    plt = _pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    names = [row["method"] for row in benchmarks]
    axes[0].bar(names, [row["feature_seconds"] for row in benchmarks], label="Descriptor extraction")
    axes[0].bar(names, [row["decode_normalize_seconds"] for row in benchmarks],
                bottom=[row["feature_seconds"] for row in benchmarks], label="Decode + normalize")
    axes[0].set(title="50 training images: measured times", ylabel="Seconds")
    axes[0].legend(fontsize=8)
    bars = axes[1].bar(names, [row["projected_payload_bytes"] / 1024**2 for row in decisions])
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set(title="1,300 images: projected float32 payload", ylabel="MiB (headers excluded)")
    axes[1].set_ylim(0, max(row["projected_payload_bytes"] / 1024**2 for row in decisions) * 1.18)
    figure.suptitle("CPU feature extraction: resource comparison, not model performance")
    _save_eda(figure, path)


def plot_texture_features(samples: list[dict], directory: Path) -> list[dict]:
    """Compare native HOG renderings and GLCM probabilities/properties in training."""
    from src.feature_extraction import GLCM_PROPERTIES, glcm_matrix

    _training_only(samples)
    if any(sample["image"].shape != (512, 512) for sample in samples):
        raise ValueError("Texture figures require native 512 x 512 arrays.")
    plt = _pyplot()
    ordered = sorted(samples, key=lambda r: CLASS_ORDER.index(r["class_name"]))
    figure, axes = plt.subplots(len(ordered), 3, figsize=(9, 4.3 * len(ordered)), squeeze=False)
    for row_index, sample in enumerate(ordered):
        image = sample["image"]
        renderings = [extract_hog(image, FeatureConfig(pixels_per_cell=(cell, cell)), visualize=True)[1]
                      for cell in (8, 16)]
        common_max = max(float(np.percentile(values, 99)) for values in renderings) or 1.0
        for column, values in enumerate([image, *renderings]):
            axes[row_index, column].imshow(values, cmap="gray", vmin=0,
                                          vmax=1 if column == 0 else common_max,
                                          interpolation="nearest")
            title = "Normalized 512 x 512" if column == 0 else f"HOG: {8 if column == 1 else 16} x {8 if column == 1 else 16} px/cell"
            axes[row_index, column].set_title(f"{CLASS_TITLES[sample['class_name']]}: {title}")
            axes[row_index, column].axis("off")
    figure.suptitle("Training-only HOG renderings at native image resolution")
    figure.text(0.5, 0.012, "HOG renderings share a per-row display range; these images are not the flattened descriptors.",
                ha="center", fontsize=9)
    _save_eda(figure, directory / "feature_hog_comparison.png")

    matrices = [glcm_matrix(sample["image"]) for sample in ordered]
    maximum = max(float(values[:, :, 0, :].max()) for values in matrices)
    figure, axes = plt.subplots(len(ordered), 4, figsize=(10.5, 3.3 * len(ordered)), squeeze=False)
    for row_index, (sample, matrix) in enumerate(zip(ordered, matrices)):
        for angle_index, degrees in enumerate((0, 45, 90, 135)):
            axis = axes[row_index, angle_index]
            axis.imshow(matrix[:, :, 0, angle_index], cmap="viridis", vmin=0, vmax=maximum,
                        origin="lower", interpolation="nearest")
            axis.set(title=f"{CLASS_TITLES[sample['class_name']]}: {degrees} degrees",
                     xlabel="Neighbor gray level", ylabel="Reference gray level")
    figure.suptitle("GLCM probabilities: 32 gray levels, distance 1 pixel")
    figure.text(0.5, 0.012, f"Shared probability color range: 0 to {maximum:.4f}; each symmetric matrix sums to one.",
                ha="center", fontsize=9)
    _save_eda(figure, directory / "feature_glcm_matrices.png")

    properties = [glcm_properties(sample["image"]) for sample in ordered]
    figure, axes = plt.subplots(2, 3, figsize=(11, 7))
    records = []
    for axis, name in zip(axes.flat, GLCM_PROPERTIES):
        for sample, values in zip(ordered, properties):
            for distance_index, distance in enumerate((1, 2)):
                axis.plot([0, 45, 90, 135], values[name][distance_index],
                          color=CLASS_COLORS[sample["class_name"]],
                          linestyle="-" if distance == 1 else "--", marker="o", markersize=3,
                          label=f"{CLASS_TITLES[sample['class_name']]}: d={distance}")
                for angle_index, degrees in enumerate((0, 45, 90, 135)):
                    records.append({"image_path": sample["image_path"], "class_name": sample["class_name"],
                                    "split": "train", "property": name, "distance": distance,
                                    "angle_degrees": degrees, "value": float(values[name][distance_index, angle_index])})
        axis.set(title=name.upper() if name == "ASM" else name.capitalize(),
                 xlabel="Angle (degrees)", ylabel="Property value", xticks=[0, 45, 90, 135])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
                  ncol=4, fontsize=8, frameon=False)
    figure.suptitle("GLCM properties for one reviewed training image per class")
    figure.text(0.5, 0.012, "Illustrative samples only; curves do not establish class separation or classification performance.",
                ha="center", fontsize=9)
    _save_eda(figure, directory / "feature_glcm_properties.png")
    return records
