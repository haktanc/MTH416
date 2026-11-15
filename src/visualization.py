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


def _save(figure, path: Path) -> None:
    plt = _pyplot()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
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
