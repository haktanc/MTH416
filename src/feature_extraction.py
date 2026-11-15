"""Configurable HOG and GLCM texture descriptors."""

from dataclasses import dataclass
from math import pi

import numpy as np


GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy",
                   "correlation", "ASM")


@dataclass(frozen=True)
class FeatureConfig:
    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (8, 8)
    cells_per_block: tuple[int, int] = (2, 2)
    glcm_levels: int = 32
    distances: tuple[int, ...] = (1, 2)
    angles: tuple[float, ...] = (0.0, pi / 4, pi / 2, 3 * pi / 4)


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("Expected a finite, nonempty grayscale image.")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("Feature input must be normalized to [0, 1].")


def extract_hog(image: np.ndarray, config: FeatureConfig = FeatureConfig(),
                visualize: bool = False):
    """Return a HOG vector, optionally paired with its visualization."""
    from skimage.feature import hog

    _validate_image(image)
    return hog(image, orientations=config.orientations,
               pixels_per_cell=config.pixels_per_cell,
               cells_per_block=config.cells_per_block, block_norm="L2-Hys",
               visualize=visualize, feature_vector=True)


def glcm_properties(
    image: np.ndarray,
    config: FeatureConfig = FeatureConfig(),
) -> dict[str, np.ndarray]:
    """Calculate each property for every configured distance and angle."""
    from skimage.feature import graycomatrix, graycoprops

    _validate_image(image)
    if not 2 <= config.glcm_levels <= 256:
        raise ValueError("GLCM levels must be between 2 and 256.")
    quantized = np.minimum((image * config.glcm_levels).astype(np.uint16),
                           config.glcm_levels - 1)
    matrix = graycomatrix(quantized, distances=config.distances,
                         angles=config.angles, levels=config.glcm_levels,
                         symmetric=True, normed=True)
    return {name: graycoprops(matrix, name) for name in GLCM_PROPERTIES}


def extract_glcm(image: np.ndarray,
                 config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Flatten properties in property, distance, angle order."""
    properties = glcm_properties(image, config)
    return np.concatenate([
        properties[name].ravel() for name in GLCM_PROPERTIES
    ])


def extract_features(
    image: np.ndarray,
    config: FeatureConfig = FeatureConfig(),
    method: str = "combined",
) -> np.ndarray:
    """Extract HOG, GLCM, or both in a stable feature order."""
    if method == "hog":
        vector = extract_hog(image, config)
    elif method == "glcm":
        vector = extract_glcm(image, config)
    elif method == "combined":
        vector = np.concatenate((
            extract_hog(image, config), extract_glcm(image, config)
        ))
    else:
        raise ValueError(
            "Feature method must be 'hog', 'glcm', or 'combined'."
        )
    if not np.isfinite(vector).all():
        raise ValueError("Feature vector contains non-finite values.")
    return vector.astype(np.float32)
