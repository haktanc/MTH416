"""Configurable HOG and GLCM texture descriptors."""

from dataclasses import dataclass
from math import pi

import numpy as np

from src.preprocessing import validate_grayscale


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
    block_norm: str = "L2-Hys"

    def __post_init__(self):
        for name in ("pixels_per_cell", "cells_per_block"):
            value = getattr(self, name)
            if len(value) != 2 or any(
                isinstance(v, (bool, np.bool_))
                or not isinstance(v, (int, np.integer)) or v < 1 for v in value
            ):
                raise ValueError(f"{name} must contain two positive integers.")
        if isinstance(self.orientations, bool) or not isinstance(self.orientations, int) or self.orientations < 1:
            raise ValueError("orientations must be a positive integer.")
        if isinstance(self.glcm_levels, bool) or not isinstance(self.glcm_levels, int) or not 2 <= self.glcm_levels <= 256:
            raise ValueError("GLCM levels must be between 2 and 256.")
        if not self.distances or any(
            isinstance(d, bool) or not isinstance(d, int) or d <= 0
            for d in self.distances
        ):
            raise ValueError("GLCM distances must be positive integers.")
        if not self.angles or not np.isfinite(self.angles).all():
            raise ValueError("GLCM angles must be finite radians.")
        if self.block_norm != "L2-Hys":
            raise ValueError("The project uses L2-Hys HOG block normalization.")


def _validate_image(image: np.ndarray) -> np.ndarray:
    values = validate_grayscale(image)
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Feature input must be normalized to [0, 1].")
    return values.astype(np.float32, copy=False)


def hog_feature_length(shape: tuple[int, int], config: FeatureConfig = FeatureConfig()) -> int:
    """Count blocks in scikit-image's flattened descriptor (partial cells omitted)."""
    if len(shape) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in shape):
        raise ValueError("HOG requires a positive two-dimensional shape.")
    blocks = [s // cell - block + 1 for s, cell, block in zip(
        shape, config.pixels_per_cell, config.cells_per_block
    )]
    if min(blocks) < 1:
        raise ValueError("Image is too small for the configured HOG block.")
    return int(np.prod(blocks) * np.prod(config.cells_per_block) * config.orientations)


def extract_hog(image: np.ndarray, config: FeatureConfig = FeatureConfig(),
                visualize: bool = False):
    """Return a HOG vector, optionally paired with its visualization."""
    from skimage.feature import hog

    values = _validate_image(image)
    hog_feature_length(values.shape, config)
    result = hog(values, orientations=config.orientations,
               pixels_per_cell=config.pixels_per_cell,
               cells_per_block=config.cells_per_block, block_norm=config.block_norm,
               visualize=visualize, feature_vector=True, transform_sqrt=False,
               channel_axis=None)
    if visualize:
        vector, rendering = result
        return vector.astype(np.float32, copy=False), rendering.astype(np.float32, copy=False)
    return result.astype(np.float32, copy=False)


def quantize_grayscale(image: np.ndarray, levels: int = 32) -> np.ndarray:
    """Map [0, 1] to floor(image * levels), clipping endpoint 1 to levels - 1."""
    FeatureConfig(glcm_levels=levels)
    values = _validate_image(image)
    return np.minimum((values * levels).astype(np.uint16), levels - 1).astype(np.uint8)


def glcm_matrix(image: np.ndarray, config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Return a symmetric probability matrix for each distance/angle pair."""
    from skimage.feature import graycomatrix

    quantized = quantize_grayscale(image, config.glcm_levels)
    return graycomatrix(quantized, distances=config.distances,
                        angles=config.angles, levels=config.glcm_levels,
                        symmetric=True, normed=True)


def glcm_properties(
    image: np.ndarray,
    config: FeatureConfig = FeatureConfig(),
) -> dict[str, np.ndarray]:
    """Calculate each property for every configured distance and angle."""
    from skimage.feature import graycoprops

    matrix = glcm_matrix(image, config)
    return {name: graycoprops(matrix, name) for name in GLCM_PROPERTIES}


def extract_glcm(image: np.ndarray,
                 config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Flatten properties in property, distance, angle order."""
    properties = glcm_properties(image, config)
    return np.concatenate([
        properties[name].ravel() for name in GLCM_PROPERTIES
    ]).astype(np.float32)


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
