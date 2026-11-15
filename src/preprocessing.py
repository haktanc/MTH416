"""Prepare grayscale images for texture feature extraction."""

import cv2
import numpy as np


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Scale finite pixel values to [0, 1]; constant images become zero."""
    values = np.asarray(image, dtype=np.float32)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Image must contain finite pixel values.")
    minimum, maximum = float(values.min()), float(values.max())
    if maximum == minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def prepare_image(image: np.ndarray, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Resize to (height, width), then apply per-image Min-Max normalization."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a nonempty two-dimensional grayscale image.")
    if len(size) != 2 or any(value <= 0 for value in size):
        raise ValueError("Image dimensions must be positive.")
    resized = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return min_max_normalize(resized)
