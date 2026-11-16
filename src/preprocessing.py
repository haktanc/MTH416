"""Stateless grayscale resizing and per-image intensity normalization."""

import cv2
import numpy as np


DEFAULT_IMAGE_SIZE = (512, 512)


def validate_grayscale(image: np.ndarray) -> np.ndarray:
    """Require a finite, nonempty two-dimensional real numeric array."""
    values = np.asarray(image)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Expected a nonempty two-dimensional grayscale image.")
    if not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise ValueError("Image pixels must have a real numeric dtype.")
    if not np.isfinite(values).all():
        raise ValueError("Image must contain finite pixel values.")
    return values


def validate_size(size: tuple[int, int]) -> tuple[int, int]:
    """Validate dimensions in (height, width) order, excluding booleans."""
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("Size must contain a positive integer height and width.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer)) or value <= 0
        for value in size
    ):
        raise ValueError("Image dimensions must be positive integers.")
    return int(size[0]), int(size[1])


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """Use only this image's extrema; return float32 in [0, 1].

    Constant images become all-zero arrays. Float64 intermediate arithmetic
    avoids integer subtraction overflow and premature float32 rounding.
    """
    values = validate_grayscale(image).astype(np.float64)
    minimum, maximum = float(values.min()), float(values.max())
    if maximum == minimum:
        return np.zeros(values.shape, dtype=np.float32)
    span = maximum - minimum
    if np.isfinite(span):
        normalized = (values - minimum) / span
    else:
        # Opposite extreme finite float64 values can overflow their span.
        scale = max(abs(minimum), abs(maximum))
        scaled = values / scale
        normalized = (scaled - minimum / scale) / (
            maximum / scale - minimum / scale
        )
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def resize_grayscale(
    image: np.ndarray, size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return float32 pixels, optionally resizing to an explicit target.

    With no target, preserve the input dimensions. Matching dimensions also
    bypass resampling. An explicit different target uses area interpolation
    for downsampling in both dimensions, otherwise bilinear interpolation.
    No intensity normalization is performed here.
    """
    values = validate_grayscale(image)
    height, width = values.shape if size is None else validate_size(size)
    with np.errstate(over="ignore"):
        pixels = values.astype(np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError("Resize input must be representable as finite float32.")
    if (height, width) == values.shape:
        return pixels
    minimum, maximum = float(pixels.min()), float(pixels.max())
    if minimum == maximum:
        return np.full((height, width), minimum, dtype=np.float32)
    interpolation = (
        cv2.INTER_AREA
        if height <= values.shape[0] and width <= values.shape[1]
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        pixels, (width, height), interpolation=interpolation
    )
    if not np.isfinite(resized).all():
        raise ValueError("Resizing produced nonfinite pixel values.")
    np.clip(resized, minimum, maximum, out=resized)
    return resized.astype(np.float32, copy=False)


def prepare_image(
    image: np.ndarray, size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Normalize at native dimensions unless an explicit size is supplied.

    Project images are 512 x 512. Min-Max normalization alone always preserves
    the input shape; only an explicit different size permits resampling.
    """
    if size is None:
        return min_max_normalize(image)
    return min_max_normalize(resize_grayscale(image, size))
