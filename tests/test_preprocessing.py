"""Test grayscale preprocessing arithmetic, dimensions, types, and edge cases."""

import numpy as np
import pytest

from src.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    min_max_normalize,
    prepare_image,
    resize_grayscale,
    validate_size,
)


def test_min_max_formula_uses_this_image_only():
    image = np.array([[10, 20], [30, 50]], dtype=np.uint8)
    before = image.copy()
    expected = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(min_max_normalize(image), expected)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(
        min_max_normalize(image.astype(np.float64) * 2 + 100), expected
    )


@pytest.mark.parametrize("value", [0, 127, 255, -20, 0.125])
@pytest.mark.parametrize("size", [(128, 128), (7, 13), (600, 513)])
def test_constant_images_remain_zero_after_any_resize(value, size):
    image = np.full((512, 512), value, dtype=np.float64)
    result = prepare_image(image, size)
    assert result.shape == size
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert np.count_nonzero(result) == 0


@pytest.mark.parametrize("size", [(16, 24), (64, 48), (1, 1), (8, 80)])
def test_dimensions_finite_float32_and_bounds(size):
    image = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    result = prepare_image(image, size)
    assert result.shape == size
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert 0.0 <= result.min() <= result.max() <= 1.0


def test_resize_retains_original_intensity_units():
    image = np.array([[0, 0, 100, 100], [0, 0, 100, 100]], dtype=np.uint8)
    resized = resize_grayscale(image, (1, 2))
    np.testing.assert_array_equal(resized, [[0.0, 100.0]])
    assert resized.dtype == np.float32
    np.testing.assert_array_equal(prepare_image(image, (1, 2)), [[0.0, 1.0]])


def test_resizing_precedes_normalization():
    image = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    # Area averaging to one pixel is constant and therefore normalizes to zero.
    np.testing.assert_array_equal(prepare_image(image, (1, 1)), [[0.0]])


def test_float64_near_constant_values_do_not_collapse_early():
    image = np.array([[1.0, 1.0 + 1e-10]], dtype=np.float64)
    np.testing.assert_array_equal(min_max_normalize(image), [[0.0, 1.0]])


def test_extreme_finite_range_does_not_overflow_normalization():
    extreme = np.finfo(np.float64).max
    image = np.array([[-extreme, 0.0, extreme]])
    np.testing.assert_array_equal(min_max_normalize(image), [[0.0, 0.5, 1.0]])


def test_readonly_input_is_not_mutated():
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    original = image.copy()
    image.setflags(write=False)
    prepare_image(image, (4, 6))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("image", [
    np.empty((0, 2)), np.array([1.0, 2.0]), np.zeros((2, 2, 3)),
    np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-np.inf]]),
    np.array([[True]]), np.array([["pixel"]]), np.array([[1 + 2j]]),
])
def test_invalid_grayscale_inputs_are_rejected(image):
    with pytest.raises(ValueError):
        min_max_normalize(image)
    with pytest.raises(ValueError):
        prepare_image(image)


@pytest.mark.parametrize("size", [
    (0, 10), (-1, 10), (3.5, 4), (True, 5), (4,), (3, 4, 5), "128",
])
def test_invalid_resize_configuration(size):
    with pytest.raises(ValueError):
        validate_size(size)
    with pytest.raises(ValueError):
        prepare_image(np.zeros((4, 4)), size)


def test_resize_rejects_unrepresentable_float32_values():
    with pytest.raises(ValueError, match="finite float32"):
        resize_grayscale(np.array([[0.0, 1e100]]), (1, 2))


def test_native_512_defaults_do_not_call_spatial_resampling(monkeypatch):
    import src.preprocessing as preprocessing

    assert DEFAULT_IMAGE_SIZE == (512, 512)
    image = np.arange(512 * 512, dtype=np.uint8).reshape(512, 512)
    before = image.copy()

    def forbidden_resize(*args, **kwargs):
        raise AssertionError("Default processing must not call cv2.resize")

    monkeypatch.setattr(preprocessing.cv2, "resize", forbidden_resize)
    for target in (None, DEFAULT_IMAGE_SIZE):
        processed = prepare_image(image, target)
        assert processed.shape == DEFAULT_IMAGE_SIZE
        assert processed.dtype == np.float32
        np.testing.assert_array_equal(processed, min_max_normalize(image))
        np.testing.assert_array_equal(resize_grayscale(image, target), image)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("shape", [(512, 512), (73, 121), (1, 5)])
def test_normalization_and_no_target_preparation_preserve_dimensions(shape):
    image = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    assert min_max_normalize(image).shape == shape
    assert prepare_image(image).shape == shape
    assert resize_grayscale(image).shape == shape
