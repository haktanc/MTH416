"""Check descriptor geometry and texture mathematics without fitting models."""

from dataclasses import replace

import numpy as np
import pytest

from src.feature_extraction import (
    FeatureConfig, extract_features, extract_glcm, extract_hog, glcm_matrix,
    glcm_properties, hog_feature_length, quantize_grayscale,
)


@pytest.mark.parametrize("cell,dimension", [(8, 142884), (16, 34596)])
def test_native_hog_geometry_and_block_normalization(cell, dimension):
    image = np.tile(np.linspace(0, 1, 512, dtype=np.float32), (512, 1))
    config = FeatureConfig(pixels_per_cell=(cell, cell))
    before = image.copy()
    vector, rendering = extract_hog(image, config, visualize=True)
    assert hog_feature_length(image.shape, config) == dimension
    assert vector.shape == (dimension,) and vector.dtype == np.float32
    assert rendering.shape == (512, 512)
    assert np.isfinite(vector).all() and (vector >= 0).all()
    np.testing.assert_allclose(np.linalg.norm(vector.reshape(-1, 36), axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("cell", [8, 16])
def test_constant_hog_has_zero_gradient(cell):
    vector = extract_hog(np.full((64, 64), 0.5), FeatureConfig(pixels_per_cell=(cell, cell)))
    assert vector.dtype == np.float32
    np.testing.assert_array_equal(vector, np.zeros_like(vector))


def test_quantization_boundaries_and_input_preservation():
    image = np.array([[0, 1/32, 0.5, 1]], dtype=np.float32)
    image.flags.writeable = False
    quantized = quantize_grayscale(image)
    np.testing.assert_array_equal(quantized, [[0, 1, 16, 31]])
    assert quantized.dtype == np.uint8
    np.testing.assert_array_equal(image, [[0, 1/32, 0.5, 1]])


def test_glcm_hand_computed_alternating_stripes_and_distances():
    image = np.tile([0.0, 1.0], (32, 16)).astype(np.float32)
    config = FeatureConfig(glcm_levels=2, angles=(0.0, np.pi/2))
    matrix = glcm_matrix(image, config)
    assert matrix.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(matrix.sum(axis=(0, 1)), 1)
    np.testing.assert_array_equal(matrix, matrix.transpose(1, 0, 2, 3))
    np.testing.assert_array_equal(matrix[:, :, 0, 0], [[0, 0.5], [0.5, 0]])
    np.testing.assert_array_equal(matrix[:, :, 1, 0], [[0.5, 0], [0, 0.5]])
    properties = glcm_properties(image, config)
    expected = {"contrast": 1, "dissimilarity": 1, "homogeneity": 0.5,
                "energy": np.sqrt(0.5), "correlation": -1, "ASM": 0.5}
    for name, value in expected.items():
        assert properties[name][0, 0] == pytest.approx(value)
    assert properties["contrast"][1, 0] == 0
    assert properties["correlation"][1, 0] == pytest.approx(1)
    assert properties["homogeneity"][1, 0] == 1


def test_default_glcm_dimension_order_and_constant_conventions():
    image = np.zeros((64, 64), dtype=np.float32)
    vector = extract_glcm(image)
    assert vector.shape == (48,) and vector.dtype == np.float32
    # Property-major: contrast/dissimilarity are zero; the other four are one.
    np.testing.assert_array_equal(vector[:16], 0)
    np.testing.assert_array_equal(vector[16:], 1)
    combined = extract_features(image)
    assert combined.size == hog_feature_length(image.shape) + 48
    np.testing.assert_array_equal(combined[-48:], vector)


@pytest.mark.parametrize("image", [
    np.zeros((2, 2, 3)), np.empty((0, 2)), np.array([[np.nan]]),
    np.array([[np.inf]]), np.array([[-0.01]]), np.array([[1.01]]),
    np.array([[1j]]), np.array([[True]]), np.array([["text"]]),
])
def test_invalid_feature_inputs_are_rejected(image):
    for function in (extract_hog, extract_glcm):
        with pytest.raises(ValueError):
            function(image)


@pytest.mark.parametrize("changes", [
    {"pixels_per_cell": (0, 8)}, {"cells_per_block": (2,)},
    {"orientations": 0}, {"orientations": True}, {"glcm_levels": 1},
    {"glcm_levels": 257}, {"distances": (0,)}, {"angles": (np.nan,)},
    {"block_norm": "L1"},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(FeatureConfig(), **changes)


def test_small_images_and_unknown_methods_fail_explicitly():
    with pytest.raises(ValueError, match="small"):
        extract_hog(np.zeros((8, 8)))
    with pytest.raises(ValueError, match="method"):
        extract_features(np.zeros((32, 32)), method="invalid")
