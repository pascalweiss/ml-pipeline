"""Tests for preprocessing module."""

from ml_pipeline.preprocessing import normalize, remove_outliers, standard_scale


def test_normalize_basic():
    result = normalize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_normalize_empty():
    assert normalize([]) == []


def test_normalize_single():
    assert normalize([42.0]) == [0.0]


def test_remove_outliers_basic():
    values = [1.0, 2.0, 3.0, 100.0]
    result = remove_outliers(values, threshold=1.5)
    assert 100.0 not in result
    assert 1.0 in result


def test_remove_outliers_no_outliers():
    values = [1.0, 2.0, 3.0]
    result = remove_outliers(values, threshold=3.0)
    assert len(result) == 3


def test_standard_scale_basic():
    result = standard_scale([1.0, 2.0, 3.0, 4.0, 5.0])
    # Mean = 3.0, Std = sqrt(2.0) ≈ 1.414
    assert len(result) == 5
    assert abs(result[0] - (-1.414)) < 0.01
    assert abs(result[2] - 0.0) < 0.01
    assert abs(result[4] - 1.414) < 0.01


def test_standard_scale_empty():
    assert standard_scale([]) == []


def test_standard_scale_single():
    result = standard_scale([42.0])
    assert result == [0.0]


def test_standard_scale_uniform():
    result = standard_scale([5.0, 5.0, 5.0])
    assert result == [0.0, 0.0, 0.0]
