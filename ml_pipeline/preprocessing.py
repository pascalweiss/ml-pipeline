"""Data preprocessing utilities."""


def normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of values to [0, 1] range."""
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def remove_outliers(values: list[float], threshold: float = 2.0) -> list[float]:
    """Remove values more than `threshold` standard deviations from the mean."""
    if len(values) < 3:
        return values
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    if std == 0:
        return values
    return [v for v in values if abs(v - mean) / std <= threshold]


def standard_scale(values: list[float]) -> list[float]:
    """Apply z-score normalization to a list of values.

    Transforms each value by subtracting the mean and dividing by the
    standard deviation, resulting in a distribution with mean 0 and
    standard deviation 1.

    Args:
        values: List of numeric values to normalize.

    Returns:
        List of z-score normalized values.
    """
    if not values:
        return []
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    if std == 0:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]
