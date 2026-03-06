# Contributing to ML Pipeline

## Code Style

- Use type hints for all function signatures
- Follow PEP 8 conventions
- Maximum line length: 100 characters (enforced by ruff)
- Use docstrings for all public functions (Google style)

## Testing

- All new features must have tests in the `tests/` directory
- Use pytest fixtures for shared test data
- Test edge cases: empty inputs, single elements, large datasets

## Architecture

- Keep preprocessing functions in `ml_pipeline/preprocessing.py`
- Each function should be pure (no side effects)
- Prefer list comprehensions over loops for simple transformations
