# Contributing to expectllm

Thank you for your interest in contributing!

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dev dependencies: `pip install -e ".[dev]"`

## Running Tests

```bash
# Unit tests (no API key needed)
pytest tests/ --ignore=tests/test_integration.py

# With coverage
pytest tests/ --cov=expectllm --cov-report=term-missing

# Integration tests (requires ANTHROPIC_API_KEY or OPENAI_API_KEY)
pytest tests/test_integration.py
```

## Code Style

We use ruff for linting and formatting:

```bash
ruff check src/
ruff format src/
```

## Type Checking

```bash
mypy src/
```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a PR with a clear description

## Constraints

Remember our core constraints:
- Total code must be < 600 lines (excluding tests)
- No dependencies beyond `anthropic` and `openai`
- Must work with Python 3.9+
