# Testing Guide for KA Agents

This document describes how to run tests locally and understand the CI pipeline.

## Quick Start

```bash
# Install dependencies
make install

# Run all tests
make test-all

# Run unit tests only
make test

# Run smoke tests (fastest)
make test-smoke
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared pytest fixtures
├── test_langgraph_workflow.py  # Issue 7.1: LangGraph workflow tests
├── test_vector_retrieval.py    # Issue 7.2: Vector DB/retrieval tests
└── test_llm_resilience.py      # Issue 7.3: LLM failure mode tests
```

## Running Tests

### Using Make (Recommended)

| Command | Description |
|---------|-------------|
| `make test` | Run unit tests (default) |
| `make test-unit` | Run unit tests only |
| `make test-smoke` | Run quick smoke tests (~10 seconds) |
| `make test-eval` | Run evaluation harness |
| `make test-all` | Run all test suites |
| `make test-ci` | CI mode (fail fast, with coverage) |
| `make coverage` | Generate coverage report |

### Using pytest directly

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_langgraph_workflow.py -v

# Run specific test class
pytest tests/test_langgraph_workflow.py::TestRouterFunctions -v

# Run with coverage
pytest tests/ --cov=core --cov=agents --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"

# Run with timeout
pytest tests/ --timeout=30
```

### Using the test runner script

```bash
# Run all tests
python scripts/run_tests.py

# Run only unit tests
python scripts/run_tests.py --unit

# Run only evaluation tests
python scripts/run_tests.py --eval --test-suite quality

# CI mode
python scripts/run_tests.py --ci
```

## Test Categories

### Unit Tests (test_*.py)

Standard pytest unit tests covering:

- **Graph compilation**: Verifies LangGraph workflows compile correctly
- **Router functions**: Tests all router decision logic with fallbacks
- **State isolation**: Ensures concurrent runs don't share state
- **Error handling**: Tests error paths and exception handling
- **Vector operations**: Tests ChromaDB tenant isolation and caching
- **LLM resilience**: Tests timeout, rate limit, and fallback behavior

### Smoke Tests

Quick sanity checks that verify:
- Core imports work
- Configuration loads correctly
- Error handlers are available
- Evaluation harness initializes

### Evaluation Tests

Uses the evaluation harness (`core/evaluation_harness.py`) to run:

- **Quality tests**: Output format and completeness
- **Toxicity tests**: Content safety checks
- **Data leakage tests**: PII and input data isolation

## CI Pipeline

The CI workflow (`.github/workflows/test.yml`) runs automatically on:
- Push to `develop`, `main`, or `feature/**` branches
- Pull requests to `develop` or `main`

### Pipeline Stages

1. **Lint**: Code quality checks with Ruff
2. **Smoke Tests**: Quick validation (must pass to continue)
3. **Unit Tests**: Full pytest suite with coverage
4. **Evaluation Tests**: Harness tests for quality, toxicity, and leakage
5. **Summary**: Aggregated results report

### Artifacts

CI generates these artifacts (retained for 7 days):
- `test-results.xml`: JUnit XML test results
- `eval-results-*.json`: Evaluation harness reports
- Coverage reports uploaded to Codecov

## Writing Tests

### Test File Naming

- Test files: `test_<feature>.py`
- Test classes: `Test<Feature>`
- Test functions: `test_<behavior>`

### Using Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
def test_my_feature(minimal_state, mock_invoke_llm):
    """Use fixtures for common setup."""
    result = some_function(minimal_state)
    assert result["status"] == "success"
```

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

### Marking Tests

```python
import pytest

@pytest.mark.slow
def test_expensive_operation():
    """Skipped with: pytest -m 'not slow'"""
    pass

@pytest.mark.integration
def test_external_api():
    """Integration test requiring external services."""
    pass
```

## Coverage

Generate an HTML coverage report:

```bash
make coverage
open htmlcov/index.html
```

Coverage targets:
- `core/`: Core business logic
- `agents/`: Agent implementations
- `utils/`: Utility functions

## Troubleshooting

### Import Errors

Ensure PYTHONPATH includes the project root:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Timeout Issues

Increase timeout for slow tests:
```bash
pytest tests/ --timeout=120
```

### Missing Dependencies

Install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-timeout pytest-cov
```

### Mock Issues

If mocks aren't working, check the patch path matches the import location:
```python
# If file imports: from models.llm_invoker import invoke_llm
# Patch at: @patch('agents.my_agent.invoke_llm')  # Where it's used
# Not at: @patch('models.llm_invoker.invoke_llm')  # Where it's defined
```

## Related Issues

- Issue 7.1: LangGraph workflow tests
- Issue 7.2: Vector retrieval tests
- Issue 7.3: LLM failure mode tests
- Issue 7.4: CI integration (this setup)
