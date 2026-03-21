# KA Agents - Development & Testing Commands
# Issue 7.4: Evaluation harness wired into CI

.PHONY: help install test test-unit test-smoke test-eval test-all test-ci coverage lint clean

PYTHON = python3
PYTEST = python -m pytest

help:
	@echo "KA Agents - Available Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run unit tests (default)"
	@echo "  make test-unit    Run unit tests only"
	@echo "  make test-smoke   Run quick smoke tests"
	@echo "  make test-eval    Run evaluation harness tests"
	@echo "  make test-all     Run all test suites"
	@echo "  make test-ci      Run tests in CI mode (fail fast)"
	@echo "  make coverage     Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linter checks"
	@echo "  make syntax       Check Python syntax"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove generated files"
	@echo ""

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Default test target
test: test-unit

test-unit:
	@echo "🧪 Running unit tests..."
	$(PYTEST) tests/ -v --tb=short

test-smoke:
	@echo "🔥 Running smoke tests..."
	$(PYTHON) scripts/run_tests.py --smoke

test-eval:
	@echo "📊 Running evaluation tests..."
	$(PYTHON) scripts/run_tests.py --eval

test-all:
	@echo "🚀 Running all tests..."
	$(PYTHON) scripts/run_tests.py

test-ci:
	@echo "🤖 Running tests in CI mode..."
	$(PYTHON) scripts/run_tests.py --ci

coverage:
	@echo "📈 Running tests with coverage..."
	$(PYTEST) tests/ --cov=core --cov=agents --cov=utils --cov-report=term-missing --cov-report=html

lint:
	@echo "🔍 Running linter..."
	ruff check . || true

syntax:
	@echo "✅ Checking Python syntax..."
	$(PYTHON) -m py_compile tests/*.py
	$(PYTHON) -m py_compile scripts/*.py

clean:
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf tests/__pycache__ core/__pycache__ agents/__pycache__ utils/__pycache__
	rm -f test-results.xml eval-results-*.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
