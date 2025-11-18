.PHONY: help install test test-unit test-integration test-gpu coverage lint format clean

# Default target
help:
	@echo "HPC Pipeline Development Commands"
	@echo "=================================="
	@echo "install          - Install development dependencies"
	@echo "test             - Run all tests"
	@echo "test-unit        - Run unit tests only"
	@echo "test-integration - Run integration tests only"
	@echo "test-gpu         - Run GPU tests only"
	@echo "coverage         - Run tests with coverage report"
	@echo "lint             - Run linters (flake8, mypy, pylint)"
	@echo "format           - Format code with black and isort"
	@echo "clean            - Clean build artifacts"

install:
	python3.11 -m pip install -r requirements.txt
	python3.11 -m pip install -r requirements-dev.txt
	python3.11 setup.py build_ext --inplace

validate:
	python3.11 -m black --check .
	python3.11 -m isort --check-only .
	python3.11 -m flake8 .
	python3.11 -m mypy --ignore-missing-imports .
	if command -v terraform >/dev/null 2>&1; then terraform fmt -check -recursive; else echo "terraform not installed, skipping fmt check"; fi
	if command -v helm >/dev/null 2>&1; then helm lint infra/helm/hpc-pipeline --strict; else echo "helm not installed, skipping chart lint"; fi
	python3.11 -m pytest tests/unit -m "not gpu"

# Code quality
lint:
	python3.11 -m flake8 optimization distributed monitoring --max-line-length=100
	python3.11 -m mypy optimization distributed monitoring --ignore-missing-imports
	python3.11 -m pylint optimization distributed monitoring --max-line-length=100 || true

format:
	python3.11 -m black optimization distributed monitoring tests
	python3.11 -m isort optimization distributed monitoring tests

# Testing
test:
	python3.11 -m pytest tests -v

test-unit:
	python3.11 -m pytest tests/unit -v -m "not gpu"

test-integration:
	python3.11 -m pytest tests -v -m integration

test-gpu:
	python3.11 -m pytest tests -v -m gpu

coverage:
	python3.11 -m pytest tests --cov=. --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Cleanup
clean:
	rm -rf build dist *.egg-info htmlcov .coverage .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" -delete
