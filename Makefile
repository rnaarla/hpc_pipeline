.PHONY: help install dev devshell test test-unit test-integration test-gpu coverage lint format clean validate

VENV ?= .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
DEPS_STAMP := $(VENV)/.deps.installed

help:
	@echo "HPC Pipeline Development Commands"
	@echo "=================================="
	@echo "install          - Bootstrap virtualenv and install dependencies"
	@echo "devshell         - Spawn a shell with the virtualenv activated"
	@echo "test             - Run all tests"
	@echo "test-unit        - Run unit tests only"
	@echo "test-integration - Run integration tests only"
	@echo "test-gpu         - Run GPU tests only"
	@echo "coverage         - Run tests with coverage report"
	@echo "lint             - Run linters (flake8, mypy, pylint)"
	@echo "format           - Format code with black and isort"
	@echo "clean            - Clean build artifacts"
	@echo "validate         - Run formatting, linting, and CPU-only tests"

$(PYTHON):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip wheel setuptools

$(DEPS_STAMP): requirements.txt requirements-dev.txt | $(PYTHON)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	if [ -n "$$CUDA_HOME" ]; then \
		echo "Building CUDA extensions with CUDA_HOME=$$CUDA_HOME"; \
		$(PYTHON) setup.py build_ext --inplace; \
	else \
		echo "Skipping CUDA extension build (set CUDA_HOME to enable)"; \
	fi
	touch $(DEPS_STAMP)

install: $(DEPS_STAMP)

dev: install
	@echo "Virtual environment ready at $(VENV)"

devshell: install
	@echo "Activating virtual environment $(VENV)..."
	@. $(VENV)/bin/activate && exec $$SHELL -i

validate: install
	$(PYTHON) -m black --check .
	$(PYTHON) -m isort --check-only .
	$(PYTHON) -m flake8 .
	$(PYTHON) -m mypy --ignore-missing-imports .
	if command -v terraform >/dev/null 2>&1; then terraform fmt -check -recursive; else echo "terraform not installed, skipping fmt check"; fi
	if command -v helm >/dev/null 2>&1; then helm lint infra/helm/hpc-pipeline --strict; else echo "helm not installed, skipping chart lint"; fi
	$(PYTHON) -m pytest tests/unit -m "not gpu"

lint: install
	$(PYTHON) -m flake8 optimization distributed monitoring --max-line-length=100
	$(PYTHON) -m mypy optimization distributed monitoring --ignore-missing-imports
	$(PYTHON) -m pylint optimization distributed monitoring --max-line-length=100 || true

format: install
	$(PYTHON) -m black optimization distributed monitoring tests
	$(PYTHON) -m isort optimization distributed monitoring tests

test: install
	$(PYTHON) -m pytest tests -v

test-unit: install
	$(PYTHON) -m pytest tests/unit -v -m "not gpu"

test-integration: install
	$(PYTHON) -m pytest tests -v -m integration

test-gpu: install
	$(PYTHON) -m pytest tests -v -m gpu

coverage: install
	$(PYTHON) -m pytest tests --cov=. --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

clean:
	rm -rf build dist *.egg-info htmlcov .coverage .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.so" ! -path "$(VENV)/*" -delete
	rm -f $(DEPS_STAMP)
