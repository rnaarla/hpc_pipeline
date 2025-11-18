"""Pytest configuration and shared fixtures."""
import pytest
import torch
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock
from prometheus_client import CollectorRegistry
from prometheus_client import registry as prometheus_registry
from prometheus_client import metrics as prometheus_metrics

# GPU availability check
def pytest_configure(config):
    """Configure pytest with custom markers and GPU detection."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    
    # Set environment for reproducibility
    os.environ["PYTHONHASHSEED"] = "0"
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def pytest_collection_modifyitems(config, items):
    """Skip GPU-marked tests when CUDA is unavailable."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()

@pytest.fixture
def device(cuda_available):
    """Return appropriate device for testing."""
    return torch.device("cuda" if cuda_available else "cpu")

@pytest.fixture
def mock_kernel_benchmark(monkeypatch):
    """Mock KernelBenchmark for CPU testing."""
    mock_bm = MagicMock()
    mock_bm.benchmark_fused_ops.return_value = {
        "speedup": 1.8,
        "roofline_efficiency": 0.74,
        "numerical_max_diff": 6e-4,
        "tflops": 12.5,
        "time_ms": 2.3,
    }
    mock_bm.get_prometheus_metrics.return_value = [
        "kernel_tflops",
        "kernel_memory_bandwidth_gbps",
        "kernel_sm_occupancy_pct",
    ]
    return mock_bm

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir

@pytest.fixture
def sample_config():
    """Sample training configuration."""
    return {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "checkpoint_dir": "./checkpoints",
        "use_amp": True,
        "gradient_accumulation_steps": 4,
    }

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set environment variables for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""
    os.environ["PYTHONPATH"] = "/Users/rnaarla8/code/hpc-pipeline"
    
    yield
    
    # Cleanup after tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def fresh_prometheus_registry(monkeypatch):
    """Provide a clean Prometheus registry to avoid metric duplication."""
    registry = CollectorRegistry()
    monkeypatch.setattr(prometheus_registry, "REGISTRY", registry)
    monkeypatch.setattr(prometheus_metrics, "REGISTRY", registry)
    return registry
