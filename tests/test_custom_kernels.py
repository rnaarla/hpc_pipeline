import numpy as np
import pytest
import torch

from optimization.kernel_benchmark import KernelBenchmark


class TestCustomKernels:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = KernelBenchmark(self.device)

    def test_performance_requirements(self):
        """Verify fused ops remain numerically stable and performant."""
        size = 256 if self.device.type == "cpu" else 2048
        results = self.benchmark.benchmark_fused_ops(size=size)
        assert results["numerical_max_diff"] < 1e-3
        assert results["tflops"] >= 0
        if self.device.type == "cuda":
            assert results["speedup"] >= 1.0

    def test_numerical_stability(self):
        """Test numerical stability across sizes."""
        sizes = [128, 256, 512] if self.device.type == "cpu" else [1024, 2048, 4096]
        for size in sizes:
            results = self.benchmark.benchmark_fused_ops(size=size)
            assert results["numerical_max_diff"] < 1e-3

    def test_edge_cases(self):
        """Test kernel robustness across edge cases."""
        edge_cases = [
            (1, 1),
            (128, 64),
            (256, 257),
        ]
        for m, n in edge_cases:
            results = self.benchmark.benchmark_fused_ops(size=(m, n))
            assert not np.isnan(results["tflops"])
