"""Unit tests for kernel benchmarking."""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

@pytest.mark.unit
class TestKernelBenchmarkUnit:
    """Unit tests for KernelBenchmark class."""
    
    def test_get_prometheus_metrics(self, mock_kernel_benchmark):
        """Test metric name retrieval."""
        metrics = mock_kernel_benchmark.get_prometheus_metrics()
        assert isinstance(metrics, list)
        assert "kernel_tflops" in metrics
        
    def test_calculate_tflops(self):
        """Test TFLOPS calculation."""
        from optimization.kernel_benchmark import KernelBenchmark
        device = torch.device("cpu")
        kb = KernelBenchmark(device)
        
        # 2048x2048x2048 GEMM
        size = 2048
        time_ms = 10.0
        tflops = kb.calculate_tflops(size, time_ms)
        
        expected_flops = 2 * size * size * size
        expected_tflops = expected_flops / (time_ms * 1e-3) * 1e-12
        assert abs(tflops - expected_tflops) < 1e-6
        
    def test_get_peak_tflops_cpu(self):
        """Test peak TFLOPS estimation on CPU."""
        from optimization.kernel_benchmark import KernelBenchmark
        device = torch.device("cpu")
        kb = KernelBenchmark(device)
        
        # CPU should return non-zero peak
        peak = kb.get_peak_tflops()
        assert peak > 0
        
    @pytest.mark.gpu
    def test_benchmark_size_variants(self, device):
        """Test benchmarking with different input sizes."""
        from optimization.kernel_benchmark import KernelBenchmark
        kb = KernelBenchmark(device)
        
        # Test tuple size
        with patch.object(kb, 'benchmark_fused_ops') as mock_bench:
            mock_bench.return_value = {"tflops": 10.0}
            result = kb.benchmark_fused_ops(size=(1024, 2048))
            assert result["tflops"] > 0
