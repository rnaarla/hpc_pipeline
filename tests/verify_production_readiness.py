import pytest
import torch

from optimization.kernel_benchmark import KernelBenchmark
from monitoring.kernel_monitor import KernelMonitor

try:
    import custom_kernels
    _HAS_CUSTOM_KERNEL = hasattr(custom_kernels, "fused_matmul_bias_gelu_dropout")
except Exception:  # pragma: no cover - custom build not present
    _HAS_CUSTOM_KERNEL = False

_SKIP_REASON = None
if not torch.cuda.is_available():
    _SKIP_REASON = "Production readiness suite requires CUDA hardware."
elif not _HAS_CUSTOM_KERNEL:
    _SKIP_REASON = "Custom CUDA kernels not built; skipping production readiness checks."

if _SKIP_REASON:
    pytestmark = pytest.mark.skip(reason=_SKIP_REASON)


@pytest.mark.production
class TestProductionReadiness:
    """Comprehensive production readiness validation."""
    
    @pytest.fixture(scope="class", autouse=True)
    def bm(self):
        device = torch.device("cuda")
        return KernelBenchmark(device)

    def test_complete_feature_set(self):
        """Verify all required features are implemented."""
        benchmark = KernelBenchmark(torch.device("cuda"))
        monitor = KernelMonitor()
        
        required_features = [
            "tensor_core_support",
            "memory_optimization",
            "error_handling",
            "performance_monitoring",
            "fault_tolerance"
        ]
        
        for feature in required_features:
            assert hasattr(benchmark, f"validate_{feature}")

    def test_performance_requirements(self, bm):
        """Phase 4: speedup, roofline efficiency, numerical tolerance."""
        results = bm.benchmark_fused_ops(size=2048)
        assert results["speedup"] >= 1.5
        assert results["roofline_efficiency"] >= 0.7
        assert results["numerical_max_diff"] < 1e-3

    @pytest.mark.parametrize("size", [1024, 2048, 4096])
    def test_numerical_stability(self, bm, size):
        """Phase 4: numerical stability across sizes."""
        results = bm.benchmark_fused_ops(size=size)
        assert results["numerical_max_diff"] < 1e-3

    def test_metrics_exported(self, bm):
        """Metrics presence check (observability)."""
        names = set(bm.get_prometheus_metrics())
        assert "kernel_tflops" in names
        assert "kernel_efficiency_roofline_pct" in names
        assert "kernel_sm_occupancy_pct" in names

    def test_monitoring_integration(self):
        """Verify monitoring stack integration."""
        monitor = KernelMonitor()
        metrics = monitor.collect_metrics()
        
        required_metrics = [
            "kernel_latency",
            "memory_bandwidth",
            "tensor_core_usage",
            "error_counters"
        ]
        
        for required_metric in required_metrics:
            assert required_metric in metrics
            assert metrics[required_metric] is not None
            
    def test_fault_recovery(self):
        """Verify fault handling and recovery."""
        benchmark = KernelBenchmark(torch.device("cuda"))
        
        failure_scenarios = [
            "oom_condition",
            "numerical_instability",
            "hardware_error"
        ]
        
        for scenario in failure_scenarios:
            with self.subTest(scenario=scenario):
                try:
                    if scenario == "oom_condition":
                        # Test OOM recovery
                        benchmark.benchmark_fused_ops(size=1000000)
                    elif scenario == "numerical_instability":
                        # Test numerical error handling
                        benchmark.validate_numerical_stability(None, None, -1)
                    elif scenario == "hardware_error":
                        # Test hardware error recovery
                        benchmark._validate_tensor_core_usage()
                except Exception as e:
                    assert "recovered" in str(e).lower()
