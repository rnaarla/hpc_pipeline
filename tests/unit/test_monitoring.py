"""Unit tests for monitoring components."""

import pytest
from prometheus_client import CollectorRegistry

from monitoring.kernel_monitor import KernelMonitor


@pytest.mark.unit
class TestKernelMonitor:
    """Unit tests for KernelMonitor."""
    
    def _make_monitor(self) -> KernelMonitor:
        return KernelMonitor(registry=CollectorRegistry())
    
    def test_default_registry(self):
        """KernelMonitor should create a registry when not provided."""
        monitor = KernelMonitor()
        assert isinstance(monitor.registry, CollectorRegistry)
    
    def test_init(self):
        """Test KernelMonitor initialization."""
        monitor = self._make_monitor()
        assert monitor is not None
        assert hasattr(monitor, 'kernel_latency')
        assert hasattr(monitor, 'kernel_errors')
        
    def test_collect_metrics(self):
        """Test metrics collection."""
        monitor = self._make_monitor()
        metrics = monitor.collect_metrics()
        
        assert isinstance(metrics, dict)
        assert "kernel_latency" in metrics
        assert "memory_bandwidth" in metrics
        assert "tensor_core_usage" in metrics
        assert "error_counters" in metrics
        
    def test_alert_on_anomaly_low_occupancy(self, caplog):
        """Test alerting on low occupancy."""
        monitor = self._make_monitor()
        test_metrics = {
            'occupancy': 0.5,
            'tensor_core_util': 0.8
        }
        
        monitor.alert_on_anomaly(test_metrics)
        assert "Low kernel occupancy" in caplog.text
        
    def test_alert_on_anomaly_low_tensor_core(self, caplog):
        """Test alerting on low Tensor Core usage."""
        monitor = self._make_monitor()
        test_metrics = {
            'occupancy': 0.9,
            'tensor_core_util': 0.3
        }
        
        monitor.alert_on_anomaly(test_metrics)
        assert "Low Tensor Core utilization" in caplog.text

    def test_start_stop_monitoring(self):
        """Ensure start/stop monitoring hooks execute without error."""
        monitor = self._make_monitor()
        monitor.start_monitoring()
        metrics = monitor.stop_monitoring()
        assert isinstance(metrics, dict)
