#!/usr/bin/env python3
"""
GPU Telemetry with DCGM Integration and ECC Error Monitoring
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import pynvml
import torch
from prometheus_client import Gauge, Counter, start_http_server
from dataclasses import dataclass
import psutil

# Prometheus Metrics
gpu_utilization_percent = Gauge("gpu_utilization_percent", "GPU utilization", ["rank", "device"])
gpu_memory_used_gb = Gauge("gpu_memory_used_gb", "GPU memory used", ["rank", "device"])
gpu_memory_total_gb = Gauge("gpu_memory_total_gb", "GPU memory total", ["rank", "device"])
gpu_temperature_celsius = Gauge("gpu_temperature_celsius", "GPU temperature", ["rank", "device"])
gpu_power_watts = Gauge("gpu_power_watts", "GPU power consumption", ["rank", "device"])
gpu_clock_graphics_mhz = Gauge("gpu_clock_graphics_mhz", "GPU graphics clock", ["rank", "device"])
gpu_clock_memory_mhz = Gauge("gpu_clock_memory_mhz", "GPU memory clock", ["rank", "device"])
ecc_errors_total = Counter("gpu_ecc_errors_total", "ECC errors", ["rank", "device", "error_type"])
ecc_error_threshold_exceeded = Counter("gpu_ecc_error_threshold_exceeded_total", "ECC threshold exceeded", ["rank", "device"])
gpu_pcie_throughput_gbps = Gauge("gpu_pcie_throughput_gbps", "PCIe throughput", ["rank", "device", "direction"])

logger = logging.getLogger("GPUTelemetry")

@dataclass
class GPUStats:
    """GPU statistics container."""
    device_id: int
    name: str
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    temperature_celsius: float
    power_watts: float
    graphics_clock_mhz: int
    memory_clock_mhz: int
    ecc_errors: Dict[str, int]
    pcie_throughput_rx_gbps: float
    pcie_throughput_tx_gbps: float

class DCGMCollector:
    """DCGM-based GPU telemetry collector."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.collection_interval = config.get('collection_interval_s', 1.0)
        self.ecc_error_threshold = config.get('ecc_error_threshold', 10)
        
        # Initialize NVIDIA ML
        try:
            pynvml.nvmlInit()
            self.nvml_available = True
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML initialized with {self.device_count} GPUs")
        except Exception as e:
            logger.warning(f"NVML initialization failed: {e}")
            self.nvml_available = False
            self.device_count = 0
        
        # ECC error tracking
        self.previous_ecc_counts = {}
        self.ecc_error_history = {}
        
        # Collection thread
        self.is_collecting = False
        self.collection_thread = None
    
    def get_gpu_stats(self, device_id: int) -> Optional[GPUStats]:
        """Get comprehensive GPU statistics."""
        if not self.nvml_available:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # Basic info
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            
            # Memory
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0
            
            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power = 0
            
            # Clock speeds
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except:
                graphics_clock = 0
                memory_clock = 0
            
            # ECC errors
            ecc_errors = self._get_ecc_errors(handle, device_id)
            
            # PCIe throughput
            pcie_rx, pcie_tx = self._get_pcie_throughput(handle)
            
            return GPUStats(
                device_id=device_id,
                name=name,
                utilization_percent=gpu_util,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                temperature_celsius=temperature,
                power_watts=power,
                graphics_clock_mhz=graphics_clock,
                memory_clock_mhz=memory_clock,
                ecc_errors=ecc_errors,
                pcie_throughput_rx_gbps=pcie_rx,
                pcie_throughput_tx_gbps=pcie_tx
            )
            
        except Exception as e:
            logger.error(f"Error getting GPU {device_id} stats: {e}")
            return None
    
    def _get_ecc_errors(self, handle, device_id: int) -> Dict[str, int]:
        """Get ECC error counts."""
        ecc_errors = {
            'single_bit': 0,
            'double_bit': 0,
            'aggregate_single_bit': 0,
            'aggregate_double_bit': 0
        }
        
        try:
            # Single bit errors
            single_bit = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_SINGLE_BIT_ECC, pynvml.NVML_VOLATILE_ECC
            )
            ecc_errors['single_bit'] = single_bit
            
            # Double bit errors
            double_bit = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_DOUBLE_BIT_ECC, pynvml.NVML_VOLATILE_ECC
            )
            ecc_errors['double_bit'] = double_bit
            
            # Aggregate errors
            agg_single = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_SINGLE_BIT_ECC, pynvml.NVML_AGGREGATE_ECC
            )
            ecc_errors['aggregate_single_bit'] = agg_single
            
            agg_double = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_DOUBLE_BIT_ECC, pynvml.NVML_AGGREGATE_ECC
            )
            ecc_errors['aggregate_double_bit'] = agg_double
            
            # Check for threshold violations
            self._check_ecc_thresholds(device_id, ecc_errors)
            
        except Exception as e:
            logger.debug(f"ECC error collection failed for device {device_id}: {e}")
        
        return ecc_errors
    
    def _check_ecc_thresholds(self, device_id: int, current_ecc: Dict[str, int]):
        """Check ECC error thresholds and trigger alerts."""
        prev_ecc = self.previous_ecc_counts.get(device_id, {})
        
        for error_type, current_count in current_ecc.items():
            prev_count = prev_ecc.get(error_type, 0)
            new_errors = current_count - prev_count
            
            if new_errors > self.ecc_error_threshold:
                logger.error(f"ECC error threshold exceeded on GPU {device_id}: "
                           f"{new_errors} new {error_type} errors")
                ecc_error_threshold_exceeded.labels(
                    rank=self.rank, device=f"cuda:{device_id}"
                ).inc()
        
        self.previous_ecc_counts[device_id] = current_ecc.copy()
    
    def _get_pcie_throughput(self, handle) -> tuple[float, float]:
        """Get PCIe throughput in GB/s."""
        try:
            # Get PCIe throughput counters
            rx_bytes = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            tx_bytes = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            
            # Convert to GB/s (values are in KB/s)
            rx_gbps = rx_bytes / (1024 * 1024)
            tx_gbps = tx_bytes / (1024 * 1024)
            
            return rx_gbps, tx_gbps
        except:
            return 0.0, 0.0
    
    def start_collection(self):
        """Start GPU telemetry collection."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logger.info("GPU telemetry collection started")
    
    def stop_collection(self):
        """Stop GPU telemetry collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("GPU telemetry collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.is_collecting:
            try:
                for device_id in range(self.device_count):
                    stats = self.get_gpu_stats(device_id)
                    if stats:
                        self._update_prometheus_metrics(stats)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in telemetry collection loop: {e}")
                time.sleep(1)
    
    def _update_prometheus_metrics(self, stats: GPUStats):
        """Update Prometheus metrics with GPU stats."""
        device_label = f"cuda:{stats.device_id}"
        
        # Basic metrics
        gpu_utilization_percent.labels(rank=self.rank, device=device_label).set(stats.utilization_percent)
        gpu_memory_used_gb.labels(rank=self.rank, device=device_label).set(stats.memory_used_gb)
        gpu_memory_total_gb.labels(rank=self.rank, device=device_label).set(stats.memory_total_gb)
        gpu_temperature_celsius.labels(rank=self.rank, device=device_label).set(stats.temperature_celsius)
        gpu_power_watts.labels(rank=self.rank, device=device_label).set(stats.power_watts)
        gpu_clock_graphics_mhz.labels(rank=self.rank, device=device_label).set(stats.graphics_clock_mhz)
        gpu_clock_memory_mhz.labels(rank=self.rank, device=device_label).set(stats.memory_clock_mhz)
        
        # ECC errors
        for error_type, count in stats.ecc_errors.items():
            ecc_errors_total.labels(rank=self.rank, device=device_label, error_type=error_type).inc(count)
        
        # PCIe throughput
        gpu_pcie_throughput_gbps.labels(rank=self.rank, device=device_label, direction="rx").set(stats.pcie_throughput_rx_gbps)
        gpu_pcie_throughput_gbps.labels(rank=self.rank, device=device_label, direction="tx").set(stats.pcie_throughput_tx_gbps)

class NCCLMonitor:
    """NCCL communication imbalance monitor."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.imbalance_threshold = config.get('imbalance_threshold_percent', 20.0)
        
        # NCCL timing tracking
        self.communication_times = []
        self.imbalance_history = []
    
    def measure_communication_time(self, operation_name: str):
        """Context manager for measuring NCCL communication time."""
        return NCCLTimingContext(self, operation_name)
    
    def record_communication_time(self, operation: str, time_ms: float):
        """Record communication timing."""
        self.communication_times.append({
            'operation': operation,
            'time_ms': time_ms,
            'timestamp': time.time(),
            'rank': self.rank
        })
        
        # Check for imbalance
        self._check_communication_imbalance(operation, time_ms)
    
    def _check_communication_imbalance(self, operation: str, time_ms: float):
        """Check for NCCL communication imbalance."""
        if self.world_size <= 1:
            return
        
        # Collect timing from all ranks (simplified simulation)
        # In real implementation, this would use NCCL collective operations
        
        # For demonstration, simulate imbalance detection
        if len(self.communication_times) >= 10:
            recent_times = [t['time_ms'] for t in self.communication_times[-10:]]
            avg_time = sum(recent_times) / len(recent_times)
            max_time = max(recent_times)
            min_time = min(recent_times)
            
            if max_time > 0:
                imbalance_percent = ((max_time - min_time) / max_time) * 100
                
                if imbalance_percent > self.imbalance_threshold:
                    logger.warning(f"NCCL imbalance detected for {operation}: "
                                 f"{imbalance_percent:.1f}% (threshold: {self.imbalance_threshold}%)")
                    
                    # Update metrics
                    nccl_imbalance_detected.labels(
                        rank=self.rank, operation=operation
                    ).inc()

class NCCLTimingContext:
    """Context manager for NCCL timing measurement."""
    
    def __init__(self, monitor: NCCLMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.monitor.record_communication_time(self.operation_name, elapsed_ms)

# Add missing metrics
nccl_imbalance_detected = Counter("nccl_imbalance_detected_total", "NCCL imbalance detected", ["rank", "operation"])

def main():
    parser = argparse.ArgumentParser(description="GPU Telemetry with DCGM")
    parser.add_argument("--collection-interval", type=float, default=1.0,
                       help="Collection interval in seconds")
    parser.add_argument("--ecc-threshold", type=int, default=10,
                       help="ECC error threshold")
    parser.add_argument("--test-ecc-alerts", action="store_true",
                       help="Test ECC error alerting")
    args = parser.parse_args()
    
    # Start metrics server
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        start_http_server(8006)
    
    # Configuration
    config = {
        'collection_interval_s': args.collection_interval,
        'ecc_error_threshold': args.ecc_threshold,
        'imbalance_threshold_percent': 20.0
    }
    
    # Initialize collectors
    dcgm_collector = DCGMCollector(config)
    nccl_monitor = NCCLMonitor(config)
    
    if args.test_ecc_alerts:
        # Test ECC error threshold alerting
        logger.info("Testing ECC error threshold alerting...")
        
        for device_id in range(dcgm_collector.device_count):
            # Simulate high ECC error count
            fake_ecc_errors = {
                'single_bit': 50,  # Above threshold
                'double_bit': 5,
                'aggregate_single_bit': 100,
                'aggregate_double_bit': 10
            }
            dcgm_collector._check_ecc_thresholds(device_id, fake_ecc_errors)
        
        logger.info("✅ ECC error threshold test completed")
        return
    
    # Start telemetry collection
    dcgm_collector.start_collection()
    
    # Simulate NCCL communication monitoring
    logger.info("GPU telemetry collection running...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Simulate some NCCL operations for monitoring
        while True:
            time.sleep(5)
            
            # Simulate NCCL all-reduce timing
            with nccl_monitor.measure_communication_time("all_reduce"):
                time.sleep(0.01)  # Simulate communication time
            
            # Simulate NCCL broadcast timing
            with nccl_monitor.measure_communication_time("broadcast"):
                time.sleep(0.005)  # Simulate communication time
            
    except KeyboardInterrupt:
        logger.info("Stopping GPU telemetry collection...")
        dcgm_collector.stop_collection()

if __name__ == "__main__":
    main()
