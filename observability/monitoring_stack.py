#!/usr/bin/env python3
"""
Full-stack Observability Integration
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch.distributed as dist
from prometheus_client import start_http_server, Gauge, Counter
from .gpu_telemetry import DCGMCollector, NCCLMonitor
import opentelemetry.trace as trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger("Observability")

class MonitoringStack:
    """Integrated monitoring stack with GPU telemetry and tracing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Initialize components
        self.gpu_telemetry = DCGMCollector(config)
        self.nccl_monitor = NCCLMonitor(config)
        
        # OpenTelemetry setup
        if self.rank == 0:
            trace.set_tracer_provider(TracerProvider())
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.get('otlp_endpoint', 'localhost:4317')
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Start Prometheus metrics server
        if self.rank == 0:
            prometheus_port = config.get('prometheus_port', 8000)
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
    
    def start_monitoring(self):
        """Start all monitoring components."""
        # Start GPU telemetry
        self.gpu_telemetry.start_collection()
        
        # Add monitoring context managers to training components
        if dist.is_initialized():
            dist.monitored_barrier = self.nccl_monitor.measure_communication_time
        
        logger.info("Full monitoring stack started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.gpu_telemetry.stop_collection()
        logger.info("Monitoring stack stopped")
    
    @staticmethod
    def create_trace_span(name: str, attributes: Dict[str, Any] = None):
        """Create OpenTelemetry trace span."""
        tracer = trace.get_tracer(__name__)
        return tracer.start_as_current_span(
            name,
            attributes=attributes or {}
        )

def main():
    # Example usage
    config = {
        'collection_interval_s': 1.0,
        'ecc_error_threshold': 10,
        'otlp_endpoint': 'localhost:4317',
        'prometheus_port': 8000
    }
    
    monitoring = MonitoringStack(config)
    monitoring.start_monitoring()
    
    try:
        # Example trace span
        with monitoring.create_trace_span("training_iteration",
                                        {"batch_size": 32}):
            time.sleep(1)  # Simulate work
            
    finally:
        monitoring.stop_monitoring()

if __name__ == "__main__":
    main()
