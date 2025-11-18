#!/usr/bin/env python3
"""
PyTorch Profiler with NVTX Annotations and Multi-rank Trace Collation
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from prometheus_client import Gauge, Counter, start_http_server
import nvidia_dlprof_pytorch_nvtx

# Prometheus Metrics
profiling_overhead_percent = Gauge("profiling_overhead_percent", "Profiling overhead", ["rank"])
profiler_traces_generated = Counter("profiler_traces_total", "Number of traces generated", ["rank"])
memory_peak_allocated_gb = Gauge("memory_peak_allocated_gb", "Peak memory allocated", ["rank", "device"])
flops_per_second = Gauge("flops_per_second", "Floating point operations per second", ["rank"])

logger = logging.getLogger("PyTorchProfiler")

class HPCProfiler:
    """Production-grade PyTorch profiler with NVTX annotations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Profiling configuration
        self.profile_dir = Path(config.get('profile_dir', './profiles'))
        self.profile_dir.mkdir(exist_ok=True)
        
        self.trace_steps = config.get('trace_steps', 10)
        self.warmup_steps = config.get('warmup_steps', 5)
        self.active_steps = config.get('active_steps', 5)
        self.overhead_threshold = config.get('overhead_threshold_percent', 2.0)
        
        # NVTX configuration
        self.nvtx_enabled = config.get('nvtx_enabled', True)
        if self.nvtx_enabled:
            nvidia_dlprof_pytorch_nvtx.init()
        
        # Profiler state
        self.profiler: Optional[profile] = None
        self.current_step = 0
        self.baseline_times: List[float] = []
        self.profiled_times: List[float] = []

    def create_profiler(self) -> profile:
        """Create PyTorch profiler with appropriate configuration."""
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        return profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.warmup_steps,
                warmup=2,
                active=self.active_steps,
                repeat=1
            ),
            on_trace_ready=self._trace_ready_callback,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        )

    def _trace_ready_callback(self, prof: profile):
        """Callback when trace is ready."""
        timestamp = int(time.time())
        trace_file = self.profile_dir / f"trace_rank_{self.rank}_{timestamp}.json"
        
        # Export trace
        prof.export_chrome_trace(str(trace_file))
        profiler_traces_generated.labels(rank=self.rank).inc()
        
        # Export memory timeline
        memory_file = self.profile_dir / f"memory_rank_{self.rank}_{timestamp}.html"
        prof.export_memory_timeline(str(memory_file), device="cuda:0")
        
        # Calculate and export metrics
        self._export_performance_metrics(prof)
        
        logger.info(f"Trace exported: {trace_file}")

    def _export_performance_metrics(self, prof: profile):
        """Export performance metrics from profiler."""
        # Memory metrics
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            memory_peak_allocated_gb.labels(rank=self.rank, device="cuda:0").set(peak_memory)
        
        # FLOPs calculation
        total_flops = 0
        total_time = 0
        
        for event in prof.events():
            if event.flops > 0:
                total_flops += event.flops
                total_time += event.cuda_time_total if event.cuda_time_total > 0 else event.cpu_time_total
        
        if total_time > 0:
            flops_rate = total_flops / (total_time * 1e-6)  # FLOPS per second
            flops_per_second.labels(rank=self.rank).set(flops_rate)

    @torch.profiler.record_function("training_step")
    def profile_training_step(self, model: nn.Module, batch: torch.Tensor, 
                            targets: torch.Tensor) -> torch.Tensor:
        """Profile a training step with NVTX annotations."""
        with record_function("forward_pass"):
            if self.nvtx_enabled:
                nvidia_dlprof_pytorch_nvtx.range_push("forward_pass")
            
            try:
                outputs = model(batch)
                loss = nn.functional.cross_entropy(outputs, targets)
            finally:
                if self.nvtx_enabled:
                    nvidia_dlprof_pytorch_nvtx.range_pop()
        
        with record_function("backward_pass"):
            if self.nvtx_enabled:
                nvidia_dlprof_pytorch_nvtx.range_push("backward_pass")
            
            try:
                loss.backward()
            finally:
                if self.nvtx_enabled:
                    nvidia_dlprof_pytorch_nvtx.range_pop()
        
        return loss

    def measure_overhead(self, model: nn.Module, test_batch: torch.Tensor, 
                        test_targets: torch.Tensor, num_iterations: int = 100) -> float:
        """Measure profiling overhead."""
        # Baseline without profiling
        model.eval()
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            outputs = model(test_batch)
            loss = nn.functional.cross_entropy(outputs, test_targets)
            loss.backward()
            
        torch.cuda.synchronize()
        baseline_time = time.time() - start_time
        
        # Clear gradients
        model.zero_grad()
        
        # With profiling
        torch.cuda.synchronize()
        start_time = time.time()
        
        with self.create_profiler() as prof:
            for i in range(num_iterations):
                prof.step()
                with record_function("test_iteration"):
                    outputs = model(test_batch)
                    loss = nn.functional.cross_entropy(outputs, test_targets)
                    loss.backward()
        
        torch.cuda.synchronize()
        profiled_time = time.time() - start_time
        
        overhead_percent = ((profiled_time - baseline_time) / baseline_time) * 100
        profiling_overhead_percent.labels(rank=self.rank).set(overhead_percent)
        
        logger.info(f"Profiling overhead: {overhead_percent:.2f}% "
                   f"(baseline: {baseline_time:.3f}s, profiled: {profiled_time:.3f}s)")
        
        model.train()
        return overhead_percent

    def start_profiling(self, model: nn.Module):
        """Start profiling session."""
        self.profiler = self.create_profiler()
        self.profiler.__enter__()
        self.current_step = 0
        logger.info("Profiling session started")

    def step(self):
        """Advance profiler step."""
        if self.profiler:
            self.profiler.step()
            self.current_step += 1

    def stop_profiling(self):
        """Stop profiling session."""
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
            logger.info("Profiling session stopped")

    def collate_traces(self) -> Path:
        """Collate traces from all ranks."""
        if self.rank != 0:
            return None
        
        # Wait for all ranks to finish
        if dist.is_initialized():
            dist.barrier()
        
        collated_dir = self.profile_dir / "collated"
        collated_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        collated_file = collated_dir / f"global_trace_{timestamp}.json"
        
        # Simple trace collation (in production, use more sophisticated merging)
        all_traces = []
        for trace_file in self.profile_dir.glob("trace_rank_*.json"):
            if trace_file.exists():
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                    all_traces.append(trace_data)
        
        # Merge traces (simplified - real implementation would merge events properly)
        if all_traces:
            merged_trace = all_traces[0]  # Start with first trace
            # Add metadata about ranks
            merged_trace['metadata'] = {
                'world_size': self.world_size,
                'collated_ranks': len(all_traces),
                'timestamp': timestamp
            }
            
            with open(collated_file, 'w') as f:
                json.dump(merged_trace, f, indent=2)
        
        logger.info(f"Global trace collated: {collated_file}")
        return collated_file


def main():
    parser = argparse.ArgumentParser(description="PyTorch Profiler with NVTX")
    parser.add_argument("--config", type=str, default="configs/profiling.yaml")
    parser.add_argument("--trace-steps", type=int, default=10)
    parser.add_argument("--measure-overhead", action="store_true")
    args = parser.parse_args()
    
    # Configuration
    config = {
        'profile_dir': './profiles',
        'trace_steps': args.trace_steps,
        'warmup_steps': 5,
        'active_steps': 5,
        'overhead_threshold_percent': 2.0,
        'nvtx_enabled': True
    }
    
    # Start metrics server
    start_http_server(8001)
    
    # Initialize profiler
    profiler = HPCProfiler(config)
    
    # Create test model
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    ).cuda()
    
    test_batch = torch.randn(32, 128).cuda()
    test_targets = torch.randint(0, 10, (32,)).cuda()
    
    if args.measure_overhead:
        overhead = profiler.measure_overhead(model, test_batch, test_targets)
        if overhead > config['overhead_threshold_percent']:
            logger.warning(f"Profiling overhead {overhead:.2f}% exceeds threshold {config['overhead_threshold_percent']}%")
        return
    
    # Profile training steps
    profiler.start_profiling(model)
    
    for step in range(config['trace_steps']):
        loss = profiler.profile_training_step(model, test_batch, test_targets)
        profiler.step()
        
        if step % 10 == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.4f}")
    
    profiler.stop_profiling()
    
    # Collate traces
    global_trace = profiler.collate_traces()
    if global_trace:
        logger.info(f"View traces with: chrome://tracing (load {global_trace})")

if __name__ == "__main__":
    main()
