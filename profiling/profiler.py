#!/usr/bin/env python3
"""
profiler.py
-----------
Principal Engineer-level GPU profiling utility for deep learning workloads.
Designed for multi-node, multi-GPU environments (NVIDIA DGX / H100 clusters).

Features:
- PyTorch Profiler with Kineto backend
- NVTX annotations for Nsight Systems
- Aggregated trace collation for distributed training
- Prometheus-compatible metric exporter
- Fault-tolerant logging for long jobs
"""

import os
import sys
import time
import json
import socket
import logging
import threading
import torch
import torch.profiler as profiler
import torch.distributed as dist
from prometheus_client import start_http_server, Gauge

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HPCProfiler")

# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------
gpu_util = Gauge("gpu_utilization", "GPU utilization (%)", ["rank"])
gpu_mem = Gauge("gpu_memory", "GPU memory usage (bytes)", ["rank"])
step_time = Gauge("step_time", "Training step latency (s)", ["rank"])

# -----------------------------------------------------------------------------
# NVTX Utilities
# -----------------------------------------------------------------------------
try:
    import torch.cuda.nvtx as nvtx
except ImportError:
    class DummyNVTX:
        def range_push(self, msg): pass
        def range_pop(self): pass
    nvtx = DummyNVTX()

def nvtx_wrap(fn):
    """Decorator for NVTX annotation."""
    def wrapped(*args, **kwargs):
        nvtx.range_push(fn.__name__)
        out = fn(*args, **kwargs)
        nvtx.range_pop()
        return out
    return wrapped

# -----------------------------------------------------------------------------
# Multi-Node Helper
# -----------------------------------------------------------------------------
def init_distributed():
    """Initialize torch.distributed if available."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size()

def barrier():
    if dist.is_initialized():
        dist.barrier()

# -----------------------------------------------------------------------------
# Profiler Class
# -----------------------------------------------------------------------------
class HPCProfiler:
    def __init__(self, model, dataloader, output_dir="./logs", port=9000,
                 warmup=5, active=20, wait=2, rank=0):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.port = port
        self.rank = rank
        self.prof = None

        os.makedirs(output_dir, exist_ok=True)

        self.schedule = profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1
        )

        self.prof = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU,
                        profiler.ProfilerActivity.CUDA],
            schedule=self.schedule,
            on_trace_ready=profiler.tensorboard_trace_handler(
                os.path.join(output_dir, f"rank{rank}")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

    @nvtx_wrap
    def train_step(self, batch):
        """Dummy training step with NVTX annotation."""
        x, y = batch
        x, y = x.cuda(), y.cuda()
        out = self.model(x)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        return loss.item()

    def start_prometheus(self):
        """Start Prometheus server in background thread."""
        def run_server():
            logger.info(f"[Rank {self.rank}] Starting Prometheus exporter on port {self.port}")
            start_http_server(self.port)
            while True:
                time.sleep(1)
        t = threading.Thread(target=run_server, daemon=True)
        t.start()

    def run(self, steps=100):
        logger.info(f"[Rank {self.rank}] Starting profiling run for {steps} steps...")
        self.start_prometheus()

        self.prof.start()
        step_durations = []

        for step, batch in enumerate(self.dataloader):
            if step >= steps:
                break

            start = time.time()
            loss = self.train_step(batch)
            torch.cuda.synchronize()
            duration = time.time() - start
            step_durations.append(duration)

            # Update Prometheus
            step_time.labels(rank=self.rank).set(duration)
            mem = torch.cuda.memory_allocated()
            gpu_mem.labels(rank=self.rank).set(mem)
            util = torch.cuda.utilization() if hasattr(torch.cuda, "utilization") else -1
            gpu_util.labels(rank=self.rank).set(util)

            self.prof.step()
            logger.info(f"[Rank {self.rank}] Step {step} | Loss {loss:.4f} | {duration:.3f}s")

        self.prof.stop()

        avg_time = sum(step_durations) / len(step_durations)
        logger.info(f"[Rank {self.rank}] Avg Step Time: {avg_time:.3f}s")

        # Export JSON trace
        trace_file = os.path.join(self.output_dir, f"profile_rank{self.rank}.json")
        self.prof.export_chrome_trace(trace_file)
        logger.info(f"[Rank {self.rank}] Exported Chrome trace to {trace_file}")

# -----------------------------------------------------------------------------
# Aggregator for Multi-Node
# -----------------------------------------------------------------------------
class TraceAggregator:
    """Collates per-rank traces into a global profile."""
    def __init__(self, output_dir="./logs", world_size=1):
        self.output_dir = output_dir
        self.world_size = world_size

    def aggregate(self, output_file="global_trace.json"):
        traces = []
        for r in range(self.world_size):
            f = os.path.join(self.output_dir, f"profile_rank{r}.json")
            if os.path.exists(f):
                with open(f) as fh:
                    traces.append(json.load(fh))
        merged = {"traces": traces}
        out = os.path.join(self.output_dir, output_file)
        with open(out, "w") as fh:
            json.dump(merged, fh)
        logger.info(f"Aggregated trace written to {out}")
        return out

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Distributed init
    try:
        rank, world_size = init_distributed()
    except Exception:
        rank, world_size = 0, 1

    # Dummy model + data
    model = nn.Linear(4096, 4096).cuda()
    data = torch.randn(1024, 4096), torch.randn(1024, 4096)
    ds = TensorDataset(*data)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    profiler = HPCProfiler(model, dl, output_dir="./logs", rank=rank)
    profiler.run(steps=50)

    barrier()
    if rank == 0:
        agg = TraceAggregator("./logs", world_size)
        agg.aggregate()
        logger.info("Profiling complete.")
    else:
        logger.info(f"[Rank {rank}] Profiling complete.")