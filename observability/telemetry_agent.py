#!/usr/bin/env python3
"""
telemetry_agent.py
------------------
Observability agent for large-scale LLM training.

Features:
- GPU telemetry via DCGM
- NCCL trace parsing for imbalance detection
- Training telemetry (loss, throughput, grad norms)
- Prometheus exporter
- ECC/power/thermal alerts
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import torch
import pynvml
from prometheus_client import Gauge, start_http_server

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TelemetryAgent")

# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------
gpu_util = Gauge("gpu_utilization", "GPU Utilization (%)", ["rank", "gpu"])
gpu_mem = Gauge("gpu_memory_bytes", "GPU Memory (bytes)", ["rank", "gpu"])
gpu_power = Gauge("gpu_power_watts", "GPU Power (W)", ["rank", "gpu"])
gpu_temp = Gauge("gpu_temperature", "GPU Temp (C)", ["rank", "gpu"])
gpu_ecc = Gauge("gpu_ecc_errors", "ECC Error Count", ["rank", "gpu"])

loss_metric = Gauge("train_loss", "Training loss", ["rank"])
throughput = Gauge("train_throughput", "Samples/sec", ["rank"])
grad_norm = Gauge("grad_norm", "Gradient L2 norm", ["rank"])

nccl_imbalance = Gauge("nccl_imbalance", "NCCL comm imbalance (%)", ["rank"])

# -----------------------------------------------------------------------------
# GPU Telemetry Collector
# -----------------------------------------------------------------------------
class GPUTelemetry:
    def __init__(self, rank=0):
        self.rank = rank
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"[Rank {rank}] DCGM initialized for {self.device_count} GPUs")

    def collect(self):
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            ecc = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                pynvml.NVML_VOLATILE_ECC
            )

            gpu_util.labels(rank=self.rank, gpu=i).set(util.gpu)
            gpu_mem.labels(rank=self.rank, gpu=i).set(mem.used)
            gpu_power.labels(rank=self.rank, gpu=i).set(power)
            gpu_temp.labels(rank=self.rank, gpu=i).set(temp)
            gpu_ecc.labels(rank=self.rank, gpu=i).set(ecc)

            if ecc > 10:
                logger.warning(f"[Rank {self.rank}] GPU{i} ECC errors {ecc} > threshold!")

# -----------------------------------------------------------------------------
# NCCL Trace Parser
# -----------------------------------------------------------------------------
class NCCLTraceParser:
    def __init__(self, log_file="/tmp/nccl_debug.log", rank=0):
        self.log_file = log_file
        self.rank = rank
        self.last_pos = 0

    def parse(self):
        """Parses NCCL log for collective imbalance metrics."""
        if not os.path.exists(self.log_file):
            return
        with open(self.log_file, "r") as f:
            f.seek(self.last_pos)
            lines = f.readlines()
            self.last_pos = f.tell()

        imbalance_detected = 0
        for line in lines:
            if "AllReduce" in line and "latency" in line:
                parts = line.strip().split()
                try:
                    latency = float(parts[-2])
                    expected = float(parts[-1])
                    if expected > 0:
                        imbalance = 100 * (latency - expected) / expected
                        if imbalance > 20:  # >20% imbalance
                            imbalance_detected += 1
                            nccl_imbalance.labels(rank=self.rank).set(imbalance)
                            logger.warning(f"[Rank {self.rank}] NCCL imbalance {imbalance:.2f}%")
                except Exception:
                    continue
        return imbalance_detected

# -----------------------------------------------------------------------------
# Training Telemetry Hooks
# -----------------------------------------------------------------------------
class TrainingTelemetry:
    def __init__(self, rank=0):
        self.rank = rank

    def log_step(self, loss, tokens, duration, grad_tensor=None):
        loss_metric.labels(rank=self.rank).set(loss)
        tput = tokens / duration if duration > 0 else 0
        throughput.labels(rank=self.rank).set(tput)
        if grad_tensor is not None:
            norm = torch.norm(grad_tensor).item()
            grad_norm.labels(rank=self.rank).set(norm)
        logger.info(f"[Rank {self.rank}] Loss={loss:.4f} | Throughput={tput:.1f} tok/s")

# -----------------------------------------------------------------------------
# Telemetry Agent
# -----------------------------------------------------------------------------
class TelemetryAgent:
    def __init__(self, rank=0, nccl_log="/tmp/nccl_debug.log", port=9500):
        self.rank = rank
        self.gpu = GPUTelemetry(rank)
        self.nccl = NCCLTraceParser(nccl_log, rank)
        self.train = TrainingTelemetry(rank)
        self.port = port

    def start_prometheus(self):
        def run():
            logger.info(f"[Rank {self.rank}] Prometheus exporter started on port {self.port+self.rank}")
            start_http_server(self.port + self.rank)
            while True:
                time.sleep(1)
        t = threading.Thread(target=run, daemon=True)
        t.start()

    def run(self):
        self.start_prometheus()
        while True:
            try:
                self.gpu.collect()
                self.nccl.parse()
                time.sleep(5)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Telemetry error: {e}")
                traceback.print_exc()
                time.sleep(5)

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    agent = TelemetryAgent(rank=int(os.environ.get("RANK", 0)))
    agent.run()
# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Dummy model and data
    model = nn.Linear(1024, 512).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = TensorDataset(torch.randn(1000, 1024), torch.randn(1000, 512))
    dataloader = DataLoader(data, batch_size=32)

    # Start telemetry agent
    agent = TelemetryAgent(rank=0)
    agent.run()

    # Simulate training loop
    for epoch in range(2):
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            loss = nn.MSELoss()(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            agent.train.log_step(loss.item(), x.size(0), 0.1, grad_tensor=model.weight.grad)

    logger.info("Telemetry agent example completed successfully.")  