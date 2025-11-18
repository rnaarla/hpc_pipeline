#!/usr/bin/env python3
"""
roofline_analysis.py
--------------------
Benchmarking & roofline analysis for LLM training.

Features:
- FLOP & memory bandwidth roofline analysis
- Throughput scaling benchmarks (tokens/sec vs GPUs)
- Kaplan scaling law fitting for LLMs
- Prometheus metrics export
- CLI interface for benchmarking
"""

import os
import sys
import time
import json
import math
import logging
import argparse
import random
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from prometheus_client import Gauge, start_http_server

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Benchmarking")

# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------
roofline_flops = Gauge("roofline_flops", "Achieved FLOPs (TFLOPs)", ["rank"])
roofline_bw = Gauge("roofline_bw", "Achieved memory bandwidth (GB/s)", ["rank"])
tokens_per_sec = Gauge("tokens_per_sec", "Training throughput (tok/s)", ["rank"])
scaling_efficiency = Gauge("scaling_efficiency", "Scaling efficiency (%)", ["rank"])

# -----------------------------------------------------------------------------
# Distributed Helpers
# -----------------------------------------------------------------------------
def init_distributed():
    if not dist.is_initialized():
        try:
            dist.init_process_group("nccl", init_method="env://")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed: {e}")
            return 0, 1
    return dist.get_rank(), dist.get_world_size()

# -----------------------------------------------------------------------------
# Roofline Analysis
# -----------------------------------------------------------------------------
def run_roofline(M=4096, N=4096, K=4096, iters=20, rank=0):
    """
    Performs roofline analysis on a GEMM operation (torch.matmul).
    """
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return 0.0, 0.0, 0.0

    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")

    # Warmup
    for _ in range(5):
        torch.matmul(A, B)

    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    duration = (time.time() - start) / iters

    flops = 2 * M * N * K
    achieved_flops = flops / (duration * 1e12)  # TFLOPs
    mem_bytes = (M*K + K*N + M*N) * 4
    bw = mem_bytes / (duration * 1e9)  # GB/s
    ai = flops / mem_bytes

    roofline_flops.labels(rank=rank).set(achieved_flops)
    roofline_bw.labels(rank=rank).set(bw)

    logger.info(f"[Rank {rank}] Roofline: {achieved_flops:.2f} TFLOPs | {bw:.2f} GB/s | AI={ai:.2f}")
    return achieved_flops, bw, ai

def plot_roofline(results, out="roofline.png"):
    """
    Plots roofline results (FLOPs vs. AI).
    """
    fig, ax = plt.subplots()
    for rank, (flops, bw, ai) in results.items():
        ax.scatter(ai, flops, label=f"Rank {rank}")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
    ax.set_ylabel("TFLOPs")
    ax.set_title("Roofline Analysis")
    ax.legend()
    plt.savefig(out)
    logger.info(f"Saved roofline plot to {out}")

# -----------------------------------------------------------------------------
# Throughput Benchmark
# -----------------------------------------------------------------------------
def throughput_bench(model, tokenizer=None, seq_len=2048, batch_size=8,
                     steps=20, rank=0):
    """
    Measures training throughput (tokens/sec).
    """
    if not torch.cuda.is_available():
        logger.error("CUDA not available for throughput benchmark")
        return 0.0

    x = torch.randint(0, 32000, (batch_size, seq_len), device="cuda")
    emb = nn.Embedding(32000, 4096).cuda()
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(emb.parameters()),
        lr=1e-4
    )

    torch.cuda.synchronize()
    start = time.time()

    tokens = 0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(emb(x))
        loss = out.sum()
        loss.backward()
        optimizer.step()
        tokens += batch_size * seq_len

    torch.cuda.synchronize()
    duration = (time.time() - start) / steps
    tput = tokens / (steps * duration)
    tokens_per_sec.labels(rank=rank).set(tput)
    logger.info(f"[Rank {rank}] Throughput {tput:.1f} tokens/s")
    return tput

def plot_scaling(data, out="scaling.png"):
    """
    Plots scaling efficiency vs. GPU count.
    """
    gpus, tputs = zip(*data)
    baseline = tputs[0]
    efficiency = [100 * (t/baseline) / (g/baseline) for g, t in zip(gpus, tputs)]

    fig, ax = plt.subplots()
    ax.plot(gpus, efficiency, marker="o")
    ax.set_xlabel("GPU Count")
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_title("Scaling Benchmark")
    plt.savefig(out)
    logger.info(f"Saved scaling plot to {out}")

# -----------------------------------------------------------------------------
# Kaplan Scaling Laws
# -----------------------------------------------------------------------------
def kaplan_loss(dataset_size, model_size, params):
    """
    Simple Kaplan scaling law approximation:
    loss = a * (N^-alpha) + b * (D^-beta) + c
    """
    a, b, c, alpha, beta = params
    return a * (model_size ** -alpha) + b * (dataset_size ** -beta) + c

def fit_kaplan(data):
    """
    Fits scaling law params to (dataset, model_size, loss).
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        logger.error("scipy not available for Kaplan fitting")
        return [1.0, 1.0, 1.0, 0.1, 0.1]

    def wrapper(inputs, a, b, c, alpha, beta):
        ds, ms = inputs
        return kaplan_loss(ds, ms, (a, b, c, alpha, beta))

    datasets, models, losses = zip(*data)
    try:
        params, _ = curve_fit(wrapper, (np.array(datasets), np.array(models)), np.array(losses),
                              p0=(1.0, 1.0, 1.0, 0.1, 0.1))
        logger.info(f"Fitted Kaplan params: {params}")
        return params
    except Exception as e:
        logger.error(f"Failed to fit Kaplan parameters: {e}")
        return [1.0, 1.0, 1.0, 0.1, 0.1]

def plot_kaplan(data, params, out="scaling_laws.png"):
    """
    Plots scaling law fit.
    """
    datasets, models, losses = zip(*data)
    pred = [kaplan_loss(d, m, params) for d, m in zip(datasets, models)]

    fig, ax = plt.subplots()
    ax.scatter(models, losses, label="Observed")
    ax.plot(models, pred, label="Predicted", color="red")
    ax.set_xlabel("Model Size (params)")
    ax.set_ylabel("Loss")
    ax.set_title("Kaplan Scaling Law")
    ax.legend()
    plt.savefig(out)
    logger.info(f"Saved Kaplan scaling law plot to {out}")

# -----------------------------------------------------------------------------
# Prometheus Exporter
# -----------------------------------------------------------------------------
def start_prometheus(rank, port=9600):
    def run():
        try:
            logger.info(f"[Rank {rank}] Prometheus exporter started on port {port+rank}")
            start_http_server(port + rank)
            while True:
                time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    t = threading.Thread(target=run, daemon=True)
    t.start()

# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roofline & Benchmarking Suite")
    parser.add_argument("--mode", choices=["roofline", "throughput", "kaplan"], required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--out", type=str, default="plot.png")
    parser.add_argument("--plot-all", nargs="*", help="Benchmark JSON files to analyze")
    parser.add_argument("--threshold", type=float, default=0.85, help="Performance threshold")
    args = parser.parse_args()

    rank = args.rank
    start_prometheus(rank)

    if args.mode == "roofline":
        results = {}
        for i in range(2):  # simulate 2 benchmarks
            flops, bw, ai = run_roofline(rank=rank)
            results[i] = (flops, bw, ai)
        plot_roofline(results, out=args.out)

    elif args.mode == "throughput":
        if not torch.cuda.is_available():
            logger.error("CUDA not available for throughput benchmark")
            sys.exit(1)
        model = nn.Linear(4096, 4096)
        tput = throughput_bench(model, seq_len=1024, batch_size=4, steps=10, rank=rank)
        logger.info(f"Throughput {tput:.1f} tokens/s")

    elif args.mode == "kaplan":
        data = [
            (1e8, 1e9, 2.5),
            (1e9, 1e9, 2.2),
            (1e8, 1e10, 2.1),
            (1e9, 1e10, 1.9),
        ]
        params = fit_kaplan(data)
        plot_kaplan(data, params, out=args.out)
    else:
        logger.error("Invalid mode specified. Use --help for options.")
        sys.exit(1)

    logger.info(f"[Rank {rank}] Benchmarking complete.")
    sys.exit(0)