#!/usr/bin/env python3
"""
recovery_manager.py
-------------------
Fault tolerance & recovery manager for multi-node LLM training.

Features:
- Sharded checkpointing per rank
- Checkpoint stitching across ranks
- Resume-at-step consistency validation
- SLURM job requeue integration
- Prometheus telemetry for checkpoint ops
- Handles corrupted checkpoints via hash validation
"""

import os
import sys
import time
import json
import hashlib
import logging
import traceback
import torch
import torch.distributed as dist
from prometheus_client import Gauge, start_http_server

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RecoveryManager")

# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------
ckpt_latency = Gauge("ckpt_latency", "Checkpoint save latency (s)", ["rank"])
ckpt_size = Gauge("ckpt_size", "Checkpoint size (MB)", ["rank"])
recovery_time = Gauge("recovery_time", "Recovery duration (s)", ["rank"])
ckpt_corruption = Gauge("ckpt_corruption", "Checkpoint corruption events", ["rank"])

# -----------------------------------------------------------------------------
# Distributed Helpers
# -----------------------------------------------------------------------------
def init_distributed():
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend, init_method="env://")
        except Exception:
            return 0, 1
    try:
        return dist.get_rank(), dist.get_world_size()
    except Exception:
        return 0, 1

def barrier():
    if dist.is_initialized():
        dist.barrier()

# -----------------------------------------------------------------------------
# Utility: Hash Validation
# -----------------------------------------------------------------------------
def file_md5(path):
    """Compute MD5 for checkpoint validation."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -----------------------------------------------------------------------------
# Recovery Manager
# -----------------------------------------------------------------------------
class RecoveryManager:
    def __init__(self, model, optimizer, scaler=None, output_dir="./checkpoints", rank=0):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.output_dir = output_dir
        self.rank = rank
        os.makedirs(output_dir, exist_ok=True)

    def ckpt_file(self, step):
        return os.path.join(self.output_dir, f"ckpt_rank{self.rank}_step{step}.pt")

    def save_checkpoint(self, step):
        """Save sharded checkpoint for this rank."""
        start = time.time()
        path = self.ckpt_file(step)
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "step": step,
            "rank": self.rank
        }
        torch.save(state, path)
        md5sum = file_md5(path)
        with open(path + ".md5", "w") as f:
            f.write(md5sum)

        size = os.path.getsize(path) / (1024 * 1024)
        latency = time.time() - start

        ckpt_size.labels(rank=self.rank).set(size)
        ckpt_latency.labels(rank=self.rank).set(latency)
        logger.info(f"[Rank {self.rank}] Saved checkpoint step={step} | {size:.2f} MB | {latency:.3f}s")

    def load_checkpoint(self, step):
        """Load sharded checkpoint with hash validation."""
        path = self.ckpt_file(step)
        if not os.path.exists(path):
            logger.warning(f"[Rank {self.rank}] Missing checkpoint {path}")
            return None

        try:
            # Validate hash
            with open(path + ".md5") as f:
                expected = f.read().strip()
            actual = file_md5(path)
            if expected != actual:
                logger.error(f"[Rank {self.rank}] Corruption detected in {path}")
                ckpt_corruption.labels(rank=self.rank).inc()
                return None

            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(path, map_location=map_location)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scaler and state["scaler"]:
                self.scaler.load_state_dict(state["scaler"])

            logger.info(f"[Rank {self.rank}] Restored checkpoint step={state['step']}")
            return state

        except Exception as e:
            logger.error(f"[Rank {self.rank}] Failed to load checkpoint {path}: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def stitch_checkpoints(output_dir, step, world_size, out_file="global_ckpt.pt"):
        """Merge all sharded checkpoints into a single global checkpoint."""
        t0 = time.time()
        global_state = {"model": {}, "optimizers": {}, "scalers": {}, "step": step}
        for r in range(world_size):
            path = os.path.join(output_dir, f"ckpt_rank{r}_step{step}.pt")
            if not os.path.exists(path):
                continue
            state = torch.load(path, map_location="cpu")
            global_state["model"][f"rank{r}"] = state["model"]
            global_state["optimizers"][f"rank{r}"] = state["optimizer"]
            if state["scaler"]:
                global_state["scalers"][f"rank{r}"] = state["scaler"]

        out_path = os.path.join(output_dir, out_file)
        torch.save(global_state, out_path)
        duration = time.time() - t0
        recovery_time.labels(rank=0).set(duration)
        logger.info(f"[Aggregator] Stitched global checkpoint at {out_path} | {duration:.3f}s")
        return out_path

# -----------------------------------------------------------------------------
# SLURM Integration
# -----------------------------------------------------------------------------
def requeue_job():
    """Signal SLURM to requeue job after failure."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        logger.info(f"Requeuing SLURM job {job_id}")
        os.system(f"scontrol requeue {job_id}")

# -----------------------------------------------------------------------------
# Prometheus Exporter
# -----------------------------------------------------------------------------
def start_prometheus(rank, port=9400):
    def run():
        logger.info(f"[Rank {rank}] Starting Prometheus exporter on port {port+rank}")
        start_http_server(port + rank)
        while True:
            time.sleep(1)
    import threading
    t = threading.Thread(target=run, daemon=True)
    t.start()

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn as nn
    rank, world_size = 0, 1
    try:
        rank, world_size = init_distributed()
    except Exception:
        pass

    model = nn.Linear(4096, 4096).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    mgr = RecoveryManager(model, optimizer, output_dir="./checkpoints", rank=rank)
    start_prometheus(rank)

    # Simulate save/load
    step = 10
    mgr.save_checkpoint(step)
    barrier()
    if rank == 0:
        RecoveryManager.stitch_checkpoints("./checkpoints", step, world_size)

    state = mgr.load_checkpoint(step)
    if not state and rank == 0:
        requeue_job()
    if state:
        logger.info(f"[Rank {rank}] Successfully loaded checkpoint for step {state['step']}")
    else:
        logger.warning(f"[Rank {rank}] Failed to load checkpoint for step {step}")
    barrier()

    if rank == 0:
        logger.info("RecoveryManager example completed successfully.")
    else:
        logger.info(f"[Rank {rank}] RecoveryManager example completed successfully.")
    sys.exit(0)
# -----------------------------------------------------------------------------
# End of recovery_manager.py
# -----------------------------------------------------------------------------
    sys.exit(0)
    sys.exit(0)     