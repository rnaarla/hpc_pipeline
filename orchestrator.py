#!/usr/bin/env python3
"""
orchestrator.py
---------------
Pipeline orchestrator for HPC Deep Learning.

Features:
- Config-driven orchestration
- Integrates profiling, training, data pipeline, recovery, observability
- SLURM + Kubernetes launch hooks
- Experiment metadata logging
- Fault-tolerant execution with resume
"""

import os
import sys
import json
import yaml
import time
import socket
import argparse
import logging
import subprocess
import threading
import traceback

import torch
import torch.distributed as dist

# Import pipeline modules
from profiling.profiler import HPCProfiler
from optimization.amp_training import AMPTrainer
from distributed.ddp_training import DDPTrainer
from distributed.deepspeed_adapter import DeepSpeedTrainer, build_default_config as ds_config
from data.pipeline import MultiTierDataPipeline, GPUDirectDataset
from fault_tolerance.recovery_manager import RecoveryManager, requeue_job
from benchmarking.roofline_analysis import run_roofline, throughput_bench, fit_kaplan

from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Orchestrator")

# -----------------------------------------------------------------------------
# Distributed Setup
# -----------------------------------------------------------------------------
def init_distributed():
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % max(cuda_count, 1))
    return rank, world_size, local_rank

# -----------------------------------------------------------------------------
# Experiment Metadata
# -----------------------------------------------------------------------------
def log_metadata(config, output_dir="./experiments"):
    os.makedirs(output_dir, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", f"job_{int(time.time())}")
    host = socket.gethostname()
    metadata = {
        "job_id": job_id,
        "host": host,
        "timestamp": time.time(),
        "config": config
    }
    out = os.path.join(output_dir, f"{job_id}.json")
    with open(out, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Experiment metadata written to {out}")

# -----------------------------------------------------------------------------
# SLURM Integration
# -----------------------------------------------------------------------------
def launch_slurm(config_file, nodes=2, gpus_per_node=8, time_limit="48:00:00"):
    cmd = [
        "sbatch",
        f"--nodes={nodes}",
        f"--gres=gpu:{gpus_per_node}",
        f"--time={time_limit}",
        config_file
    ]
    logger.info("Submitting SLURM job: " + " ".join(cmd))
    subprocess.run(cmd)

# -----------------------------------------------------------------------------
# Kubernetes Integration
# -----------------------------------------------------------------------------
def launch_k8s(yaml_file):
    cmd = ["kubectl", "apply", "-f", yaml_file]
    logger.info("Launching K8s job: " + " ".join(cmd))
    subprocess.run(cmd)

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, config):
        self.config = config
        self.rank, self.world_size, self.local_rank = 0, 1, 0
        try:
            self.rank, self.world_size, self.local_rank = init_distributed()
        except Exception:
            pass
        self.output_dir = config.get("output_dir", "./runs")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_profiler(self, model, dataloader):
        steps = self.config.get("profile_steps", 20)
        profiler = HPCProfiler(model, dataloader, output_dir=self.output_dir, rank=self.rank)
        profiler.run(steps=steps)

    def _amp_config(self):
        cfg = {
            "amp_enabled": True,
            "activation_checkpointing": True,
            "oom_recovery_enabled": True,
            "initial_grad_accum_steps": 1,
            "max_grad_accum_steps": 32,
            "checkpoint_dir": os.path.join(self.output_dir, "amp"),
            "save_every_n_steps": self.config.get("save_every_n_steps", 100),
        }
        user_cfg = self.config.get("amp", {})
        cfg.update(user_cfg)
        return cfg

    def _ddp_config(self):
        cfg = {
            "amp_enabled": True,
            "learning_rate": self.config.get("learning_rate", 1e-4),
            "weight_decay": self.config.get("weight_decay", 0.01),
            "find_unused_parameters": False,
            "bucket_cap_mb": 25,
            "checkpoint_every_n_steps": self.config.get("checkpoint_every_n_steps", 100),
            "enable_elastic": True,
            "num_workers": self.config.get("num_workers", 4),
            "checkpoint_dir": os.path.join(self.output_dir, "ddp"),
        }
        user_cfg = self.config.get("ddp", {})
        cfg.update(user_cfg)
        return cfg

    def run_training(self, model, dataloader):
        mode = str(self.config.get("mode", "ddp")).lower()
        epochs = int(self.config.get("epochs", 2))
        max_steps = int(self.config.get("max_steps_per_epoch", 50))
        lr = float(self.config.get("learning_rate", 1e-4))

        if mode == "amp":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            trainer = AMPTrainer(model, optimizer, self._amp_config())
            for ep in range(epochs):
                avg_loss = trainer.train_epoch(dataloader, epoch=ep, max_steps=max_steps)
                if self.rank == 0:
                    trainer.save_checkpoint(step=(ep + 1) * max_steps, loss=avg_loss)

        elif mode == "ddp":
            trainer = DDPTrainer(model, self._ddp_config())
            dataset = getattr(dataloader, "dataset", None)
            batch_size = getattr(dataloader, "batch_size", None) or 32
            ddp_loader = dataloader
            if dataset is not None:
                ddp_loader = trainer.create_dataloader(dataset, batch_size=batch_size)

            for ep in range(epochs):
                avg_loss = trainer.train_epoch(epoch=ep, dataloader=ddp_loader, max_steps=max_steps)
                if self.rank == 0:
                    trainer.save_checkpoint(step=(ep + 1) * max_steps, loss=avg_loss)

        elif mode == "deepspeed":
            try:
                config = ds_config(self.output_dir, overrides=self.config.get("deepspeed", {}))
                trainer = DeepSpeedTrainer(model, dataloader, config, rank=self.rank, world_size=self.world_size)
                for ep in range(epochs):
                    trainer.train_epoch(epoch=ep, max_steps=max_steps)
                    if self.rank == 0:
                        trainer.save_checkpoint(tag=f"ep{ep}")
            except RuntimeError as exc:
                logger.warning("DeepSpeed training unavailable: %s", exc)
        else:
            raise ValueError(f"Unsupported training mode: {mode}")

    def run_data_pipeline(self):
        pipeline_cfg = {
            "nvme_path": self.config.get("nvme_path", f"/tmp/orch_nvme_rank_{self.rank}"),
            "hbm_capacity_gb": self.config.get("hbm_capacity_gb", 40),
            "dram_capacity_gb": self.config.get("dram_capacity_gb", 64),
            "nvme_capacity_gb": self.config.get("nvme_capacity_gb", 500),
            "prefetch_queue_size": self.config.get("prefetch_queue_size", 16),
            "prefetch_workers": self.config.get("prefetch_workers", 4),
        }
        pipeline_cfg.update(self.config.get("data_pipeline", {}))
        pipeline = MultiTierDataPipeline(pipeline_cfg, rank=self.rank)
        dataset = GPUDirectDataset(
            pipeline,
            num_samples=pipeline_cfg.get("prefill_samples", 1024),
        )
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def run_recovery(self, model, optimizer):
        mgr = RecoveryManager(model, optimizer, output_dir="./checkpoints", rank=self.rank)
        mgr.save_checkpoint(step=100)
        if self.rank == 0:
            RecoveryManager.stitch_checkpoints("./checkpoints", step=100, world_size=self.world_size)
        state = mgr.load_checkpoint(step=100)
        if not state and self.rank == 0:
            requeue_job()

    def run_observability(self):
        from observability.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(rank=self.rank, port=9500)
        t = threading.Thread(target=agent.run, daemon=True)
        t.start()

    def run_benchmarking(self, model):
        run_roofline(rank=self.rank)
        throughput_bench(model, seq_len=1024, batch_size=4, steps=10, rank=self.rank)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPC Deep Learning Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Config YAML/JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        if args.config.endswith(".yaml") or args.config.endswith(".yml"):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    orch = Orchestrator(config)
    log_metadata(config)

    # Dummy model + dataset fallback
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10),
    )

    if config.get("enable_data_pipeline", False):
        dl = orch.run_data_pipeline()
    else:
        x = torch.randn(2048, 1024)
        y = torch.randint(0, 10, (2048,))
        ds = TensorDataset(x, y)
        dl = DataLoader(ds, batch_size=config.get("batch_size", 32), shuffle=True)

    if config.get("enable_profiler", False):
        orch.run_profiler(model, dl)

    if config.get("enable_training", True):
        orch.run_training(model, dl)

    if config.get("enable_recovery", True):
        orch.run_recovery(model, torch.optim.AdamW(model.parameters(), lr=1e-4))

    if config.get("enable_observability", True):
        orch.run_observability()

    if config.get("enable_benchmarking", True):
        orch.run_benchmarking(model)

    logger.info("✅ Orchestration complete.")
    sys.exit(0) 