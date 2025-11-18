#!/usr/bin/env python3
"""
Enhanced AMP Training with OOM Recovery and Checkpointing
"""

import os
import sys
import time
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from prometheus_client import Gauge, Counter, start_http_server

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AMPTraining")

# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------
step_latency = Gauge("amp_step_latency", "Step latency (s)", ["rank"])
gpu_mem_used = Gauge("gpu_memory_used", "GPU memory used (bytes)", ["rank"])
cpu_mem_used = Gauge("cpu_memory_used", "CPU memory used (bytes)", ["rank"])
oom_events = Gauge("oom_events", "OOM error count", ["rank"])
amp_speedup_ratio = Gauge("amp_speedup_ratio", "AMP speedup vs FP32", ["rank"])
oom_recovery_count = Counter("oom_recovery_total", "OOM recovery events", ["rank"])
grad_accumulation_steps = Gauge("grad_accumulation_steps", "Current grad accumulation steps", ["rank"])
checkpoint_save_time = Gauge("checkpoint_save_seconds", "Checkpoint save duration", ["rank"])

# -----------------------------------------------------------------------------
# Memory Utilities
# -----------------------------------------------------------------------------
def report_memory(rank):
    """Reports memory across tiers: GPU + CPU."""
    gpu_bytes = torch.cuda.memory_allocated()
    cpu_bytes = torch.cuda.memory_reserved() if hasattr(torch, "memory_reserved") else -1
    gpu_mem_used.labels(rank=rank).set(gpu_bytes)
    cpu_mem_used.labels(rank=rank).set(cpu_bytes)
    return gpu_bytes, cpu_bytes

def safe_cuda_call(fn, *args, **kwargs):
    """Wrap CUDA calls with OOM safety."""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise OOMError(str(e))
        raise

class OOMError(Exception): pass

# -----------------------------------------------------------------------------
# AMP Trainer
# -----------------------------------------------------------------------------
class AMPTrainer:
    """Enhanced AMP trainer with OOM recovery and checkpointing."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 config: Dict[str, Any]):
        self.config = config
        self.rank, self.world_size = init_distributed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        
        # AMP components
        self.autocast_enabled = config.get('amp_enabled', True) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.autocast_enabled)
        
        # OOM recovery
        self.grad_accum_steps = config.get('initial_grad_accum_steps', 1)
        self.max_grad_accum_steps = config.get('max_grad_accum_steps', 64)
        self.oom_recovery_enabled = config.get('oom_recovery_enabled', True)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_every_n_steps = config.get('save_every_n_steps', 1000)
        
        # Activation checkpointing
        if config.get('activation_checkpointing', False) and self.device.type == "cuda":
            self._enable_activation_checkpointing()
        
        # Performance tracking
        self.fp32_baseline_time = None
        self.amp_training_time = None
        self.global_step = 0
        
    def _enable_activation_checkpointing(self):
        """Enable activation checkpointing for memory efficiency."""
        def checkpoint_wrapper(module):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
            # For transformer layers
            if hasattr(module, 'forward'):
                original_forward = module.forward
                def checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)
                module.forward = checkpointed_forward
        
        self.model.apply(checkpoint_wrapper)
        logger.info("Activation checkpointing enabled")

    def _calculate_checksum(self, checkpoint_path: Path) -> str:
        """Calculate SHA256 checksum of checkpoint file."""
        hash_sha256 = hashlib.sha256()
        with open(checkpoint_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def save_checkpoint(self, step: int, loss: float, metadata: Dict[str, Any] = None) -> Path:
        """Save checkpoint with hash validation."""
        start_time = time.time()
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}_rank_{self.rank}.pt"
        
        checkpoint_data = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'grad_accum_steps': self.grad_accum_steps,
            'rank': self.rank,
            'world_size': self.world_size,
            'metadata': metadata or {}
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate and save checksum
        checksum = self._calculate_checksum(checkpoint_path)
        checksum_path = checkpoint_path.with_suffix('.sha256')
        with open(checksum_path, 'w') as f:
            f.write(f"{checksum}  {checkpoint_path.name}\n")
        
        save_time = time.time() - start_time
        checkpoint_save_time.labels(rank=self.rank).set(save_time)
        
        logger.info(f"Checkpoint saved: {checkpoint_path} (checksum: {checksum[:8]}...)")
        return checkpoint_path

    def train_epoch(self, dataloader, epoch: int, max_steps: Optional[int] = None) -> float:
        """Run a single training epoch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        processed_steps = 0

        for _, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                raise ValueError("Dataloader must return (inputs, targets)")

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            loss, _ = self.training_step(inputs, targets, step=self.global_step)
            running_loss += loss
            processed_steps += 1
            self.global_step += 1

            if max_steps and processed_steps >= max_steps:
                break

        avg_loss = running_loss / max(processed_steps, 1)
        logger.info(
            f"[AMP][Rank {self.rank}] Epoch {epoch} | steps={processed_steps} | loss={avg_loss:.4f}"
        )
        return avg_loss

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint with hash validation."""
        # Validate checksum
        checksum_path = checkpoint_path.with_suffix('.sha256')
        if checksum_path.exists():
            with open(checksum_path, 'r') as f:
                expected_checksum = f.read().split()[0]
            
            actual_checksum = self._calculate_checksum(checkpoint_path)
            if actual_checksum != expected_checksum:
                raise RuntimeError(f"Checkpoint corruption detected: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Load states
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        self.grad_accum_steps = checkpoint_data.get('grad_accum_steps', 1)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data

    def _handle_oom_recovery(self):
        """Handle OOM by doubling gradient accumulation steps."""
        if not self.oom_recovery_enabled:
            raise RuntimeError("OOM occurred and recovery is disabled")
        
        if self.grad_accum_steps >= self.max_grad_accum_steps:
            raise RuntimeError(f"Max gradient accumulation steps reached: {self.max_grad_accum_steps}")
        
        # Double gradient accumulation steps
        old_steps = self.grad_accum_steps
        self.grad_accum_steps *= 2
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Update metrics
        oom_recovery_count.labels(rank=self.rank).inc()
        grad_accumulation_steps.labels(rank=self.rank).set(self.grad_accum_steps)
        
        logger.warning(f"OOM recovery: grad_accum_steps {old_steps} -> {self.grad_accum_steps}")

    def training_step(self, batch: torch.Tensor, targets: torch.Tensor, 
                     step: int) -> Tuple[float, Dict[str, float]]:
        """Enhanced training step with OOM recovery."""
        metrics = {}
        
        try:
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.autocast_enabled):
                outputs = self.model(batch)
                loss = nn.functional.cross_entropy(outputs, targets)
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                metrics['grad_norm'] = grad_norm.item()
            
            metrics['loss'] = loss.item() * self.grad_accum_steps
            return loss.item() * self.grad_accum_steps, metrics
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM detected at step {step}: {e}")
                self._handle_oom_recovery()
                # Retry with increased gradient accumulation
                torch.cuda.empty_cache()
                return self.training_step(batch, targets, step)
            else:
                raise e

    def benchmark_amp_speedup(self, test_batch: torch.Tensor, test_targets: torch.Tensor, 
                            num_iterations: int = 100) -> float:
        """Benchmark AMP speedup vs FP32."""
        self.model.eval()
        
        # Warmup
        for _ in range(10):
            with torch.cuda.amp.autocast(enabled=False):
                _ = self.model(test_batch)
        
        # FP32 benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model(test_batch)
                loss = nn.functional.cross_entropy(outputs, test_targets)
                loss.backward()
        torch.cuda.synchronize()
        fp32_time = time.time() - start_time
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # AMP benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(test_batch)
                loss = nn.functional.cross_entropy(outputs, test_targets)
            self.scaler.scale(loss).backward()
        torch.cuda.synchronize()
        amp_time = time.time() - start_time
        
        speedup_ratio = fp32_time / amp_time
        amp_speedup_ratio.labels(rank=self.rank).set(speedup_ratio)
        
        logger.info(f"AMP Speedup: {speedup_ratio:.2f}x (FP32: {fp32_time:.3f}s, AMP: {amp_time:.3f}s)")
        self.model.train()
        return speedup_ratio

# -----------------------------------------------------------------------------
# Distributed Setup
# -----------------------------------------------------------------------------
def init_distributed():
    if not dist.is_initialized():
        try:
            dist.init_process_group("nccl")
        except Exception as e:
            logger.warning(f"Failed to initialize distributed: {e}")
            return 0, 1
    return dist.get_rank(), dist.get_world_size()

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Enhanced AMP Training")
    parser.add_argument("--config", type=str, default="configs/amp_training.yaml")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'amp_enabled': True,
        'activation_checkpointing': True,
        'oom_recovery_enabled': True,
        'initial_grad_accum_steps': 1,
        'max_grad_accum_steps': 64,
        'checkpoint_dir': './checkpoints',
        'save_every_n_steps': 1000
    }
    
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Initialize model and trainer
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 10)
    ).cuda()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = AMPTrainer(model, optimizer, config)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.checkpoint:
        checkpoint_data = trainer.load_checkpoint(Path(args.checkpoint))
        start_step = checkpoint_data['step'] + 1
    
    logger.info("Enhanced AMP training started")

if __name__ == "__main__":
    main()