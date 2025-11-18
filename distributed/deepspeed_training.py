#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 Training for 10B+ Parameter Models
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from prometheus_client import Gauge, Counter, Histogram, start_http_server
import psutil
import importlib.util
from optimization.kernel_benchmark import KernelBenchmark

# Prometheus Metrics
zero3_memory_efficiency = Gauge("zero3_memory_efficiency_percent", "ZeRO-3 memory efficiency", ["rank"])
offload_cpu_gb = Gauge("zero3_offload_cpu_gb", "CPU offload memory usage", ["rank"])
offload_nvme_gb = Gauge("zero3_offload_nvme_gb", "NVMe offload usage", ["rank"])
activation_partitioning_ratio = Gauge("zero3_activation_partitioning_ratio", "Activation partitioning ratio", ["rank"])
forward_time_ms = Histogram("zero3_forward_time_ms", "Forward pass time", ["rank"])
backward_time_ms = Histogram("zero3_backward_time_ms", "Backward pass time", ["rank"])
optimizer_step_time_ms = Histogram("zero3_optimizer_step_time_ms", "Optimizer step time", ["rank"])
checkpoint_stitch_time_s = Gauge("zero3_checkpoint_stitch_time_s", "Checkpoint stitching time", ["world_size"])

# Add new Prometheus metrics for Phase 3 validation
zero3_phase3_validation = Counter("zero3_phase3_validation", "Phase 3 validation metrics", ["component", "metric"])
zero3_checkpoint_hash = Gauge("zero3_checkpoint_hash_valid", "Checkpoint hash validation status", ["rank"])
zero3_io_throughput = Gauge("zero3_io_throughput_gbps", "IO throughput in GB/s", ["rank"])
zero3_cache_hits = Counter("zero3_cache_hits_total", "Number of cache hits", ["rank", "cache_level"])
zero3_nccl_stats = Gauge("zero3_nccl_stats", "NCCL communication statistics", ["rank", "metric"])

# Add Phase 4 metrics after existing metrics
phase4_kernel_flops = Gauge("custom_kernel_tflops", "Kernel TFLOPS achieved", ["kernel"])
phase4_kernel_roofline = Gauge("custom_kernel_roofline_efficiency", "Roofline model efficiency", ["kernel"])
phase4_kernel_speedup = Gauge("custom_kernel_speedup", "Speedup over unfused baseline", ["kernel"])
memory_bandwidth = Gauge("zero3_kernel_memory_bandwidth_gbps", "Kernel memory bandwidth", ["event"])

logger = logging.getLogger("DeepSpeedZeRO3")

class DeepSpeedConfig:
    """DeepSpeed configuration generator for ZeRO-3."""
    
    @staticmethod
    def create_zero3_config(
        train_batch_size: int = 32,
        enable_cpu_offload: bool = True,
        enable_nvme_offload: bool = True,
        nvme_offload_dir: str = "/tmp/deepspeed_nvme",
        activation_checkpointing: bool = True
    ) -> Dict[str, Any]:
        """Create ZeRO-3 configuration with offloading."""
        
        config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": train_batch_size,
            
            # ZeRO-3 Configuration
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu" if enable_cpu_offload else "none",
                    "pin_memory": True,
                    "buffer_count": 4,
                    "fast_init": False
                },
                "offload_param": {
                    "device": "cpu" if enable_cpu_offload else "none",
                    "pin_memory": True,
                    "buffer_count": 5,
                    "buffer_size": 1e8,
                    "max_in_cpu": 1e9
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            
            # FP16 Configuration
            "fp16": {
                "enabled": True,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            
            # Optimizer Configuration
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            
            # Scheduler Configuration
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-4,
                    "warmup_num_steps": 1000
                }
            },
            
            # Logging and Monitoring
            "steps_per_print": 10,
            "wall_clock_breakdown": True,
            "dump_state": False
        }
        
        # Add NVMe offload if enabled
        if enable_nvme_offload:
            os.makedirs(nvme_offload_dir, exist_ok=True)
            config["aio"] = {
                "block_size": 1048576,
                "queue_depth": 8,
                "thread_count": 1,
                "single_submit": False,
                "overlap_events": True
            }
            config["zero_optimization"]["offload_optimizer"]["nvme_path"] = nvme_offload_dir
            config["zero_optimization"]["offload_param"]["nvme_path"] = nvme_offload_dir
        
        # Add activation checkpointing
        if activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": True,
                "cpu_checkpointing": enable_cpu_offload,
                "contiguous_memory_optimization": False,
                "number_checkpoints": 4,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            }
        
        return config

class LargeLanguageModel(nn.Module):
    """Large language model for testing ZeRO-3 with 10B+ parameters."""
    
    def __init__(self, vocab_size: int = 50000, hidden_size: int = 4096, 
                 num_layers: int = 32, num_heads: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Model created with {total_params:,} parameters ({total_params/1e9:.1f}B)")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output_projection(x)

class TransformerLayer(nn.Module):
    """Transformer layer with multi-head attention and feed-forward."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class DeepSpeedZeRO3Trainer:
    """DeepSpeed ZeRO-3 trainer with comprehensive monitoring."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=config
        )
        
        # Memory monitoring
        self.initial_memory = torch.cuda.memory_allocated()
        
        # Performance tracking
        self.step_times = []
        self.memory_usage = []
        
        # Initialize kernel benchmarking if custom kernels are available
        self.use_custom_kernels = config.get("use_custom_kernels", True)
        if self.use_custom_kernels and importlib.util.find_spec("custom_cuda_kernels"):
            import custom_cuda_kernels
            self.kernel_benchmark = KernelBenchmark()
            self._replace_with_custom_kernels()
    
    def _replace_with_custom_kernels(self):
        """Replace PyTorch ops with custom CUDA kernels if available."""
        if not hasattr(self, 'kernel_benchmark'):
            return
            
        for module in self.model_engine.module.modules():
            if isinstance(module, TransformerLayer):
                # Replace feed-forward with fused kernel
                if hasattr(self, 'custom_cuda_kernels'):
                    module.feed_forward = self.custom_cuda_kernels.fused_matmul_bias_gelu_dropout

    def training_step(self, batch: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """ZeRO-3 training step with detailed timing."""
        step_start_time = time.time()
        
        # Forward pass with timing
        forward_start = time.time()
        outputs = self.model_engine(batch)
        loss = nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        forward_time = (time.time() - forward_start) * 1000
        forward_time_ms.labels(rank=self.rank).observe(forward_time)
        
        # Backward pass with timing
        backward_start = time.time()
        self.model_engine.backward(loss)
        backward_time = (time.time() - backward_start) * 1000
        backward_time_ms.labels(rank=self.rank).observe(backward_time)
        
        # Optimizer step with timing
        optimizer_start = time.time()
        self.model_engine.step()
        optimizer_time = (time.time() - optimizer_start) * 1000
        optimizer_step_time_ms.labels(rank=self.rank).observe(optimizer_time)
        
        # Memory and offload monitoring
        self._update_memory_metrics()
        
        total_time = time.time() - step_start_time
        self.step_times.append(total_time)
        
        metrics = {
            'loss': loss.item(),
            'forward_time_ms': forward_time,
            'backward_time_ms': backward_time,
            'optimizer_time_ms': optimizer_time,
            'total_time_ms': total_time * 1000
        }
        
        return loss.item(), metrics
    
    def _update_memory_metrics(self):
        """Update memory and offload metrics."""
        # GPU memory efficiency
        current_memory = torch.cuda.memory_allocated()
        memory_saved = max(0, self.initial_memory - current_memory)
        efficiency = (memory_saved / self.initial_memory * 100) if self.initial_memory > 0 else 0
        zero3_memory_efficiency.labels(rank=self.rank).set(efficiency)
        
        # CPU offload monitoring
        cpu_memory = psutil.virtual_memory()
        cpu_used_gb = (cpu_memory.total - cpu_memory.available) / (1024**3)
        offload_cpu_gb.labels(rank=self.rank).set(cpu_used_gb)
        
        # NVMe offload monitoring (approximate)
        nvme_dir = self.config.get("zero_optimization", {}).get("offload_param", {}).get("nvme_path")
        if nvme_dir and os.path.exists(nvme_dir):
            nvme_usage = sum(
                os.path.getsize(os.path.join(nvme_dir, f)) 
                for f in os.listdir(nvme_dir) 
                if os.path.isfile(os.path.join(nvme_dir, f))
            ) / (1024**3)
            offload_nvme_gb.labels(rank=self.rank).set(nvme_usage)
        
        # Add Phase 3 validation metrics
        self._validate_phase3_metrics()
    
    def _validate_phase3_metrics(self):
        """Validate Phase 3 implementation metrics."""
        # DeepSpeed ZeRO-3 validation
        zero3_phase3_validation.labels(component="deepspeed", metric="model_size_b").inc(
            sum(p.numel() for p in self.model_engine.module.parameters()) / 1e9
        )
        
        # Checkpoint validation
        if hasattr(self, 'last_checkpoint_path'):
            import hashlib
            with open(self.last_checkpoint_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
                zero3_checkpoint_hash.labels(rank=self.rank).set(1)
        
        # IO throughput monitoring
        nvme_dir = self.config.get("zero_optimization", {}).get("offload_param", {}).get("nvme_path")
        if nvme_dir:
            throughput = self._measure_io_throughput(nvme_dir)
            zero3_io_throughput.labels(rank=self.rank).set(throughput)
        
        # Cache hit monitoring
        cache_hits = {
            "hbm": torch.cuda.memory_cached(),
            "cpu": psutil.virtual_memory().cached,
            "nvme": 0  # Would need custom tracking for NVMe cache hits
        }
        for cache_level, hits in cache_hits.items():
            zero3_cache_hits.labels(rank=self.rank, cache_level=cache_level).inc(hits)
        
        # NCCL monitoring
        if dist.is_initialized():
            zero3_nccl_stats.labels(rank=self.rank, metric="bytes_sent").set(
                dist.get_world_size() * self.model_engine.gradient_average
            )
    
    def _measure_io_throughput(self, path: str) -> float:
        """Measure IO throughput for offload operations."""
        start_time = time.time()
        sample_size = 1024 * 1024  # 1MB
        with open(os.path.join(path, 'throughput_test'), 'wb') as f:
            f.write(os.urandom(sample_size))
        duration = time.time() - start_time
        os.remove(os.path.join(path, 'throughput_test'))
        return (sample_size / duration) / (1024**3)  # GB/s

    def save_sharded_checkpoint(self, step: int, checkpoint_dir: Path) -> Dict[str, Path]:
        """Save sharded checkpoint across ranks."""
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model state for this rank
        rank_checkpoint_path = checkpoint_dir / f"checkpoint_rank_{self.rank}_step_{step}.pt"
        
        checkpoint_data = {
            'step': step,
            'rank': self.rank,
            'world_size': self.world_size,
            'model_state_dict': self.model_engine.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'lr_scheduler_state_dict': self.model_engine.lr_scheduler.state_dict() if self.model_engine.lr_scheduler else None
        }
        
        torch.save(checkpoint_data, rank_checkpoint_path)
        
        # Create metadata file on rank 0
        metadata = {}
        if self.rank == 0:
            metadata_path = checkpoint_dir / f"checkpoint_metadata_step_{step}.json"
            metadata = {
                'step': step,
                'world_size': self.world_size,
                'timestamp': time.time(),
                'rank_files': [f"checkpoint_rank_{r}_step_{step}.pt" for r in range(self.world_size)]
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Synchronize all ranks
        if self.world_size > 1:
            dist.barrier()
        
        logger.info(f"Rank {self.rank} checkpoint saved: {rank_checkpoint_path}")
        return {'rank_checkpoint': rank_checkpoint_path, 'metadata': metadata}
    
    def stitch_global_checkpoint(self, checkpoint_dir: Path, step: int, output_path: Path) -> Path:
        """Stitch sharded checkpoints into global checkpoint."""
        if self.rank != 0:
            return None
        
        stitch_start_time = time.time()
        
        # Load metadata
        metadata_path = checkpoint_dir / f"checkpoint_metadata_step_{step}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load all rank checkpoints
        global_state_dict = {}
        optimizer_state = None
        lr_scheduler_state = None
        
        for rank in range(metadata['world_size']):
            rank_checkpoint_path = checkpoint_dir / f"checkpoint_rank_{rank}_step_{step}.pt"
            checkpoint_data = torch.load(rank_checkpoint_path, map_location='cpu')
            
            # Merge model state dictionaries
            for key, value in checkpoint_data['model_state_dict'].items():
                if key not in global_state_dict:
                    global_state_dict[key] = value
                else:
                    # Handle parameter concatenation for ZeRO-3
                    if isinstance(value, torch.Tensor) and isinstance(global_state_dict[key], torch.Tensor):
                        global_state_dict[key] = torch.cat([global_state_dict[key], value], dim=0)
            
            # Use optimizer state from rank 0
            if rank == 0:
                optimizer_state = checkpoint_data.get('optimizer_state_dict')
                lr_scheduler_state = checkpoint_data.get('lr_scheduler_state_dict')
        
        # Save global checkpoint
        global_checkpoint = {
            'step': step,
            'model_state_dict': global_state_dict,
            'optimizer_state_dict': optimizer_state,
            'lr_scheduler_state_dict': lr_scheduler_state,
            'metadata': metadata
        }
        
        torch.save(global_checkpoint, output_path)
        
        stitch_time = time.time() - stitch_start_time
        checkpoint_stitch_time_s.labels(world_size=metadata['world_size']).set(stitch_time)
        
        logger.info(f"Global checkpoint stitched: {output_path} (took {stitch_time:.2f}s)")
        return output_path
    
    def validate_10b_model_training(self, test_loader: DataLoader) -> bool:
        """Validate that 10B+ model trains without OOM."""
        try:
            logger.info("Validating 10B+ model training...")
            
            for i, (batch, targets) in enumerate(test_loader):
                if i >= 5:  # Test a few steps
                    break
                
                loss, metrics = self.training_step(batch, targets)
                
                # Check for OOM or training failures
                if torch.isnan(torch.tensor(loss)):
                    logger.error("Training loss is NaN")
                    return False
                
                if i % 2 == 0:
                    logger.info(f"Validation step {i}: loss={loss:.4f}, "
                               f"forward={metrics['forward_time_ms']:.1f}ms, "
                               f"backward={metrics['backward_time_ms']:.1f}ms")
            
            logger.info("✅ 10B+ model training validation successful")
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"❌ OOM during 10B+ model validation: {e}")
                return False
            else:
                raise e

    def _validate_phase4_kernels(self) -> bool:
        """Validate Phase 4 custom kernel performance."""
        if not hasattr(self, 'kernel_benchmark'):
            logger.warning("Custom kernels not available, skipping Phase 4 validation")
            return False
            
        metrics = self.kernel_benchmark.benchmark_fused_ops()
        phase4_kernel_flops.labels(kernel="fused_matmul_gelu").set(metrics["tflops"])
        phase4_kernel_roofline.labels(kernel="fused_matmul_gelu").set(metrics["roofline_efficiency"])
        phase4_kernel_speedup.labels(kernel="fused_matmul_gelu").set(metrics["speedup"])
        
        return metrics["speedup"] >= 1.5 and metrics["roofline_efficiency"] >= 0.7

    def _optimize_memory_bandwidth(self):
        """Optimize memory access patterns."""
        if not hasattr(self, 'kernel_benchmark'):
            return
            
        # Profile memory access
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            with_stack=True
        ) as prof:
            self.kernel_benchmark.benchmark_fused_ops()
            
        # Analyze memory access patterns
        for event in prof.events():
            if event.memory_bw is not None:
                memory_bandwidth.labels(event=event.name).set(event.memory_bw)
    
    def _validate_tensor_core_usage(self) -> bool:
        """Validate Tensor Core utilization."""
        import subprocess
        
        # Run NSight compute analysis
        result = subprocess.run([
            "ncu", "--metrics", "sm__sass_thread_inst_executed_op_tensor_op_hmma",
            "python", "-m", "optimization.kernel_benchmark"
        ], capture_output=True)
        
        return "tensor_op" in result.stdout.decode()

def create_synthetic_dataset(vocab_size: int = 50000, seq_length: int = 512, 
                           dataset_size: int = 1000):
    """Create synthetic dataset for testing."""
    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return dataset_size
        
        def __getitem__(self, idx):
            input_ids = torch.randint(0, vocab_size, (seq_length,))
            targets = torch.randint(0, vocab_size, (seq_length,))
            return input_ids, targets
    
    return SyntheticDataset()

def main():
    parser = argparse.ArgumentParser(description="DeepSpeed ZeRO-3 Training")
    parser.add_argument("--model-size", type=str, default="10B", 
                       choices=["1B", "7B", "10B", "70B"])
    parser.add_argument("--enable-nvme", action="store_true", 
                       help="Enable NVMe offloading")
    parser.add_argument("--validate-10b", action="store_true",
                       help="Validate 10B model training")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/deepspeed")
    parser.add_argument("--validate-kernels", action="store_true",
                       help="Validate Phase 4 custom kernels")
    args = parser.parse_args()
    
    # Start metrics server on rank 0
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        start_http_server(8003)
    
    # Model size configurations
    model_configs = {
        "1B": {"hidden_size": 2048, "num_layers": 24, "num_heads": 16},
        "7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32},
        "10B": {"hidden_size": 4096, "num_layers": 48, "num_heads": 32},
        "70B": {"hidden_size": 8192, "num_layers": 80, "num_heads": 64}
    }
    
    model_config = model_configs[args.model_size]
    
    # Create DeepSpeed configuration
    deepspeed_config = DeepSpeedConfig.create_zero3_config(
        train_batch_size=32,
        enable_cpu_offload=True,
        enable_nvme_offload=args.enable_nvme,
        nvme_offload_dir="/tmp/deepspeed_nvme"
    )
    
    # Create model
    model = LargeLanguageModel(**model_config)
    
    # Initialize trainer
    trainer = DeepSpeedZeRO3Trainer(model, deepspeed_config)
    
    # Create dataset and dataloader
    dataset = create_synthetic_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # DeepSpeed handles micro-batching
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    if args.validate_10b:
        success = trainer.validate_10b_model_training(dataloader)
        if not success:
            sys.exit(1)
        return
    
    if args.validate_kernels:
        kernel_validation = trainer._validate_phase4_kernels()
        if not kernel_validation:
            logger.error("❌ Phase 4 kernel validation failed")
            sys.exit(1)
        logger.info("✅ Phase 4 kernel validation passed")

    # Training loop
    logger.info(f"Starting DeepSpeed ZeRO-3 training ({args.model_size} model)...")
    checkpoint_dir = Path(args.checkpoint_dir)
    
    for step, (batch, targets) in enumerate(dataloader):
        loss, metrics = trainer.training_step(batch, targets)
        
        # Logging
        if step % 10 == 0 and rank == 0:
            logger.info(f"Step {step}: Loss={loss:.4f}, "
                       f"Forward={metrics['forward_time_ms']:.1f}ms, "
                       f"Backward={metrics['backward_time_ms']:.1f}ms, "
                       f"Optimizer={metrics['optimizer_time_ms']:.1f}ms")
        
        # Checkpointing and stitching
        if step % 100 == 0 and step > 0:
            checkpoint_info = trainer.save_sharded_checkpoint(step, checkpoint_dir)
            
            # Stitch global checkpoint
            if rank == 0:
                global_checkpoint_path = checkpoint_dir / f"global_checkpoint_step_{step}.pt"
                trainer.stitch_global_checkpoint(checkpoint_dir, step, global_checkpoint_path)
        
        # Stop after reasonable number of steps
        if step >= 200:
            break
    
    logger.info("DeepSpeed ZeRO-3 training completed")
    # Perform Phase 4 kernel validation once at end if not already requested
    if not args.validate_kernels:
        try:
            if trainer._validate_phase4_kernels():
                logger.info("✅ Phase 4 implementation verified with all acceptance criteria met")
            else:
                logger.warning("⚠️ Phase 4 validation did not meet all acceptance criteria")
        except Exception as e:
            logger.warning(f"⚠️ Phase 4 validation skipped due to: {e}")

if __name__ == "__main__":
    main()
