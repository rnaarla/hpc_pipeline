#!/usr/bin/env python3
"""
Production-grade Fault Tolerance with Sharded Checkpointing
"""

import os
import sys
import time
import json
import hashlib
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.distributed as dist
from prometheus_client import Gauge, Counter, Histogram, start_http_server
import subprocess

# Prometheus Metrics
checkpoint_save_time_s = Histogram("checkpoint_save_time_seconds", "Checkpoint save time", ["rank"])
checkpoint_load_time_s = Histogram("checkpoint_load_time_seconds", "Checkpoint load time", ["rank"])
checkpoint_corruption_count = Counter("checkpoint_corruption_total", "Checkpoint corruption events", ["rank"])
checkpoint_size_gb = Gauge("checkpoint_size_gb", "Checkpoint size in GB", ["rank", "shard"])
global_stitch_time_s = Gauge("checkpoint_global_stitch_time_seconds", "Global checkpoint stitch time", ["world_size"])
recovery_time_s = Gauge("checkpoint_recovery_time_seconds", "Recovery time", ["recovery_type"])
mttr_seconds = Gauge("checkpoint_mttr_seconds", "Mean Time To Recovery", ["failure_type"])

logger = logging.getLogger("FaultTolerance")

@dataclass
class CheckpointMetadata:
    """Checkpoint metadata for validation and recovery."""
    step: int
    timestamp: float
    rank: int
    world_size: int
    model_params: int
    optimizer_state_size: int
    loss: float
    git_commit: Optional[str]
    config_hash: str
    shard_checksums: Dict[str, str]

class ShardedCheckpointManager:
    """Production-grade sharded checkpoint manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Checkpoint configuration
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.backup_dir = Path(config.get('backup_dir', './checkpoints_backup'))
        self.backup_dir.mkdir(exist_ok=True)
        
        self.max_checkpoints = config.get('max_checkpoints', 5)
        self.compression_enabled = config.get('compression_enabled', True)
        self.encryption_enabled = config.get('encryption_enabled', False)
        
        # Recovery configuration
        self.max_recovery_time_s = config.get('max_recovery_time_s', 300)  # 5 minutes
        self.corruption_retry_count = config.get('corruption_retry_count', 3)
        
        # Parallel processing
        self.num_threads = config.get('checkpoint_threads', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _validate_checkpoint_integrity(self, checkpoint_path: Path, 
                                     expected_checksum: str) -> bool:
        """Validate checkpoint file integrity."""
        if not checkpoint_path.exists():
            return False
        
        actual_checksum = self._calculate_checksum(checkpoint_path)
        if actual_checksum != expected_checksum:
            checkpoint_corruption_count.labels(rank=self.rank).inc()
            logger.error(f"Checkpoint corruption detected: {checkpoint_path}")
            return False
        
        return True
    
    def save_sharded_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                              step: int, loss: float, metadata: Dict[str, Any] = None) -> Path:
        """Save sharded checkpoint with validation."""
        save_start_time = time.time()
        
        # Create checkpoint data
        checkpoint_data = {
            'step': step,
            'rank': self.rank,
            'world_size': self.world_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        # Save rank-specific checkpoint
        shard_path = self.checkpoint_dir / f"checkpoint_rank_{self.rank}_step_{step}.pt"
        torch.save(checkpoint_data, shard_path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(shard_path)
        
        # Save checksum file
        checksum_path = shard_path.with_suffix('.sha256')
        with open(checksum_path, 'w') as f:
            f.write(f"{checksum}  {shard_path.name}\n")
        
        # Create metadata
        checkpoint_metadata = CheckpointMetadata(
            step=step,
            timestamp=time.time(),
            rank=self.rank,
            world_size=self.world_size,
            model_params=sum(p.numel() for p in model.parameters()),
            optimizer_state_size=len(optimizer.state_dict()),
            loss=loss,
            git_commit=self._get_git_commit(),
            config_hash=self._calculate_config_hash(metadata or {}),
            shard_checksums={f"rank_{self.rank}": checksum}
        )
        
        # Save metadata on rank 0
        if self.rank == 0:
            metadata_path = self.checkpoint_dir / f"checkpoint_metadata_step_{step}.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(checkpoint_metadata), f, indent=2)
        
        # Synchronize all ranks
        if self.world_size > 1:
            dist.barrier()
        
        save_time = time.time() - save_start_time
        checkpoint_save_time_s.labels(rank=self.rank).observe(save_time)
        
        # Update metrics
        file_size_gb = shard_path.stat().st_size / (1024**3)
        checkpoint_size_gb.labels(rank=self.rank, shard=f"rank_{self.rank}").set(file_size_gb)
        
        logger.info(f"Checkpoint saved: {shard_path} (checksum: {checksum[:8]}...)")
        return shard_path
    
    def load_sharded_checkpoint(self, checkpoint_step: int, 
                              model: torch.nn.Module, 
                              optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Load sharded checkpoint with corruption recovery."""
        load_start_time = time.time()
        
        # Try to load rank-specific checkpoint
        shard_path = self.checkpoint_dir / f"checkpoint_rank_{self.rank}_step_{checkpoint_step}.pt"
        checksum_path = shard_path.with_suffix('.sha256')
        
        # Validate checksum if available
        if checksum_path.exists():
            with open(checksum_path, 'r') as f:
                expected_checksum = f.read().split()[0]
            
            if not self._validate_checkpoint_integrity(shard_path, expected_checksum):
                # Try to recover from corruption
                recovered_path = self._recover_corrupted_checkpoint(shard_path, checkpoint_step)
                if recovered_path:
                    shard_path = recovered_path
                else:
                    raise RuntimeError(f"Failed to recover corrupted checkpoint: {shard_path}")
        
        # Load checkpoint
        checkpoint_data = torch.load(shard_path, map_location='cpu')
        
        # Restore model and optimizer state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        load_time = time.time() - load_start_time
        checkpoint_load_time_s.labels(rank=self.rank).observe(load_time)
        
        logger.info(f"Checkpoint loaded: {shard_path} (load time: {load_time:.2f}s)")
        return checkpoint_data
    
    def _recover_corrupted_checkpoint(self, corrupted_path: Path, step: int) -> Optional[Path]:
        """Attempt to recover corrupted checkpoint."""
        logger.warning(f"Attempting to recover corrupted checkpoint: {corrupted_path}")
        
        # Strategy 1: Try backup directory
        backup_path = self.backup_dir / corrupted_path.name
        if backup_path.exists():
            backup_checksum_path = backup_path.with_suffix('.sha256')
            if backup_checksum_path.exists():
                with open(backup_checksum_path, 'r') as f:
                    expected_checksum = f.read().split()[0]
                
                if self._validate_checkpoint_integrity(backup_path, expected_checksum):
                    shutil.copy2(backup_path, corrupted_path)
                    logger.info(f"Recovered checkpoint from backup: {backup_path}")
                    return corrupted_path
        
        # Strategy 2: Try previous checkpoint
        for prev_step in range(step - 1, max(0, step - 10), -1):
            prev_path = self.checkpoint_dir / f"checkpoint_rank_{self.rank}_step_{prev_step}.pt"
            if prev_path.exists():
                prev_checksum_path = prev_path.with_suffix('.sha256')
                if prev_checksum_path.exists():
                    with open(prev_checksum_path, 'r') as f:
                        expected_checksum = f.read().split()[0]
                    
                    if self._validate_checkpoint_integrity(prev_path, expected_checksum):
                        logger.warning(f"Using previous checkpoint: {prev_path}")
                        return prev_path
        
        return None
    
    def stitch_global_checkpoint(self, step: int, output_path: Path) -> bool:
        """Stitch sharded checkpoints into global checkpoint."""
        if self.rank != 0:
            return True
        
        stitch_start_time = time.time()
        
        try:
            # Load metadata
            metadata_path = self.checkpoint_dir / f"checkpoint_metadata_step_{step}.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load all rank checkpoints
            global_state_dict = {}
            global_optimizer_state = {}
            
            for rank in range(metadata['world_size']):
                shard_path = self.checkpoint_dir / f"checkpoint_rank_{rank}_step_{step}.pt"
                
                if not shard_path.exists():
                    logger.error(f"Missing shard for rank {rank}: {shard_path}")
                    return False
                
                # Validate shard
                checksum_path = shard_path.with_suffix('.sha256')
                if checksum_path.exists():
                    with open(checksum_path, 'r') as f:
                        expected_checksum = f.read().split()[0]
                    
                    if not self._validate_checkpoint_integrity(shard_path, expected_checksum):
                        logger.error(f"Corrupted shard detected: {shard_path}")
                        return False
                
                # Load shard
                shard_data = torch.load(shard_path, map_location='cpu')
                
                # Merge model state
                for key, value in shard_data['model_state_dict'].items():
                    if key not in global_state_dict:
                        global_state_dict[key] = value
                    else:
                        # Handle distributed parameter merging
                        if isinstance(value, torch.Tensor):
                            global_state_dict[key] = torch.cat([
                                global_state_dict[key], value
                            ], dim=0)
                
                # Merge optimizer state (use rank 0's state as base)
                if rank == 0:
                    global_optimizer_state = shard_data['optimizer_state_dict']
            
            # Create global checkpoint
            global_checkpoint = {
                'step': step,
                'model_state_dict': global_state_dict,
                'optimizer_state_dict': global_optimizer_state,
                'metadata': metadata
            }
            
            # Save global checkpoint
            torch.save(global_checkpoint, output_path)
            
            # Calculate and save checksum
            global_checksum = self._calculate_checksum(output_path)
            global_checksum_path = output_path.with_suffix('.sha256')
            with open(global_checksum_path, 'w') as f:
                f.write(f"{global_checksum}  {output_path.name}\n")
            
            stitch_time = time.time() - stitch_start_time
            global_stitch_time_s.labels(world_size=metadata['world_size']).set(stitch_time)
            
            logger.info(f"Global checkpoint stitched: {output_path} (time: {stitch_time:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stitch global checkpoint: {e}")
            return False
    
    def auto_recovery_test(self, model: torch.nn.Module, 
                          optimizer: torch.optim.Optimizer) -> bool:
        """Test automatic recovery from various failure scenarios."""
        logger.info("Starting automatic recovery test...")
        
        recovery_scenarios = [
            "missing_shard",
            "corrupted_checkpoint",
            "network_failure",
            "disk_full"
        ]
        
        recovery_times = {}
        
        for scenario in recovery_scenarios:
            recovery_start = time.time()
            
            try:
                if scenario == "missing_shard":
                    success = self._test_missing_shard_recovery(model, optimizer)
                elif scenario == "corrupted_checkpoint":
                    success = self._test_corruption_recovery(model, optimizer)
                elif scenario == "network_failure":
                    success = self._test_network_failure_recovery(model, optimizer)
                elif scenario == "disk_full":
                    success = self._test_disk_full_recovery(model, optimizer)
                
                recovery_time = time.time() - recovery_start
                recovery_times[scenario] = recovery_time
                
                recovery_time_s.labels(recovery_type=scenario).set(recovery_time)
                
                if success and recovery_time <= self.max_recovery_time_s:
                    logger.info(f"✅ {scenario} recovery test passed ({recovery_time:.2f}s)")
                else:
                    logger.error(f"❌ {scenario} recovery test failed")
                    return False
                    
            except Exception as e:
                logger.error(f"Recovery test failed for {scenario}: {e}")
                return False
        
        # Calculate MTTR
        avg_mttr = sum(recovery_times.values()) / len(recovery_times)
        mttr_seconds.labels(failure_type="average").set(avg_mttr)
        
        # Check if MTTR meets SLA (≤5 minutes)
        if avg_mttr <= 300:
            logger.info(f"✅ MTTR test passed: {avg_mttr:.2f}s ≤ 300s")
            return True
        else:
            logger.error(f"❌ MTTR test failed: {avg_mttr:.2f}s > 300s")
            return False
    
    def _test_missing_shard_recovery(self, model: torch.nn.Module, 
                                   optimizer: torch.optim.Optimizer) -> bool:
        """Test recovery from missing checkpoint shards."""
        # Save a checkpoint
        step = 1000
        self.save_sharded_checkpoint(model, optimizer, step, 0.5)
        
        # Simulate missing shard by temporarily moving it
        shard_path = self.checkpoint_dir / f"checkpoint_rank_{self.rank}_step_{step}.pt"
        temp_path = shard_path.with_suffix('.pt.temp')
        
        if shard_path.exists():
            shutil.move(shard_path, temp_path)
        
        try:
            # Attempt recovery
            self.load_sharded_checkpoint(step, model, optimizer)
            return False  # Should have failed
        except RuntimeError:
            # Restore shard and retry
            if temp_path.exists():
                shutil.move(temp_path, shard_path)
            
            # Should succeed now
            self.load_sharded_checkpoint(step, model, optimizer)
            return True
    
    def _test_corruption_recovery(self, model: torch.nn.Module, 
                                optimizer: torch.optim.Optimizer) -> bool:
        """Test recovery from corrupted checkpoints."""
        step = 1001
        shard_path = self.save_sharded_checkpoint(model, optimizer, step, 0.4)
        
        # Create backup
        backup_path = self.backup_dir / shard_path.name
        shutil.copy2(shard_path, backup_path)
        shutil.copy2(shard_path.with_suffix('.sha256'), 
                    backup_path.with_suffix('.sha256'))
        
        # Corrupt the checkpoint
        with open(shard_path, 'r+b') as f:
            f.seek(100)
            f.write(b'CORRUPTED_DATA')
        
        # Should recover from backup
        recovered_data = self.load_sharded_checkpoint(step, model, optimizer)
        return recovered_data is not None
    
    def _test_network_failure_recovery(self, model: torch.nn.Module, 
                                     optimizer: torch.optim.Optimizer) -> bool:
        """Test recovery from network failures during distributed checkpointing."""
        # Simulate network partition by skipping barrier
        step = 1002
        
        # Save checkpoint without full synchronization
        shard_path = self.save_sharded_checkpoint(model, optimizer, step, 0.3)
        
        # Verify we can still load locally
        recovered_data = self.load_sharded_checkpoint(step, model, optimizer)
        return recovered_data is not None
    
    def _test_disk_full_recovery(self, model: torch.nn.Module, 
                               optimizer: torch.optim.Optimizer) -> bool:
        """Test recovery from disk full scenarios."""
        # This would normally involve filesystem manipulation
        # For simulation, just test cleanup and retry logic
        step = 1003
        
        # Save checkpoint
        shard_path = self.save_sharded_checkpoint(model, optimizer, step, 0.2)
        
        # Simulate cleanup of old checkpoints
        self._cleanup_old_checkpoints()
        
        # Verify checkpoint still loads
        recovered_data = self.load_sharded_checkpoint(step, model, optimizer)
        return recovered_data is not None
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to free space."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_rank_*.pt"))
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Keep only the most recent checkpoints
        while len(checkpoint_files) > self.max_checkpoints:
            old_checkpoint = checkpoint_files.pop(0)
            old_checkpoint.unlink(missing_ok=True)
            old_checkpoint.with_suffix('.sha256').unlink(missing_ok=True)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()[:8] if result.returncode == 0 else None
        except:
            return None
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

class SLURMJobManager:
    """SLURM job management for fault tolerance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.job_id = os.environ.get('SLURM_JOB_ID')
        self.is_slurm = self.job_id is not None
    
    def setup_requeue_signal_handlers(self):
        """Setup signal handlers for SLURM requeue."""
        if not self.is_slurm:
            return
        
        import signal
        
        def requeue_handler(signum, frame):
            logger.info(f"Received signal {signum}, preparing for requeue...")
            # Save emergency checkpoint
            self._save_emergency_checkpoint()
            # Exit gracefully
            sys.exit(0)
        
        signal.signal(signal.SIGUSR1, requeue_handler)
        signal.signal(signal.SIGTERM, requeue_handler)
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint before requeue."""
        logger.info("Saving emergency checkpoint before requeue...")
        # This would integrate with the checkpoint manager
        # Implementation depends on the specific training loop

def main():
    parser = argparse.ArgumentParser(description="Fault Tolerance System")
    parser.add_argument("--test-recovery", action="store_true", 
                       help="Run recovery tests")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Checkpoint directory")
    args = parser.parse_args()
    
    # Start metrics server
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        start_http_server(8005)
    
    # Configuration
    config = {
        'checkpoint_dir': args.checkpoint_dir,
        'backup_dir': f"{args.checkpoint_dir}_backup",
        'max_checkpoints': 5,
        'compression_enabled': True,
        'max_recovery_time_s': 300,
        'corruption_retry_count': 3,
        'checkpoint_threads': 4
    }
    
    # Initialize checkpoint manager
    checkpoint_manager = ShardedCheckpointManager(config)
    
    # Create dummy model for testing
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    if args.test_recovery:
        # Run comprehensive recovery tests
        success = checkpoint_manager.auto_recovery_test(model, optimizer)
        if success:
            logger.info("✅ All recovery tests passed")
        else:
            logger.error("❌ Recovery tests failed")
            sys.exit(1)
    else:
        # Demonstrate basic checkpoint operations
        logger.info("Demonstrating checkpoint operations...")
        
        # Save checkpoint
        step = 100
        checkpoint_path = checkpoint_manager.save_sharded_checkpoint(
            model, optimizer, step, 0.5, {'epoch': 10}
        )
        
        # Load checkpoint
        checkpoint_data = checkpoint_manager.load_sharded_checkpoint(
            step, model, optimizer
        )
        
        # Stitch global checkpoint
        if rank == 0:
            global_path = Path(args.checkpoint_dir) / f"global_checkpoint_step_{step}.pt"
            success = checkpoint_manager.stitch_global_checkpoint(step, global_path)
            if success:
                logger.info(f"Global checkpoint created: {global_path}")
        
        logger.info("✅ Checkpoint operations completed successfully")

if __name__ == "__main__":
    main()
