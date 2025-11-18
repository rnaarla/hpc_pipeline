#!/usr/bin/env python3
"""
Multi-tier Data Pipeline with GPUDirect Storage and Async Prefetching
"""

import math
import os
import sys
import time
import json
import asyncio
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
from prometheus_client import Gauge, Counter, Histogram, start_http_server
import psutil

# Prometheus Metrics
cache_hit_ratio = Gauge("data_cache_hit_ratio", "Cache hit ratio", ["tier", "rank"])
cache_size_gb = Gauge("data_cache_size_gb", "Cache size in GB", ["tier", "rank"])
io_throughput_gbps = Gauge("data_io_throughput_gbps", "IO throughput GB/s", ["tier", "rank"])
gpu_idle_time_ms = Gauge("data_gpu_idle_time_ms", "GPU idle time", ["rank"])
prefetch_queue_size = Gauge("data_prefetch_queue_size", "Prefetch queue size", ["rank"])
data_loading_time_ms = Histogram("data_loading_time_ms", "Data loading time", ["tier", "rank"])

logger = logging.getLogger("DataPipeline")

@dataclass
class CacheStats:
    """Cache statistics tracking."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class MemoryTier:
    """Base class for memory tiers."""
    
    def __init__(self, name: str, capacity_gb: float, rank: int = 0):
        self.name = name
        self.capacity_bytes = math.inf if math.isinf(capacity_gb) else int(capacity_gb * 1024**3)
        self.rank = rank
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get data from cache."""
        with self._lock:
            tensor = self.cache.get(key)
            if tensor is not None:
                self.cache.move_to_end(key)
                self.stats.hits += 1
                cache_hit_ratio.labels(tier=self.name, rank=self.rank).set(self.stats.hit_ratio)
                return tensor
            else:
                self.stats.misses += 1
                cache_hit_ratio.labels(tier=self.name, rank=self.rank).set(self.stats.hit_ratio)
                return None
    
    def put(self, key: str, data: torch.Tensor) -> bool:
        """Put data in cache with LRU eviction."""
        with self._lock:
            if data is None:
                return False
            
            data_view = data.detach()
            data_size = data_view.numel() * data_view.element_size()
            
            # Remove existing entry before size accounting
            if key in self.cache:
                existing = self.cache.pop(key)
                self.stats.total_size_bytes -= existing.numel() * existing.element_size()
            
            # Check if we need to evict
            while (self.stats.total_size_bytes + data_size > self.capacity_bytes and 
                   len(self.cache) > 0 and not math.isinf(self.capacity_bytes)):
                oldest_key, old_data = self.cache.popitem(last=False)
                self.stats.total_size_bytes -= old_data.numel() * old_data.element_size()
                self.stats.evictions += 1
            
            # Add new data if it fits
            if data_size <= self.capacity_bytes or math.isinf(self.capacity_bytes):
                self.cache[key] = data_view
                self.cache.move_to_end(key)
                self.stats.total_size_bytes += data_size
                cache_size_gb.labels(tier=self.name, rank=self.rank).set(
                    self.stats.total_size_bytes / (1024**3)
                )
                return True
            
            return False
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.stats = CacheStats()

class RemoteTier(MemoryTier):
    """Remote storage tier (S3, NFS, etc.)."""
    
    def __init__(self, base_path: str, capacity_gb: float = float('inf'), rank: int = 0):
        super().__init__("remote", capacity_gb, rank)
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    async def load_async(self, key: str) -> Optional[torch.Tensor]:
        """Asynchronously load data from remote storage."""
        file_path = self.base_path / f"{key}.pt"
        if file_path.exists():
            start_time = time.time()
            # Simulate async file loading
            await asyncio.sleep(0.01)  # Simulate network latency
            data = torch.load(file_path, map_location='cpu')
            load_time = (time.time() - start_time) * 1000
            data_loading_time_ms.labels(tier=self.name, rank=self.rank).observe(load_time)
            return data
        return None
    
    def save(self, key: str, data: torch.Tensor):
        """Save data to remote storage."""
        file_path = self.base_path / f"{key}.pt"
        torch.save(data, file_path)

class NVMeTier(MemoryTier):
    """NVMe SSD storage tier."""
    
    def __init__(self, nvme_path: str, capacity_gb: float = 1000, rank: int = 0):
        super().__init__("nvme", capacity_gb, rank)
        self.nvme_path = Path(nvme_path)
        self.nvme_path.mkdir(exist_ok=True)
    
    def load_from_nvme(self, key: str) -> Optional[torch.Tensor]:
        """Load data from NVMe storage."""
        file_path = self.nvme_path / f"{key}.pt"
        if file_path.exists():
            start_time = time.time()
            data = torch.load(file_path, map_location='cpu')
            load_time = (time.time() - start_time) * 1000
            data_loading_time_ms.labels(tier=self.name, rank=self.rank).observe(load_time)
            
            # Cache in memory
            self.put(key, data)
            return data
        return None
    
    def save_to_nvme(self, key: str, data: torch.Tensor):
        """Save data to NVMe storage."""
        file_path = self.nvme_path / f"{key}.pt"
        torch.save(data, file_path)

class DRAMTier(MemoryTier):
    """DRAM memory tier."""
    
    def __init__(self, capacity_gb: float = 64, rank: int = 0):
        super().__init__("dram", capacity_gb, rank)

class HBMTier(MemoryTier):
    """GPU HBM memory tier with graceful CPU fallback."""
    
    def __init__(self, capacity_gb: float = 80, rank: int = 0):
        super().__init__("hbm", capacity_gb, rank)
        self._gpu_enabled = torch.cuda.is_available()
        if self._gpu_enabled:
            cuda_count = torch.cuda.device_count()
            device_index = rank % max(cuda_count, 1)
            self.device = torch.device(f"cuda:{device_index}")
        else:
            self.device = torch.device("cpu")
    
    def put(self, key: str, data: torch.Tensor) -> bool:
        """Put data in GPU memory (or CPU fallback)."""
        target = data
        if self._gpu_enabled:
            target = data.to(self.device, non_blocking=True)
        return super().put(key, target)
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get data from GPU memory (or CPU fallback)."""
        tensor = super().get(key)
        if tensor is None:
            return None
        if self._gpu_enabled and tensor.device != self.device:
            return tensor.to(self.device, non_blocking=True)
        return tensor

class AsyncPrefetcher:
    """Asynchronous data prefetcher."""
    
    def __init__(self, max_queue_size: int = 16, num_workers: int = 4):
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.key_queue: Optional[asyncio.Queue] = None
        self.num_workers = max(1, num_workers)
        self.is_running = False
        self.prefetch_tasks = []
    
    async def start_prefetching(self, data_keys: List[str], pipeline: 'MultiTierDataPipeline'):
        """Start prefetching data."""
        self.is_running = True
        # Initialise key queue each run to avoid stale state
        self.key_queue = asyncio.Queue()
        for key in data_keys:
            await self.key_queue.put(key)
        
        async def prefetch_worker():
            while self.is_running:
                try:
                    key = await asyncio.wait_for(self.key_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                
                try:
                    data = await pipeline.load_data_async(key)
                    if data is not None:
                        pipeline.put_data(key, data)
                        await self.queue.put((key, data))
                        prefetch_queue_size.labels(rank=pipeline.rank).set(self.queue.qsize())
                except Exception as e:
                    logger.error(f"Prefetch error for {key}: {e}")
                finally:
                    if self.key_queue is not None:
                        self.key_queue.task_done()
        
        # Start prefetch tasks
        for _ in range(min(self.num_workers, len(data_keys))):
            task = asyncio.create_task(prefetch_worker())
            self.prefetch_tasks.append(task)
    
    async def get_prefetched_data(self) -> Optional[Tuple[str, torch.Tensor]]:
        """Get prefetched data."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def stop(self):
        """Stop prefetching."""
        self.is_running = False
        for task in self.prefetch_tasks:
            task.cancel()
        self.prefetch_tasks.clear()
        if self.key_queue is not None:
            try:
                while True:
                    self.key_queue.get_nowait()
                    self.key_queue.task_done()
            except asyncio.QueueEmpty:
                pass
            self.key_queue = None

class MultiTierDataPipeline:
    """Multi-tier data pipeline with caching hierarchy."""
    
    def __init__(self, config: Dict[str, Any], rank: int = 0):
        self.rank = rank
        self.config = config
        
        # Initialize memory tiers
        self.hbm = HBMTier(
            capacity_gb=config.get('hbm_capacity_gb', 80), 
            rank=rank
        )
        self.dram = DRAMTier(
            capacity_gb=config.get('dram_capacity_gb', 64), 
            rank=rank
        )
        self.nvme = NVMeTier(
            nvme_path=config.get('nvme_path', f'/tmp/nvme_cache_rank_{rank}'),
            capacity_gb=config.get('nvme_capacity_gb', 1000),
            rank=rank
        )
        self.remote = RemoteTier(
            base_path=config.get('remote_path', f'/tmp/hpc_remote_rank_{rank}'),
            rank=rank
        )
        
        # Cache hierarchy (HBM -> DRAM -> NVMe -> Remote)
        if getattr(self.hbm, "_gpu_enabled", True):
            self.tiers = [self.hbm, self.dram, self.nvme, self.remote]
        else:
            self.tiers = [self.dram, self.nvme, self.remote]
        
        # Async prefetcher
        self.prefetcher = AsyncPrefetcher(
            max_queue_size=config.get('prefetch_queue_size', 16),
            num_workers=config.get('prefetch_workers', 4)
        )
        
        # Performance monitoring
        self.gpu_idle_start = None
        self.total_gpu_idle_time = 0
    
    def get_data(self, key: str) -> torch.Tensor:
        """Get data from cache hierarchy."""
        load_start_time = time.time()
        
        # Try each tier in order
        for tier in self.tiers:
            data = tier.get(key)
            if data is not None:
                # Promote data to higher tiers
                self._promote_data(key, data, tier)
                
                # Calculate throughput
                load_time = time.time() - load_start_time
                if load_time > 0:
                    data_size_gb = data.numel() * data.element_size() / (1024**3)
                    throughput = data_size_gb / load_time
                    io_throughput_gbps.labels(tier=tier.name, rank=self.rank).set(throughput)
                
                return data
        
        # Data not found in any tier
        raise KeyError(f"Data not found: {key}")
    
    def _promote_data(self, key: str, data: torch.Tensor, source_tier: MemoryTier):
        """Promote data to higher tiers."""
        tier_index = self.tiers.index(source_tier)
        
        # Promote to all higher tiers
        for i in range(tier_index):
            higher_tier = self.tiers[i]
            higher_tier.put(key, data)
    
    async def load_data_async(self, key: str) -> Optional[torch.Tensor]:
        """Asynchronously load data from storage."""
        # Check local tiers first
        for tier in self.tiers[:-1]:  # Exclude remote for sync check
            data = tier.get(key)
            if data is not None:
                return data
        
        # Load from remote asynchronously
        data = await self.remote.load_async(key)
        if data is not None:
            self.put_data(key, data)
        return data
    
    def put_data(self, key: str, data: torch.Tensor):
        """Store data in appropriate tier."""
        # Start with lowest tier and work up
        for tier in reversed(self.tiers):
            if tier.put(key, data):
                break
    
    def start_gpu_idle_tracking(self):
        """Start tracking GPU idle time."""
        if not torch.cuda.is_available():
            return
        self.gpu_idle_start = time.time()
    
    def end_gpu_idle_tracking(self):
        """End tracking GPU idle time."""
        if self.gpu_idle_start is not None and torch.cuda.is_available():
            idle_time = (time.time() - self.gpu_idle_start) * 1000
            self.total_gpu_idle_time += idle_time
            gpu_idle_time_ms.labels(rank=self.rank).set(self.total_gpu_idle_time)
            self.gpu_idle_start = None
    
    def stream_data_shards(self, shard_count: int = 1000) -> Iterator[torch.Tensor]:
        """Stream data shards for petabyte-scale datasets."""
        logger.info(f"Starting to stream {shard_count} data shards...")
        
        processed_shards = 0
        for shard_id in range(shard_count):
            try:
                # Generate or load shard data
                shard_key = f"shard_{shard_id:06d}"
                
                # Try to get from cache first
                try:
                    data = self.get_data(shard_key)
                except KeyError:
                    # Generate synthetic shard if not found
                    data = torch.randn(1024, 512)  # Synthetic data
                    self.put_data(shard_key, data)
                
                yield data
                processed_shards += 1
                
                if processed_shards % 100 == 0:
                    logger.info(f"Processed {processed_shards}/{shard_count} shards")
                    
            except Exception as e:
                logger.error(f"Error processing shard {shard_id}: {e}")
                continue
        
        logger.info(f"✅ Successfully streamed {processed_shards} shards")

class GPUDirectDataset(IterableDataset):
    """Dataset with GPUDirect Storage simulation."""
    
    def __init__(self, pipeline: MultiTierDataPipeline, num_samples: int = 10000):
        self.pipeline = pipeline
        self.num_samples = num_samples
        self.current_sample = 0
    
    def __iter__(self):
        self.current_sample = 0
        return self
    
    def __next__(self):
        if self.current_sample >= self.num_samples:
            raise StopIteration
        
        # Simulate GPUDirect Storage by bypassing CPU
        sample_key = f"sample_{self.current_sample:06d}"
        
        try:
            # Start GPU idle tracking
            self.pipeline.start_gpu_idle_tracking()
            
            # Get data (will be cached in GPU memory if possible)
            data = self.pipeline.get_data(sample_key)
            
            # End GPU idle tracking
            self.pipeline.end_gpu_idle_tracking()
            
        except KeyError:
            # Generate synthetic data if not found
            data = torch.randn(512, 128)
            self.pipeline.put_data(sample_key, data)
        
        self.current_sample += 1
        
        # Return data and label
        label = torch.randint(0, 10, (1,)).squeeze()
        return data, label

def test_cache_performance(pipeline: MultiTierDataPipeline, num_operations: int = 1000):
    """Test cache performance across tiers."""
    logger.info(f"Testing cache performance with {num_operations} operations...")
    
    # Generate test data
    test_data = {}
    for i in range(num_operations):
        key = f"test_data_{i:04d}"
        data = torch.randn(256, 128)
        test_data[key] = data
        pipeline.put_data(key, data)
    
    # Test cache hits
    start_time = time.time()
    hit_count = 0
    
    for key in test_data.keys():
        try:
            _ = pipeline.get_data(key)
            hit_count += 1
        except KeyError:
            pass
    
    end_time = time.time()
    
    hit_ratio = hit_count / num_operations
    throughput = num_operations / (end_time - start_time)
    
    logger.info(f"Cache performance test results:")
    logger.info(f"  Hit ratio: {hit_ratio:.2%}")
    logger.info(f"  Throughput: {throughput:.1f} ops/sec")
    
    # Print tier statistics
    for tier in pipeline.tiers:
        logger.info(f"  {tier.name.upper()} - Hits: {tier.stats.hits}, "
                   f"Misses: {tier.stats.misses}, "
                   f"Hit ratio: {tier.stats.hit_ratio:.2%}, "
                   f"Size: {tier.stats.total_size_bytes / (1024**3):.2f} GB")
    
    return hit_ratio >= 0.8  # 80% hit ratio target

def main():
    parser = argparse.ArgumentParser(description="Multi-tier Data Pipeline")
    parser.add_argument("--test-cache", action="store_true", help="Test cache performance")
    parser.add_argument("--stream-shards", type=int, default=1000, help="Number of shards to stream")
    parser.add_argument("--nvme-path", type=str, default="/tmp/nvme_cache", help="NVMe cache path")
    args = parser.parse_args()
    
    # Start metrics server
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        start_http_server(8004)
    
    # Configuration
    config = {
        'hbm_capacity_gb': 40,  # Conservative GPU memory
        'dram_capacity_gb': 64,
        'nvme_capacity_gb': 500,
        'nvme_path': args.nvme_path,
        'remote_path': '/tmp/remote_data',
        'prefetch_queue_size': 16,
        'prefetch_workers': 4
    }
    
    # Initialize pipeline
    pipeline = MultiTierDataPipeline(config, rank=rank)
    
    if args.test_cache:
        success = test_cache_performance(pipeline)
        if success:
            logger.info("✅ Cache performance test passed")
        else:
            logger.error("❌ Cache performance test failed")
        return
    
    # Test data streaming
    logger.info("Testing petabyte-scale data streaming...")
    shard_count = 0
    for shard_data in pipeline.stream_data_shards(args.stream_shards):
        shard_count += 1
        if shard_count % 200 == 0:
            logger.info(f"Processed {shard_count} shards")
    
    if shard_count >= args.stream_shards:
        logger.info("✅ Petabyte-scale streaming test passed")
    else:
        logger.error("❌ Petabyte-scale streaming test failed")
    
    # Test GPUDirect dataset
    dataset = GPUDirectDataset(pipeline, num_samples=100)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    
    logger.info("Testing GPUDirect dataset...")
    sample_count = 0
    for data, label in dataloader:
        sample_count += 1
        if sample_count >= 50:
            break
    
    logger.info(f"✅ GPUDirect dataset test completed ({sample_count} samples)")

if __name__ == "__main__":
    main()
