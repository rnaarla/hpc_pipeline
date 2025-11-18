#!/usr/bin/env python3
"""
GPUDirect Storage Integration for Data Pipeline
"""

import os
import cupy as cp
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from prometheus_client import Gauge, Counter

# Metrics
gds_throughput_gbps = Gauge("gds_throughput_gbps", "GPUDirect Storage throughput", ["rank"])
gds_latency_us = Gauge("gds_latency_us", "GPUDirect Storage latency", ["rank"])
gds_batch_load_time_ms = Gauge("gds_batch_load_time_ms", "Batch load time", ["rank"])

class GPUDirectStorage:
    """GPUDirect Storage integration for direct GPU memory access."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.gds_enabled = self._check_gds_support()
        
        if self.gds_enabled:
            self._initialize_gds()
    
    def _check_gds_support(self) -> bool:
        """Check if GPUDirect Storage is supported."""
        try:
            import cufile
            return True
        except ImportError:
            return False
    
    def _initialize_gds(self):
        """Initialize GPUDirect Storage."""
        if not self.gds_enabled:
            return
        
        try:
            import cufile
            cufile.init()
        except Exception as e:
            self.gds_enabled = False
            raise RuntimeError(f"Failed to initialize GPUDirect Storage: {e}")
    
    def load_tensor_gds(self, file_path: Path) -> Optional[torch.Tensor]:
        """Load tensor using GPUDirect Storage."""
        if not self.gds_enabled:
            return None
        
        try:
            # Use cupy for direct GPU memory allocation
            shape = self._get_tensor_shape(file_path)
            gpu_array = cp.zeros(shape, dtype=cp.float32)
            
            # Direct load to GPU memory
            import cufile
            with cufile.CUFile(str(file_path), "r") as f:
                f.read(gpu_array.data.ptr, gpu_array.nbytes)
            
            # Convert to PyTorch tensor
            return torch.as_tensor(gpu_array.get(), device="cuda")
            
        except Exception as e:
            raise RuntimeError(f"GPUDirect Storage load failed: {e}")
    
    def save_tensor_gds(self, tensor: torch.Tensor, file_path: Path):
        """Save tensor using GPUDirect Storage."""
        if not self.gds_enabled:
            return
        
        try:
            # Ensure tensor is on GPU
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            
            # Convert to cupy array
            gpu_array = cp.asarray(tensor.detach())
            
            # Direct save from GPU memory
            import cufile
            with cufile.CUFile(str(file_path), "w") as f:
                f.write(gpu_array.data.ptr, gpu_array.nbytes)
                
        except Exception as e:
            raise RuntimeError(f"GPUDirect Storage save failed: {e}")
    
    def _get_tensor_shape(self, file_path: Path) -> tuple:
        """Get tensor shape from metadata."""
        # In production, would read from metadata file
        # For demo, use fixed shape
        return (1024, 1024)
    
    def cleanup(self):
        """Cleanup GPUDirect Storage resources."""
        if self.gds_enabled:
            try:
                import cufile
                cufile.fini()
            except:
                pass

def main():
    # Example usage
    config = {
        'gds_buffer_size': 1024 * 1024  # 1MB buffer
    }
    
    gds = GPUDirectStorage(config)
    
    if gds.gds_enabled:
        # Example tensor operations
        tensor = torch.randn(1024, 1024, device="cuda")
        save_path = Path("/tmp/gds_test.tensor")
        
        # Save and load
        gds.save_tensor_gds(tensor, save_path)
        loaded_tensor = gds.load_tensor_gds(save_path)
        
        # Verify
        if loaded_tensor is not None:
            assert torch.allclose(tensor, loaded_tensor)
            print("✅ GPUDirect Storage test passed")
    
    gds.cleanup()

if __name__ == "__main__":
    main()
