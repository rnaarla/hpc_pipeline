#!/usr/bin/env python3
"""
Custom CUDA Kernels with Tensor Core Optimization
"""

import math
import torch
import triton
import triton.language as tl
from typing import Optional
from prometheus_client import Gauge, Histogram

# Metrics
kernel_flops = Gauge("kernel_flops_per_second", "Kernel FLOPS/s", ["kernel_name"])
kernel_efficiency = Gauge("kernel_efficiency_percent", "Kernel efficiency", ["kernel_name"])
kernel_latency_us = Histogram("kernel_latency_us", "Kernel latency", ["kernel_name"])

@triton.jit
def fused_matmul_gelu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, bias_ptr, output_ptr,
    # Matrix dimensions
    M, N, K,
    # Block dimensions
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Other configs
    GROUP_SIZE_M: tl.constexpr
):
    """Fused matrix multiply + bias + GELU kernel with Tensor Core optimization."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    
    # Compute group id and offset
    group_id = pid // GROUP_SIZE_M
    group_size = min(num_pid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    
    # Initialize pointers to A, B, bias
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load bias
    bias = tl.load(bias_ptr + offs_bn)
    
    # Outer loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptr + offs_am[:, None] * K + k + offs_k[None, :])
        b = tl.load(b_ptr + (k + offs_k[:, None]) * N + offs_bn[None, :])
        
        # Matrix multiplication
        acc += tl.dot(a, b)
    
    # Add bias
    acc += bias
    
    # Apply GELU activation
    # GELU(x) = x * Φ(x) where Φ is the CDF of the standard normal distribution
    sqrt_2_over_pi = 0.7978845608028654
    acc = acc * 0.5 * (1 + tl.math.tanh(sqrt_2_over_pi * (acc + 0.044715 * acc * acc * acc)))
    
    # Store result
    output = acc
    tl.store(output_ptr + offs_am[:, None] * N + offs_bn[None, :], output)

class FusedMatmulGELU(torch.nn.Module):
    """Fused matrix multiplication + bias + GELU using custom CUDA kernel."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias
        scale = 1.0 / math.sqrt(in_features)
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        
        # Configure Tensor Core usage
        self.use_tensor_cores = torch.cuda.get_device_capability()[0] >= 7
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Ensure contiguous inputs
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous()
        
        # Allocate output
        output = torch.empty(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        # Launch kernel
        grid = lambda META: (triton.cdiv(batch_size, META['BLOCK_SIZE_M']),)
        
        stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(stream)
        fused_matmul_gelu_kernel[grid](
            x, weight, bias, output,
            batch_size, self.out_features, self.in_features,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8
        )
        end_event.record(stream)
        end_event.synchronize()
        
        # Record metrics
        latency_us = start_event.elapsed_time(end_event) * 1000
        kernel_latency_us.labels(kernel_name="fused_matmul_gelu").observe(latency_us)
        
        # Calculate FLOPS
        flops = 2 * batch_size * self.out_features * self.in_features
        if latency_us > 0:
            flops_per_second = flops / (latency_us * 1e-6)
            kernel_flops.labels(kernel_name="fused_matmul_gelu").set(flops_per_second)
        else:
            flops_per_second = 0.0
        
        # Calculate efficiency vs theoretical peak
        if self.use_tensor_cores:
            peak_flops = 312e12  # A100 Tensor Core peak FLOPS
            if peak_flops > 0:
                efficiency = (flops_per_second / peak_flops) * 100
                kernel_efficiency.labels(kernel_name="fused_matmul_gelu").set(efficiency)
        
        return output

def benchmark_kernel(batch_size: int = 1024, in_features: int = 4096, 
                    out_features: int = 4096, num_iters: int = 100):
    """Benchmark custom kernel against PyTorch implementation."""
    # Create inputs
    x = torch.randn(batch_size, in_features, device='cuda')
    
    # Custom kernel implementation
    custom_layer = FusedMatmulGELU(in_features, out_features).cuda()
    
    # PyTorch implementation
    torch_linear = torch.nn.Linear(in_features, out_features).cuda()
    torch_gelu = torch.nn.GELU().cuda()
    
    # Warmup
    for _ in range(10):
        _ = custom_layer(x)
        _ = torch_gelu(torch_linear(x))
    
    torch.cuda.synchronize()
    
    # Benchmark custom kernel
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_iters):
        _ = custom_layer(x)
    end_time.record()
    torch.cuda.synchronize()
    
    custom_time = start_time.elapsed_time(end_time) / num_iters
    
    # Benchmark PyTorch
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_iters):
        _ = torch_gelu(torch_linear(x))
    end_time.record()
    torch.cuda.synchronize()
    
    torch_time = start_time.elapsed_time(end_time) / num_iters
    
    speedup = torch_time / custom_time
    print(f"Custom Kernel: {custom_time:.3f}ms")
    print(f"PyTorch: {torch_time:.3f}ms")
    print(f"Speedup: {speedup:.2f}x")
    
    return speedup >= 1.5  # Target 1.5x speedup

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    args = parser.parse_args()
    
    success = benchmark_kernel(
        batch_size=args.batch_size,
        in_features=args.in_features,
        out_features=args.out_features
    )
    
    if success:
        print("✅ Kernel performance test passed")
    else:
        print("❌ Kernel performance test failed")
        exit(1)

if __name__ == "__main__":
    main()
