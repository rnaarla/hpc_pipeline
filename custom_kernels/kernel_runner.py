#!/usr/bin/env python3
"""
kernel_runner.py
----------------
Custom CUDA kernels for fused operations and integration with PyTorch.

Features:
- Fused matmul + bias + GELU + dropout kernel
- PyTorch cpp_extension loader
- Benchmark harness vs cuBLAS/cuDNN
- Occupancy + roofline hooks
- Error handling and logging

Target audience: HPC / Principal Engineer level.
"""

import os
import sys
import time
import logging
import torch
from torch.utils.cpp_extension import load
import numpy as np
import traceback

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CustomKernelRunner")

# -----------------------------------------------------------------------------
# CUDA Source
# -----------------------------------------------------------------------------
cuda_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <curand_kernel.h>

// GELU approximation - more accurate version
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Improved dropout RNG with curand
__device__ __forceinline__ bool dropout_mask(curandState* state, float p) {
    return curand_uniform(state) > p;
}

// Initialize curand states
extern "C" __global__ void init_curand_states(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

extern "C" __global__ void fused_mm_bias_gelu_dropout(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    curandState* __restrict__ rand_states,
    int M, int N, int K,
    float dropout_p
) {
    // Optimized block tile sizes for better occupancy
    const int BM = 64;  // rows - reduced for better occupancy
    const int BN = 64;  // cols
    const int BK = 8;   // inner - reduced for faster shared memory access
    
    // Thread block mapping
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    // Thread-level tiling for better ILP
    const int TM = 4;  // thread tile M
    const int TN = 4;  // thread tile N
    
    // Register arrays for accumulation
    float acc[TM][TN] = {0.0f};
    
    // Calculate thread's responsibility
    int thread_row = by * BM + ty * TM;
    int thread_col = bx * BN + tx * TN;

    // Main computation loop
    for (int k_block = 0; k_block < (K + BK - 1) / BK; k_block++) {
        // Load A into shared memory with coalescing
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int row = thread_row + i;
            int col = k_block * BK + tx;
            As[ty * TM + i][tx] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }
        
        // Load B into shared memory with coalescing
        #pragma unroll
        for (int i = 0; i < TN; i++) {
            int row = k_block * BK + ty;
            int col = thread_col + i;
            Bs[ty][tx * TN + i] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    acc[m][n] += As[ty * TM + m][k] * Bs[k][tx * TN + n];
                }
            }
        }
        
        __syncthreads();
    }

    // Write results with fused operations
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int row = thread_row + m;
            int col = thread_col + n;
            
            if (row < M && col < N) {
                int idx = row * N + col;
                float val = acc[m][n] + bias[col];
                val = gelu(val);
                
                // Apply dropout if enabled
                if (dropout_p > 0.0f) {
                    curandState local_state = rand_states[idx];
                    bool keep = dropout_mask(&local_state, dropout_p);
                    rand_states[idx] = local_state;  // Update state
                    val = keep ? val / (1.0f - dropout_p) : 0.0f;
                }
                
                C[idx] = val;
            }
        }
    }
}
"""

# Add C++ wrapper code
cpp_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA kernel declarations
extern "C" void init_curand_states(curandState* states, unsigned long seed, int n);
extern "C" void fused_mm_bias_gelu_dropout(
    const float* A, const float* B, const float* bias, float* C,
    curandState* rand_states, int M, int N, int K, float dropout_p
);

torch::Tensor fused_mm_bias_gelu_dropout_wrapper(
    torch::Tensor A, torch::Tensor B, torch::Tensor bias,
    torch::Tensor rand_states, float dropout_p
) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Optimized grid/block dimensions
    dim3 threads(16, 16);  // 256 threads per block
    dim3 blocks((N + 63) / 64, (M + 63) / 64);
    
    fused_mm_bias_gelu_dropout<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), bias.data_ptr<float>(),
        C.data_ptr<float>(), (curandState*)rand_states.data_ptr<int64_t>(),
        M, N, K, dropout_p
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return C;
}

torch::Tensor init_rand_states(int64_t size, int64_t seed) {
    auto states = torch::zeros({size}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    
    dim3 threads(256);
    dim3 blocks((size + 255) / 256);
    
    init_curand_states<<<blocks, threads>>>(
        (curandState*)states.data_ptr<int64_t>(), seed, size
    );
    
    return states;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mm_bias_gelu_dropout", &fused_mm_bias_gelu_dropout_wrapper, "Fused matmul+bias+gelu+dropout");
    m.def("init_rand_states", &init_rand_states, "Initialize curand states");
}
"""

# -----------------------------------------------------------------------------
# Build Extension
# -----------------------------------------------------------------------------
try:
    fused_kernel = load(
        name="fused_mm_bias_gelu_dropout",
        sources=[cpp_src],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "--expt-relaxed-constexpr"],
        extra_ldflags=["-lcurand"],
        verbose=True,
        with_cuda=True
    )
    logger.info("✅ Custom CUDA kernel compiled successfully.")
except Exception as e:
    logger.error(f"❌ Kernel compilation failed: {e}")
    logger.error(traceback.format_exc())
    raise e

# -----------------------------------------------------------------------------
# Kernel Wrapper with Validation
# -----------------------------------------------------------------------------
def run_fused_kernel(A, B, bias, dropout_p=0.1, seed=1234):
    # Input validation
    if not all(t.is_cuda for t in [A, B, bias]):
        raise ValueError("All tensors must be on CUDA device")
    if not all(t.dtype == torch.float32 for t in [A, B, bias]):
        raise ValueError("All tensors must be float32")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Shape mismatch: A.shape[1]={A.shape[1]} != B.shape[0]={B.shape[0]}")
    if bias.shape[0] != B.shape[1]:
        raise ValueError(f"Bias shape mismatch: bias.shape[0]={bias.shape[0]} != B.shape[1]={B.shape[1]}")
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError(f"Invalid dropout_p: {dropout_p}. Must be in [0, 1)")
    
    M, K = A.shape
    K2, N = B.shape
    
    # Ensure contiguous memory layout
    A = A.contiguous()
    B = B.contiguous()
    bias = bias.contiguous()
    
    # Initialize random states if dropout is enabled
    if dropout_p > 0.0:
        rand_states = fused_kernel.init_rand_states(M * N, seed)
    else:
        rand_states = torch.empty(0, dtype=torch.int64, device='cuda')
    
    try:
        C = fused_kernel.fused_mm_bias_gelu_dropout(A, B, bias, rand_states, dropout_p)
        return C
    except Exception as e:
        logger.error(f"Kernel execution failed: {e}")
        raise

# -----------------------------------------------------------------------------
# Enhanced Benchmark Harness
# -----------------------------------------------------------------------------
def benchmark(M=2048, K=2048, N=2048, iters=20, validate=True):
    logger.info(f"Benchmarking fused kernel: {M}x{K} @ {K}x{N}")
    
    # Create test data
    torch.cuda.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(5):
        try:
            _ = run_fused_kernel(A, B, bias, dropout_p=0.0)  # No dropout for warmup
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            return

    torch.cuda.synchronize()

    # Timing fused kernel
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = run_fused_kernel(A, B, bias, dropout_p=0.0)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / iters

    # Baseline implementation
    def baseline():
        x = torch.matmul(A, B)
        x = x + bias
        x = torch.nn.functional.gelu(x)
        return x

    # Timing baseline
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = baseline()
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / iters

    # Validation
    if validate:
        try:
            custom_result = run_fused_kernel(A, B, bias, dropout_p=0.0)
            baseline_result = baseline()
            max_diff = torch.max(torch.abs(custom_result - baseline_result)).item()
            logger.info(f"✅ Validation: max_diff = {max_diff:.6f}")
            if max_diff > 1e-3:
                logger.warning(f"⚠️  Large numerical difference detected: {max_diff}")
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")

    # Performance metrics
    flops = 2 * M * N * K  # Multiply-accumulate operations
    fused_gflops = flops / (fused_time * 1e9)
    baseline_gflops = flops / (baseline_time * 1e9)
    
    logger.info(f"⏱  Fused Kernel: {fused_time*1e3:.3f} ms/iter ({fused_gflops:.1f} GFLOPS)")
    logger.info(f"⏱  Baseline: {baseline_time*1e3:.3f} ms/iter ({baseline_gflops:.1f} GFLOPS)")
    
    if fused_time > 0:
        speedup = baseline_time / fused_time
        logger.info(f"🚀 Speedup: {speedup:.2f}x")
        return speedup
    else:
        logger.warning("⚠️  Invalid timing results")
        return 0.0

# -----------------------------------------------------------------------------
# Occupancy & Roofline Hook
# -----------------------------------------------------------------------------
def roofline_analysis(M=2048, N=2048, K=2048, time_ms=1.0):
    flops = 2 * M * N * K  # MACs = 2 FLOPs
    achieved_flops = flops / (time_ms * 1e-3) / 1e12  # TFLOPs
    bandwidth = (M*K + K*N + M*N) * 4 / (time_ms * 1e-3) / 1e9  # GB/s
    intensity = flops / ((M*K + K*N + M*N) * 4)
    logger.info(f"Roofline → FLOPs: {achieved_flops:.2f} TFLOPs, BW: {bandwidth:.2f} GB/s, AI={intensity:.2f}")
    return achieved_flops, bandwidth, intensity

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Run benchmarks with different sizes
        sizes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 2048, 2048)]
        for M, K, N in sizes:
            logger.info(f"\n{'='*60}")
            speedup = benchmark(M, K, N, iters=50, validate=True)
            roofline_analysis(M, K, N, time_ms=1.0)
        
        # Example usage of the kernel with dropout
        logger.info(f"\n{'='*60}")
        logger.info("Testing kernel with dropout...")
        A = torch.randn(512, 1024, device="cuda", dtype=torch.float32)
        B = torch.randn(1024, 2048, device="cuda", dtype=torch.float32)
        bias = torch.randn(2048, device="cuda", dtype=torch.float32)
        
        C = run_fused_kernel(A, B, bias, dropout_p=0.1, seed=42)
        logger.info(f"✅ Output shape: {C.shape}, dtype: {C.dtype}")
        logger.info(f"✅ Zero ratio (dropout effect): {(C == 0).float().mean().item():.3f}")
        logger.info("✅ Custom kernel execution completed successfully.")
        
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)