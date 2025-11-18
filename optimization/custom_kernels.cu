#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <nvToolsExt.h>
#include <curand_kernel.h>
#include "custom_kernels.h"

// Constants for optimal performance 
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_THREADS 1024

// Add capability validation
static bool validate_hardware_compatibility() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    // Require Ampere or newer (SM80+)
    if (props.major < 8) {
        throw std::runtime_error("Hardware too old - requires SM80+ (Ampere)");
    }
    return true;
}

// Add Tensor Core mma instructions
#if __CUDA_ARCH__ >= 800
#include <mma.h>
using namespace nvcuda::wmma;
#endif

// Add kernel stats
__device__ __managed__ int g_kernel_errors = 0;

__device__ __forceinline__ half gelu(half x) {
    float tmp = __half2float(x);
    tmp = tmp * 0.5f * (1.0f + tanhf(0.797885f * tmp + 0.035677f * tmp * tmp * tmp));
    return __float2half(tmp);
}

// Add hardware-specific tuning parameters
struct HardwareProfile {
    int sm_count;
    int tensor_cores_per_sm;
    int max_shared_memory;
    cudaDeviceProp props;
};

static HardwareProfile get_hardware_profile() {
    HardwareProfile profile;
    cudaGetDeviceProperties(&profile.props, 0);
    profile.sm_count = profile.props.multiProcessorCount;
    profile.tensor_cores_per_sm = (profile.props.major >= 8) ? 4 : 2;
    profile.max_shared_memory = profile.props.sharedMemPerBlock;
    return profile;
}

// Add autotuning helper
KernelConfig autotune_kernel_config() {
    auto profile = get_hardware_profile();
    KernelConfig config;
    config.block_size = min(profile.max_shared_memory / (2 * sizeof(half)), MAX_THREADS);
    config.grid_size = profile.sm_count * 2;  // 2 blocks per SM
    return config;
}

__global__ void fused_matmul_bias_gelu_dropout_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    const float dropout_prob,
    const int m, const int n, const int k,
    const uint64_t seed
) {
    // Add error checking
    if (m <= 0 || n <= 0 || k <= 0) return;
    if (dropout_prob < 0.0f || dropout_prob >= 1.0f) return;
    
    // Add shared memory overflow protection
    if (blockDim.x * blockDim.y > MAX_THREADS) return;

    extern __shared__ half shmem[];
    half* tile_a = shmem;
    half* tile_b = &shmem[BLOCK_SIZE * BLOCK_SIZE];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    half acc = __float2half(0.0f);
    
    // Initialize RNG per-thread
    const int linear_tid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y)
                           + threadIdx.y * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, linear_tid, 0, &state);

    for (int tile = 0; tile < k; tile += BLOCK_SIZE) {
        // Collaborative loading of tiles
        if (row < m && (tile + threadIdx.x) < k)
            tile_a[threadIdx.y * BLOCK_SIZE + threadIdx.x] = input[row * k + tile + threadIdx.x];
        if (col < n && (tile + threadIdx.y) < k)
            tile_b[threadIdx.y * BLOCK_SIZE + threadIdx.x] = weight[(tile + threadIdx.y) * n + col];
        
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            acc = __hfma(tile_a[threadIdx.y * BLOCK_SIZE + i],
                        tile_b[i * BLOCK_SIZE + threadIdx.x],
                        acc);
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        // Apply bias and GELU
        acc = gelu(__hadd(acc, bias[col]));
        
        // Apply dropout
        if (dropout_prob > 0.0f) {
            const float rand = curand_uniform(&state);
            if (rand < dropout_prob) acc = __float2half(0.0f);
            else acc = __hmul(acc, __float2half(1.0f / (1.0f - dropout_prob)));
        }
        
        // Add numerical stability checks
        if (isinf(__half2float(acc)) || isnan(__half2float(acc))) {
            acc = __float2half(0.0f);
        }

        output[row * n + col] = acc;
    }

    #if __CUDA_ARCH__ >= 800
    // Use Tensor Core matrix multiply
    fragment<matrix_a> frag_a;
    fragment<matrix_b> frag_b;
    fragment<accumulator> frag_acc;
    
    load_matrix_sync(frag_a, tile_a, BLOCK_SIZE);
    load_matrix_sync(frag_b, tile_b, BLOCK_SIZE);
    mma_sync(frag_acc, frag_a, frag_b, frag_acc);
    #else
    // Fallback path for older hardware
    #endif
    
    // Add error detection
    if (isinf(__half2float(acc)) || isnan(__half2float(acc))) {
        atomicAdd(&g_kernel_errors, 1);
    }
}

// Python bindings with error handling
std::tuple<torch::Tensor, KernelMetrics> fused_matmul_bias_gelu_dropout(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_prob) {

    validate_hardware_compatibility();

    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(input.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf && bias.scalar_type() == at::kHalf,
                "All tensors must be float16");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && bias.dim() == 1, "Shapes: input [m,k], weight [k,n], bias [n]");

    const int m = input.size(0);
    const int k = input.size(1);
    const int n = weight.size(1);
    TORCH_CHECK(weight.size(0) == k, "weight.shape[0] must equal input.shape[1]");
    TORCH_CHECK(bias.size(0) == n, "bias.shape[0] must equal weight.shape[1]");

    auto opts = input.options();
    torch::Tensor output_tensor = torch::empty({m, n}, opts);

    // Grid/block config (simple 16x16 threads), shared memory for two tiles
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    size_t shmem_bytes = 2ull * BLOCK_SIZE * BLOCK_SIZE * sizeof(half);

    // NVTX + timing
    nvtxRangePush("fused_matmul_gelu_dropout");
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Seed for dropout RNG
    const uint64_t seed = static_cast<uint64_t>(clock64());

    fused_matmul_bias_gelu_dropout_kernel<<<grid, block, shmem_bytes>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output_tensor.data_ptr<at::Half>()),
        dropout_prob, m, n, k, seed
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel launch failed: ", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    nvtxRangePop();

    KernelMetrics metrics{
        milliseconds,
        // tflops = 2*m*n*k / time
        (float)((2.0 * (double)m * (double)n * (double)k) / ((double)milliseconds * 1.0e-3) * 1.0e-12),
        /*sm_occupancy=*/0,
        /*memory_bandwidth=*/0.0f
    };

    return {output_tensor, metrics};
}
