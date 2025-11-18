#pragma once

// Minimal PyTorch header for Tensor and pybind11 interop (guarded for IntelliSense)
#if __has_include(<torch/extension.h>)
  #include <torch/extension.h>
  #define HPC_HAS_TORCH 1
#else
  namespace torch { class Tensor; }
  #define HPC_HAS_TORCH 0
#endif

// Standard includes
#include <stdexcept>
#include <tuple>

// Forward declare CUDA types to avoid pulling CUDA headers into C++ TU
struct dim3;
typedef struct CUstream_st* cudaStream_t;

// Error type
class KernelError : public std::runtime_error {
public:
    explicit KernelError(const char* msg) : std::runtime_error(msg) {}
};

// Metrics and configuration
struct KernelMetrics {
    float time_ms;
    float achieved_tflops;
    int sm_occupancy;
    float memory_bandwidth;  // GB/s

    KernelMetrics() : time_ms(0.f), achieved_tflops(0.f), sm_occupancy(0), memory_bandwidth(0.f) {}
};

struct KernelConfig {
    int block_size;
    int grid_size;
    size_t shared_memory_bytes;
    bool use_tensor_cores;

    KernelConfig() : block_size(256), grid_size(1), shared_memory_bytes(0), use_tensor_cores(true) {}
};

// Helper functions (implemented in .cu)
float calculate_tflops(int m, int n, int k, float time_ms);
int get_sm_occupancy();
bool validate_hardware_compatibility();

// Main fused op (implemented in .cu)
std::tuple<
    torch::Tensor,
    KernelMetrics
> fused_matmul_bias_gelu_dropout(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_prob
);

// Optional launch helpers (implemented in .cu)
KernelMetrics launch_kernel(const dim3& grid, const dim3& block);
KernelConfig get_optimal_kernel_config(int m, int n, int k);
