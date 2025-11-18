#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <algorithm>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Optimized fused bias addition and ReLU kernel with coalesced memory access
__global__ void fused_bias_relu_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int spatial_size,
    const long long total_elements
) {
    // Use long long to avoid integer overflow
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // More efficient channel calculation
        int channel_idx = (idx / spatial_size) % channels;
        
        // Fused operation with optimized memory access
        float input_val = input[idx];
        float bias_val = bias[channel_idx];
        output[idx] = fmaxf(0.0f, input_val + bias_val);
    }
}

// Vectorized kernel for aligned memory access (float4)
__global__ void fused_bias_relu_kernel_vectorized(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    const int batch_size,
    const int channels,
    const int spatial_size,
    const long long total_vec_elements
) {
    long long vec_idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    
    if (vec_idx < total_vec_elements) {
        long long element_idx = vec_idx * 4;
        int channel_idx = (element_idx / spatial_size) % channels;
        
        float4 input_val = input[vec_idx];
        float bias_val = bias[channel_idx];
        
        float4 result;
        result.x = fmaxf(0.0f, input_val.x + bias_val);
        result.y = fmaxf(0.0f, input_val.y + bias_val);
        result.z = fmaxf(0.0f, input_val.z + bias_val);
        result.w = fmaxf(0.0f, input_val.w + bias_val);
        
        output[vec_idx] = result;
    }
}

// Input validation function
bool validate_inputs(int batch_size, int channels, int height, int width) {
    if (batch_size <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        fprintf(stderr, "Error: Invalid tensor dimensions\n");
        return false;
    }
    
    // Check for potential overflow
    long long total_elements = static_cast<long long>(batch_size) * channels * height * width;
    if (total_elements > INT_MAX) {
        fprintf(stderr, "Warning: Large tensor size may cause integer overflow\n");
    }
    
    return true;
}

extern "C" {
    void fused_bias_relu(
        const float* d_input,
        const float* d_bias,
        float* d_output,
        int batch_size,
        int channels,
        int height,
        int width,
        cudaStream_t stream = nullptr
    ) {
        // Input validation
        if (!validate_inputs(batch_size, channels, height, width)) {
            return;
        }
        
        if (!d_input || !d_bias || !d_output) {
            fprintf(stderr, "Error: Null pointer passed to kernel\n");
            return;
        }
        
        long long total_elements = static_cast<long long>(batch_size) * channels * height * width;
        int spatial_size = height * width;
        
        // Adaptive block size based on problem size
        int block_size = (total_elements < 1024) ? 128 : 256;
        
        // Check if we can use vectorized kernel (requires alignment)
        bool use_vectorized = (total_elements % 4 == 0) && 
                             (reinterpret_cast<uintptr_t>(d_input) % 16 == 0) &&
                             (reinterpret_cast<uintptr_t>(d_output) % 16 == 0);
        
        cudaStream_t launch_stream = stream ? stream : 0;

        if (use_vectorized && total_elements >= 1024) {
            long long vec_elements = total_elements / 4;
            long long grid_size = (vec_elements + block_size - 1) / block_size;
            
            fused_bias_relu_kernel_vectorized<<<grid_size, block_size, 0, launch_stream>>>(
                reinterpret_cast<const float4*>(d_input),
                d_bias,
                reinterpret_cast<float4*>(d_output),
                batch_size, channels, spatial_size, vec_elements
            );
        } else {
            long long grid_size = (total_elements + block_size - 1) / block_size;
            
            fused_bias_relu_kernel_optimized<<<grid_size, block_size, 0, launch_stream>>>(
                d_input, d_bias, d_output,
                batch_size, channels, spatial_size, total_elements
            );
        }
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaPeekAtLastError());
    }
}

// Host function with memory management
void fused_bias_relu_host(
    const float* h_input,
    const float* h_bias,
    float* h_output,
    int batch_size,
    int channels,
    int height,
    int width
) {
    // Input validation
    if (!validate_inputs(batch_size, channels, height, width)) {
        return;
    }
    
    if (!h_input || !h_bias || !h_output) {
        fprintf(stderr, "Error: Null pointer passed to host function\n");
        return;
    }
    
    long long input_size = static_cast<long long>(batch_size) * channels * height * width;
    int bias_size = channels;
    
    size_t input_bytes = input_size * sizeof(float);
    size_t bias_bytes = bias_size * sizeof(float);
    
    // Allocate device memory with proper alignment
    float *d_input, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, input_bytes));
    
    // Use async memory copy for better performance
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, input_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bias, h_bias, bias_bytes, cudaMemcpyHostToDevice, stream));
    
    // Launch kernel
    fused_bias_relu(d_input, d_bias, d_output, batch_size, channels, height, width, stream);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, input_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
}

// Performance benchmarking function
void benchmark_fused_bias_relu(int batch_size, int channels, int height, int width, int iterations = 100) {
    if (!validate_inputs(batch_size, channels, height, width)) return;
    
    long long input_size = static_cast<long long>(batch_size) * channels * height * width;
    size_t input_bytes = input_size * sizeof(float);
    size_t bias_bytes = channels * sizeof(float);
    
    float *d_input, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, input_bytes));
    
    // Warm-up
    fused_bias_relu(d_input, d_bias, d_output, batch_size, channels, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        fused_bias_relu(d_input, d_bias, d_output, batch_size, channels, height, width);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Average kernel time: %.3f ms\n", milliseconds / iterations);
    printf("Throughput: %.2f GB/s\n", 
           (2.0f * input_bytes * iterations) / (milliseconds * 1e6)); // 2x for read+write
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
}
