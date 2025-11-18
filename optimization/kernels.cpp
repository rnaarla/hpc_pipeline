// Try to include torch/pybind11; fall back to stubs for IntelliSense-only parsing
#if __has_include(<torch/extension.h>)
  #include <torch/extension.h>
  #define HPC_HAS_TORCH 1
  namespace py = pybind11;
#else
  #define HPC_HAS_TORCH 0
  // Minimal forward decls to satisfy IntelliSense when torch headers are not found
  namespace torch { class Tensor; }
  namespace pybind11 { class module_; template <typename...> class class_; }
  namespace py = pybind11;
#endif

#include <string>
#include <tuple>
#include "custom_kernels.h"

// Forward declaration to match .cu signature (already in header, kept for clarity)
std::tuple<torch::Tensor, KernelMetrics> fused_matmul_bias_gelu_dropout(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float dropout_prob
);

#if HPC_HAS_TORCH
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_matmul_bias_gelu_dropout",
          &fused_matmul_bias_gelu_dropout,
          "Fused GEMM + Bias + GELU + Dropout (CUDA)");

    py::class_<KernelMetrics>(m, "KernelMetrics")
        .def(py::init<>())
        .def_readonly("time_ms", &KernelMetrics::time_ms)
        .def_readonly("achieved_tflops", &KernelMetrics::achieved_tflops)
        .def_readonly("sm_occupancy", &KernelMetrics::sm_occupancy)
        .def_readonly("memory_bandwidth", &KernelMetrics::memory_bandwidth);
}
#else
// No-op for IntelliSense-only environment (build uses setup.py with proper include paths)
static void hpc_kernels_intellisense_stub() {}
#endif
