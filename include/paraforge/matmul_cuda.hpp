#pragma once

#include "paraforge/matrix.hpp"

namespace paraforge {

// Per-call GPU timings in milliseconds, measured with CUDA events on the default stream.
// Device allocation/free and CUDA context creation are not included in any of them.
struct CudaTimings {
    float h2d_ms = 0.0f;     // copy A and B host -> device
    float kernel_ms = 0.0f;  // kernel only
    float d2h_ms = 0.0f;     // copy C device -> host
};

// Naive CUDA C = A * B: one thread per output element, 16x16 thread blocks, float accumulator.
// Allocates device buffers, copies inputs, runs the kernel, copies C back and frees, every call.
// Results are not bitwise equal to matmul_reference (float vs double accumulation, FMA).
// Throws std::invalid_argument on shape mismatch or dimensions the launch cannot cover,
// std::runtime_error on any CUDA error (including no device). `timings` may be null.
Matrix matmul_cuda_naive(const Matrix& a, const Matrix& b, CudaTimings* timings = nullptr);

// Same contract as matmul_cuda_naive, but the kernel stages 16x16 tiles of A and B in shared
// memory (src/matmul_tiled_kernel.cuh). Same per-element summation order as the naive kernel.
Matrix matmul_cuda_tiled(const Matrix& a, const Matrix& b, CudaTimings* timings = nullptr);

}  // namespace paraforge
