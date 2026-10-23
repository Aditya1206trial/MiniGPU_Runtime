#include "paraforge/matmul_cuda.hpp"

#include "matmul_naive_kernel.cuh"
#include "matmul_tiled_kernel.cuh"

#include <cuda_runtime.h>

#include <climits>
#include <stdexcept>
#include <string>

namespace paraforge {

namespace {

constexpr int kBlock = 16;  // 16 x 16 = 256 threads per block
static_assert(kBlock == kTile, "the tiled kernel needs kTile x kTile blocks");

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

// Frees on scope exit so a throwing CUDA call cannot leak device memory.
struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(&ptr, count * sizeof(float)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Event {
    cudaEvent_t ev = nullptr;
    Event() { check(cudaEventCreate(&ev), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(ev); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

float elapsed_ms(const Event& from, const Event& to) {
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, from.ev, to.ev), "cudaEventElapsedTime");
    return ms;
}

// Shared host side of both kernels: they take the same arguments and launch shape.
Matrix run_matmul(const char* name, bool tiled, const Matrix& a, const Matrix& b,
                  CudaTimings* timings) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument(std::string(name) + ": a.cols() must equal b.rows()");
    }
    const std::size_t m = a.rows(), k = a.cols(), n = b.cols();
    constexpr std::size_t kMaxDim = INT_MAX - kBlock;
    if (m > kMaxDim || k > kMaxDim || n > kMaxDim || (m + kBlock - 1) / kBlock > 65535) {
        // Kernels take int dimensions and add up to kBlock to them; grid.y caps at 65535 blocks.
        throw std::invalid_argument(std::string(name) + ": matrix too large for one launch");
    }
    if (timings) *timings = CudaTimings{};
    Matrix c(m, n);
    if (m == 0 || n == 0 || k == 0) return c;  // empty output, or all-zero sums; no launch needed

    DeviceBuffer d_a(m * k), d_b(k * n), d_c(m * n);
    Event start, copied_in, computed, copied_out;

    check(cudaEventRecord(start.ev), "cudaEventRecord");
    check(cudaMemcpy(d_a.ptr, a.data(), m * k * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy A");
    check(cudaMemcpy(d_b.ptr, b.data(), k * n * sizeof(float), cudaMemcpyHostToDevice),
          "cudaMemcpy B");
    check(cudaEventRecord(copied_in.ev), "cudaEventRecord");

    const dim3 block(kBlock, kBlock);
    const dim3 grid(static_cast<unsigned>((n + kBlock - 1) / kBlock),
                    static_cast<unsigned>((m + kBlock - 1) / kBlock));
    const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
    // Named launches, not a function pointer: the plain triple-chevron form nvcc documents.
    if (tiled) {
        matmul_tiled_kernel<<<grid, block>>>(d_a.ptr, d_b.ptr, d_c.ptr, mi, ni, ki);
    } else {
        matmul_naive_kernel<<<grid, block>>>(d_a.ptr, d_b.ptr, d_c.ptr, mi, ni, ki);
    }
    check(cudaGetLastError(), "kernel launch");  // bad launch config; faults surface below
    check(cudaEventRecord(computed.ev), "cudaEventRecord");

    check(cudaMemcpy(c.data(), d_c.ptr, m * n * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy C");  // blocks until the kernel is done; reports kernel faults too
    check(cudaEventRecord(copied_out.ev), "cudaEventRecord");
    check(cudaEventSynchronize(copied_out.ev), "cudaEventSynchronize");

    if (timings) {
        timings->h2d_ms = elapsed_ms(start, copied_in);
        timings->kernel_ms = elapsed_ms(copied_in, computed);
        timings->d2h_ms = elapsed_ms(computed, copied_out);
    }
    return c;
}

}  // namespace

Matrix matmul_cuda_naive(const Matrix& a, const Matrix& b, CudaTimings* timings) {
    return run_matmul("matmul_cuda_naive", false, a, b, timings);
}

Matrix matmul_cuda_tiled(const Matrix& a, const Matrix& b, CudaTimings* timings) {
    return run_matmul("matmul_cuda_tiled", true, a, b, timings);
}

}  // namespace paraforge
