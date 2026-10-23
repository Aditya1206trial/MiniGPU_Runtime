#pragma once

// Naive C = A * B, one thread per output element. Row-major: A is m x k, B is k x n, C is m x n.
// Kept free of host code and includes so bench/nvrtc_kernel_check.py can JIT this exact source.
// extern "C" gives the kernel an unmangled name for that NVRTC lookup.
extern "C" __global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int m,
                                               int n, int k) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;  // grid is rounded up to whole blocks
    const float* a_row = a + static_cast<size_t>(row) * k;  // size_t: row * k can exceed INT_MAX
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += a_row[i] * b[static_cast<size_t>(i) * n + col];
    }
    c[static_cast<size_t>(row) * n + col] = sum;
}
