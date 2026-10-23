#pragma once

// Shared-memory tiled C = A * B. Row-major: A is m x k, B is k x n, C is m x n.
// Must be launched with kTile x kTile thread blocks on a ceil(n/kTile) x ceil(m/kTile) grid.
// Each block computes one kTile x kTile tile of C, walking k in kTile-wide steps: every thread
// loads one A and one B element into shared memory, so each global value is read once per block
// instead of once per thread. Kept free of host code and includes so
// bench/nvrtc_kernel_check.py can JIT this exact source; extern "C" keeps the name unmangled.
constexpr int kTile = 16;

extern "C" __global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int m,
                                               int n, int k) {
    __shared__ float a_tile[kTile][kTile];
    __shared__ float b_tile[kTile][kTile];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;
    // No early return for out-of-range threads: every thread must reach both __syncthreads and
    // load its share of the tile (zero padding), or the block deadlocks / reads stale data.
    float sum = 0.0f;
    for (int t = 0; t < k; t += kTile) {  // host rejects k > INT_MAX - kTile, so t cannot overflow
        const int a_col = t + tx, b_row = t + ty;
        a_tile[ty][tx] = (row < m && a_col < k) ? a[static_cast<size_t>(row) * k + a_col] : 0.0f;
        b_tile[ty][tx] = (b_row < k && col < n) ? b[static_cast<size_t>(b_row) * n + col] : 0.0f;
        __syncthreads();  // tile fully loaded
        for (int i = 0; i < kTile; ++i) {
            sum += a_tile[ty][i] * b_tile[i][tx];  // zero padding adds exact zeros past k
        }
        __syncthreads();  // everyone done reading before the next iteration overwrites the tile
    }
    if (row < m && col < n) c[static_cast<size_t>(row) * n + col] = sum;
}
