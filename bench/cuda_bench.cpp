// Times the naive and tiled CUDA matmul C (m x n) = A (m x k) * B (k x n) on the same inputs and
// checks every result against the CPU.
// Usage: paraforge_cuda_bench [m] [k] [n] [repeats]
// Prints CSV to stdout, one row per kernel; every *_ms column is the median of `repeats` timed
// calls, taken per column.

#include "paraforge/matmul_cuda.hpp"
#include "paraforge/matrix.hpp"
#include "paraforge/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace paraforge;

namespace {

// Positive decimal integer or 0 on any junk ("-1", "8x", "" all rejected).
std::size_t parse_positive(const char* s) {
    if (*s == '\0' || std::strspn(s, "0123456789") != std::strlen(s)) return 0;
    try {
        const unsigned long long v = std::stoull(s);
        return v <= SIZE_MAX ? static_cast<std::size_t>(v) : 0;
    } catch (...) {
        return 0;
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t mid = v.size() / 2;
    return v.size() % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

// Largest |got - want|; the CPU reference accumulates in double, the GPU in float.
// Any NaN/Inf yields +inf so the tolerance check fails (std::max would silently drop a NaN).
double max_abs_diff(const Matrix& want, const Matrix& got) {
    double worst = 0.0;
    for (std::size_t i = 0; i < want.rows() * want.cols(); ++i) {
        const double d = std::fabs(static_cast<double>(want.data()[i]) - got.data()[i]);
        if (!std::isfinite(d)) return HUGE_VAL;
        worst = std::max(worst, d);
    }
    return worst;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t args[] = {1024, 1024, 1024, 5};  // m, k, n, repeats
    if (argc > 5) {
        std::fprintf(stderr, "usage: %s [m] [k] [n] [repeats]\n", argv[0]);
        return 2;
    }
    for (int i = 1; i < argc; ++i) {
        args[i - 1] = parse_positive(argv[i]);
        if (args[i - 1] == 0) {
            std::fprintf(stderr, "usage: %s [m] [k] [n] [repeats]\n"
                                 "all values must be positive integers, got '%s'\n",
                         argv[0], argv[i]);
            return 2;
        }
    }
    const std::size_t m = args[0], k = args[1], n = args[2], repeats = args[3];

    Matrix a, b;
    try {
        a = Matrix(m, k);
        b = Matrix(k, n);
    } catch (const std::exception&) {
        std::fprintf(stderr, "error: shape %zu x %zu x %zu is too large to allocate\n", m, k, n);
        return 2;
    }
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < m * k; ++i) a.data()[i] = dist(rng);
    for (std::size_t i = 0; i < k * n; ++i) b.data()[i] = dist(rng);

    // matmul_parallel is bitwise equal to matmul_reference and much faster to wait for.
    ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
    const Matrix want = matmul_parallel(a, b, pool, 16);
    // Inputs are in [-1, 1]; observed float-vs-double error is far below this (see docs/design.md).
    const double tol = 1e-5 * static_cast<double>(k);

    const struct {
        const char* name;
        Matrix (*fn)(const Matrix&, const Matrix&, CudaTimings*);
    } impls[] = {{"cuda_naive", matmul_cuda_naive}, {"cuda_tiled", matmul_cuda_tiled}};

    const double flop = 2.0 * static_cast<double>(m) * n * k;
    std::printf("impl,m,k,n,repeats,h2d_ms,kernel_ms,d2h_ms,end_to_end_ms,kernel_gflops,"
                "end_to_end_gflops,max_abs_err\n");
    for (const auto& impl : impls) {
        std::vector<double> h2d, kernel, d2h, e2e;
        double worst = 0.0;
        try {
            worst = max_abs_diff(want, impl.fn(a, b, nullptr));  // untimed: context creation, JIT
            for (std::size_t r = 0; r < repeats; ++r) {
                CudaTimings t;
                const auto t0 = std::chrono::steady_clock::now();
                const Matrix got = impl.fn(a, b, &t);
                const auto t1 = std::chrono::steady_clock::now();
                e2e.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                h2d.push_back(t.h2d_ms);
                kernel.push_back(t.kernel_ms);
                d2h.push_back(t.d2h_ms);
                worst = std::max(worst, max_abs_diff(want, got));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s: %s\n", impl.name, e.what());
            return 3;
        }
        if (!(worst <= tol)) {  // non-finite output arrives as inf
            std::fprintf(stderr, "error: %s: max |gpu - cpu| = %g exceeds tolerance %g\n",
                         impl.name, worst, tol);
            return 1;
        }
        const double kernel_ms = median(kernel), e2e_ms = median(e2e);
        std::printf("%s,%zu,%zu,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.3g\n", impl.name, m, k, n,
                    repeats, median(h2d), kernel_ms, median(d2h), e2e_ms,
                    flop / (kernel_ms * 1e6), flop / (e2e_ms * 1e6), worst);
    }
    return 0;
}
