// Times square N x N matmul: single-threaded reference vs matmul_parallel on a ThreadPool.
// Usage: paraforge_bench [size] [threads] [grain] [repeats]
// Prints CSV to stdout; time_ms is the median wall time of `repeats` timed runs.

#include "paraforge/matrix.hpp"
#include "paraforge/thread_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace paraforge;

namespace {

// Positive decimal integer or 0 on any junk ("-1", "8x", "" all rejected).
std::size_t parse_positive(const char* s) {
    if (*s == '\0' || std::strspn(s, "0123456789") != std::strlen(s)) return 0;
    try {
        const unsigned long long v = std::stoull(s);
        return v <= SIZE_MAX ? static_cast<std::size_t>(v) : 0;  // 32-bit size_t would truncate
    } catch (...) {
        return 0;  // out of range
    }
}

// Each run's Matrix is kept in `last` so the work stays observable and cannot be optimized away;
// the move and freeing of the previous result happen outside the timed region.
template <class F>
double median_ms(std::size_t repeats, Matrix& last, F&& run) {
    std::vector<double> ms;
    for (std::size_t r = 0; r < repeats; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        Matrix out = run();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        last = std::move(out);
    }
    std::sort(ms.begin(), ms.end());
    const std::size_t mid = ms.size() / 2;
    return ms.size() % 2 ? ms[mid] : (ms[mid - 1] + ms[mid]) / 2;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t hw = std::max(1u, std::thread::hardware_concurrency());
    std::size_t args[] = {512, hw, 16, 5};  // size, threads, grain, repeats
    if (argc > 5) {
        std::fprintf(stderr, "usage: %s [size] [threads] [grain] [repeats]\n", argv[0]);
        return 2;
    }
    for (int i = 1; i < argc; ++i) {
        args[i - 1] = parse_positive(argv[i]);
        if (args[i - 1] == 0) {
            std::fprintf(stderr, "usage: %s [size] [threads] [grain] [repeats]\n"
                                 "all values must be positive integers, got '%s'\n",
                         argv[0], argv[i]);
            return 2;
        }
    }
    const std::size_t n = args[0], threads = args[1], grain = args[2], repeats = args[3];

    Matrix a, b;
    try {
        a = Matrix(n, n);
        b = Matrix(n, n);
    } catch (const std::exception&) {  // length_error on n * n overflow, bad_alloc if too big for RAM
        std::fprintf(stderr, "error: size %zu is too large to allocate\n", n);
        return 2;
    }
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n * n; ++i) a.data()[i] = dist(rng);
    for (std::size_t i = 0; i < n * n; ++i) b.data()[i] = dist(rng);

    ThreadPool pool(threads);  // thread startup is not timed

    // Untimed warm-up doubling as a correctness check.
    const Matrix want = matmul_reference(a, b);
    const Matrix got = matmul_parallel(a, b, pool, grain);
    if (std::memcmp(want.data(), got.data(), n * n * sizeof(float)) != 0) {
        std::fprintf(stderr, "error: matmul_parallel result differs from matmul_reference\n");
        return 1;
    }

    Matrix last_single, last_parallel;
    const double single = median_ms(repeats, last_single, [&] { return matmul_reference(a, b); });
    const double parallel =
        median_ms(repeats, last_parallel, [&] { return matmul_parallel(a, b, pool, grain); });
    if (std::memcmp(want.data(), last_single.data(), n * n * sizeof(float)) != 0 ||
        std::memcmp(want.data(), last_parallel.data(), n * n * sizeof(float)) != 0) {
        std::fprintf(stderr, "error: timed result differs from untimed warm-up\n");
        return 1;
    }

    std::printf("impl,size,threads,grain,repeats,time_ms\n");
    std::printf("cpu_single,%zu,1,%zu,%zu,%.3f\n", n, n, repeats, single);
    std::printf("cpu_parallel,%zu,%zu,%zu,%zu,%.3f\n", n, threads, grain, repeats, parallel);
    return 0;
}
