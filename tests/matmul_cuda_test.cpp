#include "paraforge/matmul_cuda.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <tuple>

using paraforge::CudaTimings;
using paraforge::Matrix;

namespace {

// Skip only when CUDA reports no device; any other error is a broken install and must fail.
#define REQUIRE_GPU()                                                                     \
    do {                                                                                  \
        int count = 0;                                                                    \
        const cudaError_t err = cudaGetDeviceCount(&count);                               \
        if (err == cudaErrorNoDevice || (err == cudaSuccess && count == 0))               \
            GTEST_SKIP() << "no CUDA device";                                             \
        ASSERT_EQ(err, cudaSuccess) << "cudaGetDeviceCount: " << cudaGetErrorString(err); \
    } while (0)

Matrix random_matrix(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Matrix m(rows, cols);
    for (std::size_t i = 0; i < rows * cols; ++i) m.data()[i] = dist(rng);
    return m;
}

using MatmulFn = Matrix (*)(const Matrix&, const Matrix&, CudaTimings*);
const MatmulFn kImpls[] = {paraforge::matmul_cuda_naive, paraforge::matmul_cuda_tiled};

}  // namespace

TEST(MatmulCuda, KnownProduct) {
    REQUIRE_GPU();
    Matrix a(2, 3), b(3, 2);
    const float av[] = {1, 2, 3, 4, 5, 6}, bv[] = {7, 8, 9, 10, 11, 12};
    std::copy(av, av + 6, a.data());
    std::copy(bv, bv + 6, b.data());
    for (const MatmulFn fn : kImpls) {
        const Matrix c = fn(a, b, nullptr);
        ASSERT_EQ(c.rows(), 2u);
        ASSERT_EQ(c.cols(), 2u);
        EXPECT_EQ(c(0, 0), 58.0f);  // small integers: exact in float
        EXPECT_EQ(c(0, 1), 64.0f);
        EXPECT_EQ(c(1, 0), 139.0f);
        EXPECT_EQ(c(1, 1), 154.0f);
    }
}

class MatmulCudaShapes
    : public ::testing::TestWithParam<
          std::tuple<MatmulFn, std::tuple<std::size_t, std::size_t, std::size_t>>> {};

TEST_P(MatmulCudaShapes, MatchesReferenceWithinTolerance) {
    REQUIRE_GPU();
    const auto [fn, shape] = GetParam();
    const auto [m, k, n] = shape;
    const Matrix a = random_matrix(m, k, 1), b = random_matrix(k, n, 2);
    const Matrix want = paraforge::matmul_reference(a, b);
    CudaTimings t;
    const Matrix got = fn(a, b, &t);
    ASSERT_EQ(got.rows(), m);
    ASSERT_EQ(got.cols(), n);
    const double tol = 1e-5 * static_cast<double>(k);  // same bound as paraforge_cuda_bench
    for (std::size_t i = 0; i < m * n; ++i) {
        ASSERT_NEAR(got.data()[i], want.data()[i], tol) << "element " << i;
    }
    EXPECT_GE(t.kernel_ms, 0.0f);
}

// Sizes off multiples of the 16x16 block exercise the bounds check on both grid edges and, for
// the tiled kernel, partial last tiles along k (zero padding) as well as m and n.
INSTANTIATE_TEST_SUITE_P(
    Shapes, MatmulCudaShapes,
    ::testing::Combine(
        ::testing::ValuesIn(kImpls),
        ::testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(2, 3, 2),
                          std::make_tuple(15, 15, 15), std::make_tuple(16, 16, 16),
                          std::make_tuple(17, 17, 17), std::make_tuple(32, 48, 16),
                          std::make_tuple(64, 64, 64), std::make_tuple(17, 33, 15),
                          std::make_tuple(37, 19, 53), std::make_tuple(31, 1, 33),
                          std::make_tuple(1, 300, 70), std::make_tuple(70, 300, 1),
                          std::make_tuple(130, 257, 65), std::make_tuple(256, 512, 128))));

TEST(MatmulCuda, EmptyInnerDimensionGivesZeros) {
    // No launch happens for empty shapes, so this needs no GPU.
    for (const MatmulFn fn : kImpls) {
        const Matrix c = fn(Matrix(3, 0), Matrix(0, 4), nullptr);
        ASSERT_EQ(c.rows(), 3u);
        ASSERT_EQ(c.cols(), 4u);
        for (std::size_t i = 0; i < 12; ++i) EXPECT_EQ(c.data()[i], 0.0f);
    }
}

TEST(MatmulCuda, ShapeMismatchThrows) {
    for (const MatmulFn fn : kImpls) {
        EXPECT_THROW(fn(Matrix(2, 3), Matrix(4, 2), nullptr), std::invalid_argument);
    }
}
