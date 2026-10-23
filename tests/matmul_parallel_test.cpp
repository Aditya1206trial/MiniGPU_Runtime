#include "paraforge/matrix.hpp"
#include "paraforge/thread_pool.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <stdexcept>

using paraforge::Matrix;
using paraforge::matmul_parallel;
using paraforge::matmul_reference;
using paraforge::ThreadPool;

namespace {

Matrix random_matrix(std::size_t rows, std::size_t cols, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Matrix m(rows, cols);
    for (std::size_t i = 0; i < rows * cols; ++i) m.data()[i] = dist(rng);
    return m;
}

// Same arithmetic per element as the reference, so demand exact bit equality.
void expect_same(const Matrix& got, const Matrix& want) {
    ASSERT_EQ(got.rows(), want.rows());
    ASSERT_EQ(got.cols(), want.cols());
    EXPECT_EQ(std::memcmp(got.data(), want.data(), want.rows() * want.cols() * sizeof(float)), 0);
}

}  // namespace

TEST(MatmulParallel, MatchesReferenceAcrossShapesGrainsAndPools) {
    struct Shape { std::size_t m, k, n; };
    const Shape shapes[] = {{1, 1, 1}, {1, 9, 1}, {1, 5, 7}, {7, 1, 3},
                            {7, 13, 5}, {33, 17, 29}, {64, 64, 64}};
    const std::size_t grains[] = {1, 2, 3, 7, 64, 1000};
    const std::size_t workers[] = {1, 3, 4};
    for (std::size_t w : workers) {
        ThreadPool pool(w);  // reused across calls on purpose
        for (const Shape& s : shapes) {
            Matrix a = random_matrix(s.m, s.k, 1);
            Matrix b = random_matrix(s.k, s.n, 2);
            Matrix want = matmul_reference(a, b);
            for (std::size_t g : grains) {
                SCOPED_TRACE(testing::Message() << "workers=" << w << " shape=" << s.m << "x"
                                                << s.k << "x" << s.n << " grain=" << g);
                expect_same(matmul_parallel(a, b, pool, g), want);
            }
        }
    }
}

TEST(MatmulParallel, ZeroInnerDimensionGivesZeros) {
    ThreadPool pool(2);
    Matrix c = matmul_parallel(Matrix(5, 0), Matrix(0, 3), pool, 2);
    ASSERT_EQ(c.rows(), 5u);
    ASSERT_EQ(c.cols(), 3u);
    for (std::size_t i = 0; i < 15; ++i) EXPECT_EQ(c.data()[i], 0.0f);
}

TEST(MatmulParallel, ZeroRowsSubmitsNothingAndReturnsEmpty) {
    ThreadPool pool(2);
    Matrix c = matmul_parallel(Matrix(0, 4), Matrix(4, 3), pool, 1);
    EXPECT_EQ(c.rows(), 0u);
    EXPECT_EQ(c.cols(), 3u);
}

TEST(MatmulParallel, ThrowsOnShapeMismatch) {
    ThreadPool pool(2);
    EXPECT_THROW(matmul_parallel(Matrix(2, 3), Matrix(2, 3), pool, 1), std::invalid_argument);
    EXPECT_THROW(matmul_parallel(Matrix(4, 1), Matrix(2, 4), pool, 1), std::invalid_argument);
}

TEST(MatmulParallel, ThrowsOnZeroGrain) {
    ThreadPool pool(2);
    EXPECT_THROW(matmul_parallel(Matrix(2, 2), Matrix(2, 2), pool, 0), std::invalid_argument);
}
