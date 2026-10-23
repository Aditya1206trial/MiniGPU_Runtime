#include "paraforge/matrix.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

using paraforge::Matrix;
using paraforge::matmul_reference;

TEST(Matrix, ConstructsWithShapeAndFill) {
    Matrix m(2, 3, 1.5f);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c) EXPECT_FLOAT_EQ(m(r, c), 1.5f);
}

TEST(Matrix, RejectsOverflowingShape) {
    // (SIZE_MAX / 2 + 1) * 2 wraps to 0; unchecked it would allocate nothing and index out of bounds.
    EXPECT_THROW(Matrix(SIZE_MAX / 2 + 1, 2), std::length_error);
    EXPECT_THROW(Matrix(2, SIZE_MAX / 2 + 1), std::length_error);
    Matrix empty(SIZE_MAX, 0);  // zero elements is a valid shape, not overflow
    EXPECT_EQ(empty.rows(), SIZE_MAX);
}

TEST(Matrix, StorageIsRowMajor) {
    Matrix m(2, 3);
    m(1, 2) = 7.0f;
    m(0, 1) = 3.0f;
    EXPECT_FLOAT_EQ(m.data()[1 * 3 + 2], 7.0f);
    EXPECT_FLOAT_EQ(m.data()[0 * 3 + 1], 3.0f);
}

TEST(Matmul, KnownRectangularProduct) {
    // [1 2 3]   [ 7  8]   [ 58  64]
    // [4 5 6] x [ 9 10] = [139 154]
    //           [11 12]
    Matrix a(2, 3), b(3, 2);
    float av[] = {1, 2, 3, 4, 5, 6};
    float bv[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; ++i) {
        a.data()[i] = av[i];
        b.data()[i] = bv[i];
    }
    Matrix c = matmul_reference(a, b);
    ASSERT_EQ(c.rows(), 2u);
    ASSERT_EQ(c.cols(), 2u);
    EXPECT_FLOAT_EQ(c(0, 0), 58);
    EXPECT_FLOAT_EQ(c(0, 1), 64);
    EXPECT_FLOAT_EQ(c(1, 0), 139);
    EXPECT_FLOAT_EQ(c(1, 1), 154);
}

TEST(Matmul, IdentityLeavesMatrixUnchanged) {
    const std::size_t n = 4;
    Matrix a(n, n), id(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        id(i, i) = 1.0f;
        for (std::size_t j = 0; j < n; ++j) a(i, j) = static_cast<float>(i * n + j) - 5.0f;
    }
    Matrix left = matmul_reference(id, a);
    Matrix right = matmul_reference(a, id);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            EXPECT_FLOAT_EQ(left(i, j), a(i, j));
            EXPECT_FLOAT_EQ(right(i, j), a(i, j));
        }
}

TEST(Matmul, InnerDimensionZeroGivesZeros) {
    Matrix c = matmul_reference(Matrix(2, 0), Matrix(0, 3));
    ASSERT_EQ(c.rows(), 2u);
    ASSERT_EQ(c.cols(), 3u);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) EXPECT_FLOAT_EQ(c(i, j), 0.0f);
}

TEST(Matmul, ThrowsOnShapeMismatch) {
    EXPECT_THROW(matmul_reference(Matrix(2, 3), Matrix(2, 3)), std::invalid_argument);
}
