#include "paraforge/matrix.hpp"

#include "paraforge/thread_pool.hpp"

#include <algorithm>
#include <stdexcept>

namespace paraforge {

namespace {

// Computes rows [row_begin, row_end) of c = a * b. Shared by every CPU kernel so they agree bitwise.
void multiply_rows(const Matrix& a, const Matrix& b, Matrix& c, std::size_t row_begin,
                   std::size_t row_end) {
    for (std::size_t i = row_begin; i < row_end; ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            double sum = 0.0;  // double accumulator keeps the reference tighter than float kernels it checks
            for (std::size_t k = 0; k < a.cols(); ++k) {
                sum += static_cast<double>(a(i, k)) * b(k, j);
            }
            c(i, j) = static_cast<float>(sum);
        }
    }
}

// Rejects shapes whose element count overflows before rows * cols can wrap to a too-small size.
std::size_t checked_size(std::size_t rows, std::size_t cols) {
    if (cols != 0 && rows > std::vector<float>().max_size() / cols) {
        throw std::length_error("Matrix: rows * cols exceeds max element count");
    }
    return rows * cols;
}

}  // namespace

Matrix::Matrix(std::size_t rows, std::size_t cols, float fill)
    : rows_(rows), cols_(cols), data_(checked_size(rows, cols), fill) {}

Matrix matmul_reference(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matmul_reference: a.cols() must equal b.rows()");
    }
    Matrix c(a.rows(), b.cols());
    multiply_rows(a, b, c, 0, a.rows());
    return c;
}

Matrix matmul_parallel(const Matrix& a, const Matrix& b, ThreadPool& pool, std::size_t grain) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matmul_parallel: a.cols() must equal b.rows()");
    }
    if (grain == 0) {
        throw std::invalid_argument("matmul_parallel: grain must be > 0");
    }
    Matrix c(a.rows(), b.cols());
    try {
        for (std::size_t begin = 0, end = 0; begin < a.rows(); begin = end) {
            end = begin + std::min(grain, a.rows() - begin);  // min() avoids begin + grain overflow
            pool.submit([&a, &b, &c, begin, end] { multiply_rows(a, b, c, begin, end); });
        }
    } catch (...) {
        // Queued tasks reference a, b and c; they must finish before c is destroyed.
        try {
            pool.wait_all();
        } catch (...) {
        }
        throw;
    }
    pool.wait_all();
    return c;
}

}  // namespace paraforge
