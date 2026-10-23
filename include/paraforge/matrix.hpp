#pragma once

#include <cstddef>
#include <vector>

namespace paraforge {

class ThreadPool;

// Dense row-major matrix of floats: element (r, c) lives at data[r * cols + c].
class Matrix {
public:
    Matrix() = default;
    // Throws std::length_error if rows * cols exceeds std::vector<float>::max_size().
    Matrix(std::size_t rows, std::size_t cols, float fill = 0.0f);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    float& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    float operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> data_;
};

// Single-threaded reference C = A * B. Throws std::invalid_argument if a.cols() != b.rows().
Matrix matmul_reference(const Matrix& a, const Matrix& b);

// Parallel C = A * B: output rows are split into blocks of `grain` rows (last block may be
// shorter), one pool task per block. Each element uses the same arithmetic and k-order as
// matmul_reference, so results are bitwise identical to it.
// Blocks on pool.wait_all(), so the pool must not be used by anyone else during the call.
// Throws std::invalid_argument if a.cols() != b.rows() or grain == 0.
Matrix matmul_parallel(const Matrix& a, const Matrix& b, ThreadPool& pool, std::size_t grain);

}  // namespace paraforge
