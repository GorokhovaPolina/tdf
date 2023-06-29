#include "matrixs/matrixs.hpp"
#include <stdexcept>
#include <tuple>

MatrixS::MatrixS(const SizeType& size) : size_(size) {
    if (std::get<0>(size) < 0 || std::get<1>(size) < 0) {
        throw std::invalid_argument("invalid size");
    }
    data_ = new int[std::get<0>(size_) * std::get<1>(size_)];
    for (int i = 0; i < std::get<0>(size_) * std::get<1>(size_); i++) {
        data_[i] = 0;
    }
}
MatrixS::MatrixS(const std::ptrdiff_t m, const std::ptrdiff_t n) {
    if (m < 0 || n < 0) { throw std::invalid_argument("invalid size"); }
    std::get<0>(size_) = m;
    std::get<1>(size_) = n;
    data_ = new int[std::get<0>(size_) * std::get<1>(size_)];
    for (int i = 0; i < std::get<0>(size_) * std::get<1>(size_); i++) {
        data_[i] = 0;
    }
}

MatrixS::MatrixS(const MatrixS& other) : size_(other.size_) {
    data_ = new int[std::get<0>(size_) * std::get<1>(size_)];
    for (int i = 0; i < std::get<0>(size_) * std::get<1>(size_); i++) {
        data_[i] = other.at(i / std::get<1>(size_), i % std::get<0>(size_));
    }
}

MatrixS& MatrixS::operator=(const MatrixS& other) {
    if (data_ != other.data_) {
        size_ = other.size_;
        delete[] data_;
        data_ = new int[std::get<0>(size_) * std::get<1>(size_)];
        for (int i = 0; i < std::get<0>(size_) * std::get<1>(size_); i++) {
            data_[i] = other.at(i / std::get<1>(size_), i % std::get<1>(size_));
        }
    }
}

[[nodiscard]] const int& MatrixS::at(const SizeType& indexs) const {
    if (std::get<0>(indexs) >= std::get<0>(size_)
        || std::get<0>(indexs) < 0
        || std::get<1>(indexs) >= std::get<1>(size_)
        || std::get<1>(indexs) < 0) {
        throw std::out_of_range("invalid index");
    }
    return data_[std::get<1>(size_) * std::get<0>(indexs) + std::get<1>(indexs)];
}
[[nodiscard]] int& MatrixS::at(const SizeType& indexs) {
    if (std::get<0>(indexs) >= std::get<0>(size_)
        || std::get<0>(indexs) < 0
        || std::get<1>(indexs) >= std::get<1>(size_)
        || std::get<1>(indexs) < 0) {
        throw std::out_of_range("invalid index");
    }
    return data_[std::get<1>(size_) * std::get<0>(indexs) + std::get<1>(indexs)];
}

[[nodiscard]] const int& MatrixS::at(const std::ptrdiff_t  index_row,
    const std::ptrdiff_t index_col) const {
    if (index_row >= std::get<0>(size_)
        || index_row < 0
        || index_col >= std::get<1>(size_)
        || index_col < 0) {
        throw std::out_of_range("invalid index");
    }
    return data_[std::get<1>(size_) * index_row + index_col];
}
[[nodiscard]] int& MatrixS::at(const std::ptrdiff_t index_row,
    const std::ptrdiff_t index_col) {
    if (index_row >= std::get<0>(size_)
        || index_row < 0
        || index_col >= std::get<1>(size_)
        || index_col < 0) {
        throw std::out_of_range("invalid index");
    }
    return data_[std::get<1>(size_) * index_row + index_col];
}
[[nodiscard]] const MatrixS::SizeType& MatrixS::ssize() const noexcept {
    return size_;
}
[[nodiscard]] std::ptrdiff_t MatrixS::nRows() const noexcept {
    return std::get<0>(size_);
}
[[nodiscard]] std::ptrdiff_t MatrixS::nCols() const noexcept {
    return std::get<1>(size_);
}


void MatrixS::resize(const SizeType& size) {
    if (std::get<0>(size) <= 0
        || std::get<1>(size) <= 0) {
        throw std::invalid_argument("invalid size");
    }
    int* old = data_;
    data_ = new int[std::get<0>(size) * std::get<1>(size)];
    for (int i = 0; i < std::get<0>(size_); i++) {
        for (int j = 0; j < std::get<1>(size_); j++) {
            data_[std::get<1>(size_) * i + j] = old[std::get<1>(size_) * i + j];
        }
    }
    std::get<0>(size_) = std::get<0>(size);
    std::get<1>(size_) = std::get<1>(size);
    delete[] old;
}

void MatrixS::resize(std::ptrdiff_t nRows, ptrdiff_t nCols) {
    if (nRows <= 0 || nCols <= 0) {
        throw std::invalid_argument("invalid size");
    }
    int* old = data_;
    data_ = new int[nRows * nCols];
    for (int i = 0; i < nRows * nCols; i++) { data_[i] = 0; }
    for (int i = 0; i < std::get<0>(size_); i++) {
        for (int j = 0; j < std::get<1>(size_); j++) {
            data_[std::get<1>(size_) * i + j] = old[std::get<1>(size_) * i + j];
        }
    }
    std::get<0>(size_) = nRows;
    std::get<1>(size_) = nCols;
    delete[] old;
}