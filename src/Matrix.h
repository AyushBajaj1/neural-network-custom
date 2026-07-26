#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>

class Matrix {
    public:
        Matrix() = default;
        Matrix(size_t rows, size_t cols);

        size_t get_rows() const;
        size_t get_cols() const;

        double& at(size_t row, size_t col);
        const double& at(size_t row, size_t col) const;

        Matrix matmul(const Matrix& other) const;
        Matrix scalarmul(const double& scalar) const;
        void scalarmul_inplace(const double& scalar) const;
        Matrix transpose() const;
        Matrix splice_rows(size_t start, size_t end) const;
        Matrix splice_cols(size_t start, size_t end) const;
        Matrix combine_rows(const Matrix& other);

        void randfill_inplace(const double& range);
        Matrix randfill(const double& range) const;
        void fill_inplace(const double& value );
        Matrix fill(double num) const;

        Matrix heavside();
        void heavside_inplace();
        void leaky_heavside_inplace();

        Matrix hadamard_product(const Matrix& other) const;
        void hadamard_product_inplace(const Matrix& other);

        Matrix digit_to_one_hot_vector() const;
        Matrix hot_vector_to_digit() const;

        void relu_inplace();
        void leaky_relu_inplace();
        void softmax_inplace();

        Matrix& operator-=(const Matrix& other);
        Matrix operator-(const Matrix& other) const;
        Matrix column_wise_sum();
        void add_row_vector_inplace(const Matrix& other);

        void abs_inplace();
        Matrix abs();

        Matrix shuffle_rows();

        void read_csv(std::string filename);
        void printline() const;
        void printblock() const;

        double frobenius_norm_sq() const;
        double cross_entropy_loss(const Matrix& Y) const;

        double mean();
        double var();
        double max();
        double min();

        ~Matrix();
        Matrix(const Matrix& other);
        Matrix& operator=(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        Matrix& operator=(Matrix&& other) noexcept;

    private:
        double* list = nullptr;
        size_t rows = 0;
        size_t cols = 0;
        const static int seed = 42;

        void copy(const Matrix& other);
        void clear();

};

