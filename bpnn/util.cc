//
// Created by ruoshui on 9/22/16.
//

#include "util.h"

Matrix::Matrix(int row, int col) : row_(row), col_(col){
    Matrix(row, col, 0.0);
}
Matrix::Matrix(int row, int col, double vals):  row_(row), col_(col) {
    for ( int i = 0; i < row; ++i){
        std::vector<double> v(col, vals);
        matrix_.push_back(v);
    }
}

Matrix::Matrix(const std::vector<double> &input, const std::vector<double> &delta) {
    col_ = input.size();
    row_ = delta.size();
    for (int i = 0; i < row_; ++i) {
        std::vector<double> v(col_);
        for(int j = 0; j < col_; ++j) {
            v[j] = input[j] * delta[i];
        }
        matrix_.push_back(v);
    }
}
void Matrix::Dot(const std::vector<double> &a, std::vector<double> &x) const {
    for (int i = 0; i < row_; ++i) {
        double temp = 0.0;
        for ( int j = 0; j < col_; ++j){
            temp += matrix_[i][j] * a[j];
        }
        x[i] = temp;
    }
}

Matrix& Matrix::operator+(Matrix &m) {
    for (int i = 0; i < row_; ++i) {
        for ( int j = 0; j < col_; ++j) {
            matrix_[i][j] -= m.matrix_[i][j];
        }
    }
}

void Matrix::ComputeDelta(const std::vector<double> &delta, std::vector<double> &partial_delta) {
    for (int i = 0; i < col_; ++i) {
        double result = 0.0;
        for (int j = 0;j < row_; ++j) {
            result += matrix_[j][i] * delta[j];
        }
        partial_delta[i] = result;
    }
}

void Matrix::Sub(const double &r, const Matrix &delta_matrix) {
    for (int i = 0; i < row_; ++i) {
        for (int j = 0; j < col_; ++j) {
            matrix_[i][j] -= r * delta_matrix.matrix_[i][j];
        }
    }
}

void Matrix::Add(const Matrix &matrix) {
    for ( int i = 0; i < row_; ++i){
        for ( int j = 0; j < col_; ++j) {
            matrix_[i][j] += matrix.matrix_[i][j];
        }
    }
}