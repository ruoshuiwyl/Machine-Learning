#ifndef NN_UTIL_H
#define NN_UTIL_H

#include <vector>

class Matrix {
public:
    Matrix(int row, int col);

    Matrix(int row, int col, double vals);

    void Dot(const std::vector<double> &a, std::vector<double> &x) const;
    void ComputeDelta(const std::vector<double> &delta, const std::vector<double> &partial_delta);

    Matrix &operator+(Matrix &m);
    void  Sub(const  double &r, const Matrix &delta_matrix);
    Matrix (const std::vector<double> &input, const std::vector<double> &delta);
private:
    std::vector<std::vector<double>> matrix_;
    int row_;
    int col_;
};

#endif
