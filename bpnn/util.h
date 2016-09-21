#ifndef NN_UTIL_H
#define NN_UTIL_H

#include <vector>

class Matrix2D {
public:
    Matrix2D(int row, int col);

    Matrix2D(int row, int col, double vals);



private:
    std::vector<std::vector<double>> matrix_;
};

#endif
