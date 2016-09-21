//
// Created by ruoshui on 9/22/16.
//

#ifndef MACHINE_LEARNING_BPNN_H
#define MACHINE_LEARNING_BPNN_H


#include <vector>
#include "util.h"

class BackPropNNetwork{
public:
    BackPropNNetwork(std::vector<int> net);
    void Training(const Matrix2D &training_data, const Matrix2D &traing_y);
    void BatchTraing(const Matrix2D& batch_training_data, const Matrix2D& batch_taring_y);
    void SingleTraing(const std::vector<double> &taring_dta, const std::vector<double> &training_y);

private:
    int batch_num_;
    int max_iters_;
    double eta_;
    std::vector<Matrix2D> global_weights_;
    std::vector<double> global_biases_;
    std::vector<int> sizes;

};


#endif //MACHINE_LEARNING_BPNN_H
