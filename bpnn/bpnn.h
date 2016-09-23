//
// Created by ruoshui on 9/22/16.
//

#ifndef MACHINE_LEARNING_BPNN_H
#define MACHINE_LEARNING_BPNN_H


#include <vector>
#include "util.h"

class BackPropNNetwork{
public:
    BackPropNNetwork(std::vector<int> &sizes, double eta, int max_iters, int batch_num);
    void Training(const Matrix &training_data, const Matrix &training_result);
    void BatchTraing(const Matrix& batch_training_data, const Matrix& batch_tarining_result);
    void SingleTraing(const std::vector<double> &taring_dta, const std::vector<double> &training_result);

private:

    void ComputeDelta(const std::vector<double> &partial_delta);
    int batch_num_;
    int max_iters_;
    double eta_;
    std::vector<Matrix> global_weights_;
    std::vector<std::vector<double>> global_biases_;
    std::vector<Matrix> delta_global_wreghts_;
    std::vector<std::vector<double>> delta_global_biases_;
    std::vector<std::vector<double>> input_, output_;
    std::vector<int> sizes_;

};


#endif //MACHINE_LEARNING_BPNN_H
