//
// Created by ruoshui on 9/22/16.
//

#ifndef MACHINE_LEARNING_BPNN_H
#define MACHINE_LEARNING_BPNN_H


#include <vector>
#include "util.h"
#include <cmath>

class BackPropNNetwork{
public:
    BackPropNNetwork(std::vector<int> &sizes, double eta, int max_iters, int batch_num);
    void Training(const std::vector<std::vector<double>> &training_data, const std::vector<std::vector<double>>  &training_result);
    void BatchTraining(const std::vector<std::vector<double>>  &batch_training_data, const std::vector<std::vector<double>>  &batch_tarining_result);
    void SingleTraining(const std::vector<double> &tarining_data, const std::vector<double> &training_result);
    double Test(const std::vector<std::vector<double>> &training_data, const std::vector<std::vector<double>>  &training_result);

private:
    void ComputeSigmoid(const std::vector<double> &input, std::vector<double> &output);
    void UpdetaGobalWeightAndBiases();
    void SetGobalWeightAndBiases();
//    void ComputeDeltaWeight(const std::vector<double> &intput, const std::vector<double> &delta, Matrix &delta_weight);
    void ComputeCostDerivative(const std::vector<double> &activation, const std::vector<double> &taining_result);
    void ComputeDelta(const int &layer_num);
    void FeedForward();
    void ComputeOutput(const Matrix &weight, const std::vector<double> biases, std::vector<double> &z,
                       std::vector<double> &a, std::vector<double> &output);

    inline const double sigmoid(const double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    inline const double sigmoid_prime(const double x) {
        const double t = sigmoid(x);
        return t * (1.0 - t);
    }

    int batch_num_;
    int max_iters_;
    double eta_;
    std::vector<Matrix> global_weights_;
    std::vector<std::vector<double>> global_biases_;
//    std::vector<Matrix> delta_global_weights_;
//    std::vector<std::vector<double>> delta_global_biases_;
    std::vector<std::vector<double>> delta_, delta_biases_;
    std::vector<Matrix> delta_weight_;
    std::vector<std::vector<double>> input_z_, output_a_;
    std::vector<int> sizes_;

};


#endif //MACHINE_LEARNING_BPNN_H
