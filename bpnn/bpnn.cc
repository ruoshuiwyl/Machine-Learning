//
// Created by ruoshui on 9/22/16.
//

#include <algorithm>
#include <iostream>
#include "bpnn.h"


BackPropNNetwork::BackPropNNetwork(std::vector<int> &sizes, double eta, int max_iters, int batch_num) :
        sizes_(sizes), eta_(eta), max_iters_(max_iters), batch_num_(batch_num) {

    for (int i = 1; i < sizes.size(); ++i) {
        Matrix temp_matrix{sizes_[i], sizes_[i-1], 1.0};
        global_weights_.push_back(temp_matrix);

        std::vector<double> biase;
        biase.resize(sizes_[i], 1.0);
        global_biases_.push_back(biase);
        delta_.push_back(biase);
        input_z_.push_back(biase);
    }
    for (int i = 0; i < sizes_.size(); ++i){
        std::vector<double> t;
        t.resize(sizes_[i]);
        output_a_.push_back(t);
    }
}

void BackPropNNetwork::SingleTraining(const std::vector<double> &tarining_data,
                                      const std::vector<double> &training_result) {
    output_a_[0] = tarining_data;
    for ( int iter = 0; iter < max_iters_; ++iter) {
        FeedForward();
        for ( int j = 0; j < training_result.size(); ++j){
            double t = fabs(training_result[j] - output_a_[output_a_.size() - 1][j]);
            if ( t > 0.2){
                std::cout << "error" << std::endl;
            }
        }
        std::vector<double> &activation = output_a_[sizes_.size() - 1];
        ComputeCostDerivative(activation, training_result);
        for (int i = sizes_.size() - 3; i >= 0; i--) {
            ComputeDelta(i);
        }
    }
}

void BackPropNNetwork::ComputeCostDerivative(const std::vector<double> &activation,
                                             const std::vector<double> &taining_result) {
    int j = sizes_.size() - 2;
    for ( int i = 0; i < activation.size(); ++i) {
        delta_[j][i] = (activation[i] - taining_result[i] ) * sigmoid_prime(output_a_[j+1][i]);
        delta_biases_[j][i] += delta_[j][i];
    }
    Matrix delta_weight{output_a_[j], delta_[j] };
    delta_weight_[j].Add(delta_weight);
}


void BackPropNNetwork::ComputeDelta(const int &layer_num) {
    std::vector<double> &delta = delta_[layer_num];
    std::vector<double> &delta_biases = delta_biases_[layer_num];
    Matrix &weight = global_weights_[layer_num+1];
    std::vector<double> &back_delta = delta_[layer_num+1];
    weight.Dot(back_delta, delta);
    for (int i = 0; i < delta.size(); ++i){
        delta[i] = delta[i] * sigmoid_prime(output_a_[layer_num][i]);
        delta_biases[i] += delta[i];
    }
    Matrix delta_weight(output_a_[layer_num], delta);
    delta_weight_[layer_num].Add(delta_weight);
}


void BackPropNNetwork::ComputeSigmoid(const std::vector<double> &input, std::vector<double> &output) {
    for (int i = 0; i < input.size(); ++i ) {
        output[i] = sigmoid(input[i]);
    }
}

void BackPropNNetwork::FeedForward() {
    for (int i = 0; i < sizes_.size() - 1; ++i) {
        Matrix &weight = global_weights_[i];
        std::vector<double> &biase = global_biases_[i];
        std::vector<double> &z = input_z_[i];
        std::vector<double> &output = output_a_[i+1];
        std::vector<double> &a = output_a_[i];
        ComputeOutput(weight, biase, z, a, output);
    }
}

void BackPropNNetwork::ComputeOutput(const Matrix &weight, const std::vector<double> biases, std::vector<double> &z,
                                     std::vector<double> &a, std::vector<double> &output) {
    weight.Dot(a, z);
    for ( int i = 0 ; i < biases.size(); ++i){
        z[i] += biases[i];
    }
    ComputeSigmoid(z, output);
}
void BackPropNNetwork::SetGobalWeightAndBiases() {
    for ( int i = 1; i < sizes_.size(); ++i) {
        Matrix temp_matrix{sizes_[i], sizes_[i - 1], 0.0};
        delta_weight_.push_back(temp_matrix);
        std::vector<double> biase;
        biase.resize(sizes_[i], 0.0);
        delta_biases_.push_back(biase);
    }
}

void BackPropNNetwork::UpdetaGobalWeightAndBiases() {
    const double eta = eta_/batch_num_;
    for ( int i = 0; i < sizes_.size() - 1; ++i) {
        Matrix &weight = global_weights_[i];
        Matrix &delta_weight = delta_weight_[i];
        std::vector<double> &biases = global_biases_[i];
        std::vector<double> &delta_biases = delta_biases_[i];
        weight.Sub(eta, delta_weight);
        for ( int j = 0; j < biases.size(); ++j){
            biases[j] -= eta * delta_biases[i];
        }
    }
}


void BackPropNNetwork::BatchTraining(const std::vector<std::vector<double>> &batch_training_data,
                                     const std::vector<std::vector<double>> &batch_tarining_result) {
    SetGobalWeightAndBiases();
    for ( int i = 0; i < batch_tarining_result.size(); ++i) {
        SingleTraining(batch_training_data[i], batch_tarining_result[i]);
    }
    UpdetaGobalWeightAndBiases();
}


void BackPropNNetwork::Training(const std::vector<std::vector<double>> &training_data,
                                const std::vector<std::vector<double>> &training_result) {
    std::vector<int> shuffle;
    for (int i = 0; i < training_data.size(); ++i) {
        shuffle.push_back(i);
    }
    std::random_shuffle(shuffle.begin(), shuffle.end());
    std::vector<std::vector<double>> batch_data, batch_result;
    for (int i = 0; i < training_data.size(); i += batch_num_) {
        for (int j = 0; j < batch_num_; ++j) {
            batch_data.push_back(training_data[i + j]);
            batch_result.push_back(training_result[i + j]);
        }
        BatchTraining(batch_data, batch_result);
    }
}


double BackPropNNetwork::Test(const std::vector<std::vector<double>> &training_data,
                              const std::vector<std::vector<double>> &training_result) {
    double total = training_data.size();
    double error = 0.0;
    for ( int i = 0; i < training_data.size(); ++i) {
        output_a_[0] = training_data[i];
        FeedForward();
        bool error_flag = false;
        for ( int j = 0; j < training_result[i].size(); ++j){
            double t = fabs(training_result[i][j] - output_a_[output_a_.size() - 1][j]);
            if (t >  0.1 ) {
                error_flag = true;
            }
        }
        error += error_flag ? 1.0 : 0.0;
    }
    return 1.0 - error / total;
}

int main() {
    std::vector<std::vector<double>> tarining_data{ {1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
    std::vector<std::vector<double>> tarining_result{{1.0}, {0.0}, {0.0}, {0.0}};
    std::vector<int> sizes{2, 2, 1};
    BackPropNNetwork bpp(sizes, 0.1, 40, 2);
    bpp.Training(tarining_data, tarining_result);
    std::cout << bpp.Test(tarining_data, tarining_result) << std::endl;
    return 0;

}