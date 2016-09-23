//
// Created by ruoshui on 9/22/16.
//

#include "bpnn.h"


BackPropNNetwork::BackPropNNetwork(std::vector<int> &sizes, double eta, int max_iters, int batch_num) :
        sizes_(sizes), eta_(eta), max_iters_(max_iters), batch_num_(batch_num) {
    for (int i = 1; i < sizes.size(); ++i) {
        global_weights_.push_back({sizes_[i], sizes_[i-1], 1.0});
        std::vector<double> biase;
        biase.resize(sizes_[i], 1.0);
        global_biases_.push_back(biase);
    }
}

void BackPropNNetwork::SingleTraining(const std::vector<double> &tarining_data,
                                      const std::vector<double> &training_result) {
    FeedForward(tarining_data);
    std::vector<double> &activation = output_a_[sizes_.size()-1];
    ComputeCostDerivative(activation, training_result);
    for ( int i = sizes_.size() - 1; i >= 0; i--){

    }

}

void BackPropNNetwork::ComputeCostDerivative(const std::vector<double> &activation,
                                             const std::vector<double> &taining_result) {
    int j = sizes_.size() - 1;
    for ( int i = 0; i < activation.size(); ++i) {
        delta_[j][i] = activation[i] - taining_result[i];

    }
}


void BackPropNNetwork::ComputeDelta(const std::vector<double> &partial_delta) {


}


void BackPropNNetwork::ComputeSigmoid(const std::vector<double> &input, std::vector<double> &output) {
    for (int i = 0; i < input.size(); ++i ) {
        output[i] = sigmoid(input[i]);
    }
}
void BackPropNNetwork::ComputeSigmoidPrime(const std::vector<double> &input, std::vector<double> &output) {
    for (int i = 0; i < input.size(); ++i) {
        output[i] = sigmoid_prime(input[i]);
    }
}


void BackPropNNetwork::FeedForward(const std::vector<double> &input) {
    std::vector<double> &a = const_cast<std::vector<double> &>(input);
    for (int i = 0; i < sizes_.size(); ++i) {
        Matrix &weight = global_weights_[i];
        std::vector<double> &biase = global_biases_[i];
        std::vector<double> &z = input_z_[i];
        std::vector<double> &output = output_a_[i];
        ComputeOutput(weight, biase, z, a, output);
        a = output;
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


