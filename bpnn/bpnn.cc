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

void BackPropNNetwork::SingleTraing(const std::vector<double> &taring_dta,
                                    const std::vector<double> &training_result) {

}

