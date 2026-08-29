#pragma once 

#include "Matrix.h"

class Network {
    public:
        Network(std::vector<size_t> layer_sizes, std::vector<std::string> hidden_activations, std::string loss);
        void fit(const Matrix& input_data, const Matrix& output_data, double learning_rate, size_t batch_size, size_t epochs);
        
        Matrix predict(const Matrix& test_input);
        Matrix final_output();
        
    private:
        void forward_pass(const Matrix& input_data_batch);
        void backward_pass(const Matrix& input_data_batch, const Matrix& output_data_batch, double learning_rate);

        //layer info
        std::vector<Matrix> layer_weights;
        std::vector<Matrix> layer_biases;
        std::vector<Matrix> layer_outputs;

        //layer updates
        std::vector<Matrix> weight_updates;
        std::vector<Matrix> bias_updates;

        //architecture
        std::vector<std::string> activations;
        std::string loss;

        
};