#pragma once 

#include "Layer.h"

class Network {
    public:
        Network(std::vector<size_t> layer_sizes, std::vector<std::vector<double>>& input_data, std::vector<std::vector<double>>& output_data, double learning_rate) : input_data(input_data), output_data(output_data) {
            if (layer_sizes[layer_sizes.size()-1]!=output_data[0].size()) {
                throw std::invalid_argument("last size in layer_sizes must match output data size"); 
            }
            if (input_data.size()!=output_data.size()) {
                throw std::invalid_argument("input and output training data don't have equal size"); 
            }
            
            Layer h1 = Layer(input_data[0].size(),layer_sizes[0], "ReLU", learning_rate);
            layers.push_back(h1);
            for (unsigned i = 1; i<layer_sizes.size(); i++) {
                if (i==layer_sizes.size()-1) {
                    layers.push_back(Layer(layer_sizes[i-1],layer_sizes[i], "Linear", learning_rate));
                }
                else {
                    layers.push_back(Layer(layer_sizes[i-1],layer_sizes[i], "ReLU", learning_rate));
                }
                
                
            }
        }

        void ForwardBackwardPass(size_t n) {
            if (n>output_data.size()) {
                throw std::invalid_argument("Can't train more entries then there are in the training dataset");
            }
            
            for (size_t index = 0; index<n; index++) {
                ForwardPass(input_data[index]);
                BackwardPass(input_data[index],output_data[index]);
            }
        }

        void ForwardPass(std::vector<double>& input_data) {
            layers[0].ForwardPass(input_data);
            for (unsigned i = 1; i<layers.size(); i++) {
                layers[i].ForwardPass(layers[i-1].GetOutputs());
            }
        }

        std::vector<double> TestSingle(std::vector<double> test_input) {
            ForwardPass(test_input);
            return FinalOutput();
        }

        std::vector<std::vector<double>> Test(std::vector<std::vector<double>>& test_input) {
            std::vector<std::vector<double>> predicted_output;

            for (const std::vector<double>& vec : test_input) {
                predicted_output.push_back(TestSingle(vec));
            }

            return predicted_output;
        }


        void BackwardPass(std::vector<double>& input_data, std::vector<double>& output_data) {
            std::vector<double> MSELoss_deriv;
            for (size_t i = 0; i<layers[layers.size()-1].GetOutputs().size(); i++) {
                MSELoss_deriv.push_back(2*(layers[layers.size()-1].GetOutputs()[i]-output_data[i]));
            }
            layers[layers.size()-1].BackwardPass(MSELoss_deriv, layers[layers.size()-2].GetOutputs());

            std::vector<double> old_a_deriv = MSELoss_deriv;
            for (int k = layers.size()-2; k>=0; k--) {
                std::vector<double> a_deriv;
                for (size_t i = 0; i<layers[k].GetNeurons().size(); i++) {
                    a_deriv[i] = dotProduct(old_a_deriv, layers[k].GetNeurons()[i].GetWeights());
                    old_a_deriv = a_deriv;
                }
                if (k==0) {
                    layers[k].BackwardPass(a_deriv, input_data);
                }
                else {
                    layers[k].BackwardPass(a_deriv, layers[k-1].GetOutputs());
                }
                
            }
        }

        std::vector<double> FinalOutput() {
            return layers[layers.size()-1].GetOutputs();
        }

    private:
        std::vector<std::vector<double>>& input_data;
        std::vector<std::vector<double>>& output_data;
        std::vector<Layer> layers;

};