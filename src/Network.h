#pragma once 

#include "Layer.h"

class Network {
    public:
        Network(std::vector<size_t> layer_sizes, std::vector<double>& input_data, std::vector<double>& output_data) : input_data(input_data), output_data(output_data) {
            if (layer_sizes[layer_sizes.size()-1]!=output_data.size()) {
                throw std::invalid_argument("last size in layer_sizes must match output data size"); 
            }
            
            Layer h1 = Layer(input_data.size(),layer_sizes[0], "ReLU");
            layers.push_back(h1);
            for (unsigned i = 1; i<layer_sizes.size(); i++) {
                if (i==layer_sizes.size()-1) {
                    layers.push_back(Layer(layer_sizes[i-1],layer_sizes[i], "Linear"));
                }
                else {
                    layers.push_back(Layer(layer_sizes[i-1],layer_sizes[i], "ReLU"));
                }
                
                
            }
        }

        void ForwardPass() {
            layers[0].ForwardPass(input_data);

            for (unsigned i = 1; i<layers.size(); i++) {
                layers[i].ForwardPass(layers[i-1].GetOutputs());
            }

        }

        void BackwardPass() {
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
        std::vector<double>& input_data;
        std::vector<double>& output_data;
        std::vector<Layer> layers;

};