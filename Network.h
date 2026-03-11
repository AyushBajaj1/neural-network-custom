#pragma once 

#include "Layer.h"

class Network {
    public:
        Network(std::vector<int> layer_sizes, std::vector<double>& input_data) : input_data(input_data) {
            // if (layer_sizes[0]!=static_cast<int>(input_data.size())) {
            //     throw std::invalid_argument("Size of layer 0 must match size of input data");
            // }

            Layer input_layer = Layer(input_data.size(),layer_sizes[0]);
            layers.push_back(input_layer);
            for (unsigned i = 1; i<layer_sizes.size(); i++) {
                layers.push_back(Layer(layer_sizes[i-1],layer_sizes[i]));
            }
        }

        void ForwardPass() {
            layers[0].ForwardPass(input_data);

            for (unsigned i = 1; i<layers.size(); i++) {
                layers[i].ForwardPass(layers[i-1].GetOutputs());
            }

        }

        void BackPropagation() {
            
        }

        std::vector<double> FinalOutput() {
            return layers[layers.size()-1].GetOutputs();
        }

    private:
        std::vector<double>& input_data;
        std::vector<Layer> layers;

};