#pragma once 

#include "Neuron.h"

class Layer {
    public:
        Layer(int input_size, int output_size) {
            // neurons.resize(input_size);
            outputs.resize(output_size);
            for (int i = 0; i<output_size; i++) {
                neurons.push_back(Neuron(input_size));
            }

        }

        void ForwardPass(const std::vector<double>& inputs) {
            for (std::size_t i = 0; i<neurons.size(); i++) {
                neurons[i].ForwardPass(inputs);
                outputs[i]=neurons[i].GetOutput();
            }
        }

        std::vector<double> GetOutputs() {
            return outputs;
        }

    private:
        std::vector<double> outputs;
        std::vector<Neuron> neurons;

};