#pragma once 

#include "Neuron.h"

class Layer {
    public:
        
        Layer(int input_size, int output_size) {
            outputs.resize(output_size);
            for (int i = 0; i<output_size; i++) {
                neurons.push_back(Neuron(input_size,activation));
            }

        }

        Layer(int input_size, int output_size, std::string activation) { 
            this->activation = activation;
            outputs.resize(output_size);
            for (int i = 0; i<output_size; i++) {
                neurons.push_back(Neuron(input_size,activation));
            }
        }

        void ForwardPass(const std::vector<double>& inputs) {
            for (std::size_t i = 0; i<neurons.size(); i++) {
                neurons[i].ForwardPass(inputs);
                outputs[i]=neurons[i].GetOutput();
            }
            
        }

        void BackwardPass(const std::vector<double>& param_deriv, const std::vector<double>& inputs) {
            for (size_t i = 0; i<param_deriv.size(); i++) {
                neurons[i].BackwardPass(param_deriv[i], inputs);
            }
        }

        std::vector<double> GetOutputs() {
            return outputs;
        }

        std::vector<Neuron> GetNeurons() {
            return neurons;
        }

    private:
        std::vector<double> outputs;
        std::vector<Neuron> neurons;
        std::string activation;

};

