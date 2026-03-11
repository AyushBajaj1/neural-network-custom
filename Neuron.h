#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>


class Neuron {
    public:
        Neuron() = default;

        Neuron(int number_of_variables) {
            weights.resize(number_of_variables);

            //random device to initialize weights
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
            for (int i = 0; i<number_of_variables; i++) {
                weights[i]=dist(gen);
            }

            bias = 0.0;
            output = 0.0;
        }

        void ForwardPass(const std::vector<double>& inputs) {
            if (inputs.size()!=weights.size()) {
                throw std::invalid_argument("inputs size does not match weights size");
            }

            for (std::size_t i = 0; i<weights.size(); i++) {
                output+=weights[i]*inputs[i];
            }
            output+=bias;
            output = ReLU(output);
        }

        double GetOutput() {
            return output;
        }

    private:
        double bias = 0.0;
        std::vector<double> weights;
        double output = 0.0;

        double ReLU(double z) {
            return std::max(z,0.0);
        }

};