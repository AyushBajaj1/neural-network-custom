#pragma once

#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cmath>


double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size()!=b.size()) {
        throw std::invalid_argument("vectors must be same size to be dot product'd");
    }


    double result = 0.0;
    for (size_t i = 0; i<a.size(); i++) {
        result += a[i]*b[i];
    }
    return result;
}

double ReLU(double z) {
    return std::max(z,0.0);
}

class Neuron {
    public:
        Neuron() = default;

        // Neuron(int number_of_weights) {
        //     weights.resize(number_of_weights);

        //     // std::random_device rd;
        //     static std::mt19937 gen(42);
        //     std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        //     for (int i = 0; i<number_of_weights; i++) {
        //         weights[i]=dist(gen);
        //     }

        //     bias = 0.0;
        //     output = 0.0;
        // }

        Neuron(int number_of_weights, std::string activation, double learning_rate) {
            this->activation = activation;
            lr=learning_rate;

            weights.resize(number_of_weights);

            std::random_device rd;
            static std::mt19937 gen(42);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
            for (int i = 0; i<number_of_weights; i++) {
                weights[i]=dist(gen);
            }

            bias = 0.0;
            output = 0.0;
         }

        void ForwardPass(const std::vector<double>& inputs) {
            
            if (inputs.size()!=weights.size()) {
                throw std::invalid_argument("inputs size does not match weights size");
            }
            
            output = dotProduct(inputs,weights);
            output+=bias;

            if (activation == "ReLU") {
                output = ReLU(output);
            }
            else if (activation == "Linear" ) {
                output*=1;
            }
            else if (activation == "Softmax") {
                output = std::exp(output);
                if (output==output) {
                    std::cout<<"o:"<<output<<std::endl;
                }
                
            }
            else {
                std::cout<<"activation: "<<activation<<std::endl;
                throw std::runtime_error("activation function not specified properly");
            }
        }

        void BackwardPass(double deriv, const std::vector<double>& inputs) {
            for (size_t i = 0; i<weights.size(); i++) {
                weights[i]-= output!=0 ? deriv*inputs[i]*lr : 0;
            }
            bias-= output!=0 ? deriv*lr : 0;
        }

        std::vector<double> GetWeights() {
            return weights;
        }

        double GetOutput() {
            return output;
        }

    private:
        double bias = 0.0;
        std::vector<double> weights;
        double lr = .001;
        std::string activation = "";
        double output = 0.0;
};

