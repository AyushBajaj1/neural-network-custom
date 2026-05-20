#include <vector>
#include <algorithm>
#include <random>
#include <stdexcept>


int dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
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

        Neuron(int number_of_weights) {
            weights.resize(number_of_weights);

            // std::random_device rd;
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
            output = ReLU(output);
        }

        void BackwardPass(double deriv, const std::vector<double>& inputs) {
            for (size_t i = 0; i<weights.size(); i++) {
                weights[i]-=deriv*inputs[i]*lr;
            }
            bias-=deriv*lr;
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
        double lr = .0001;

        double output = 0.0;
};

