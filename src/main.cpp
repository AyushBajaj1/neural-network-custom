#include <iostream>
#include "Network.h"

// template <typename T>
void PrintVector(const std::vector<double>& vec) {
    std::cout<<"[";
    for (unsigned i = 0; i<vec.size(); i++) {
        std::cout<<vec[i]<<", ";
    }
    std::cout<<"]"<<std::endl;
}

int main () {

    std::vector<double> input_data = {4.5 , 5.6 , 7.8 , 10 , 11 , 15 , 15.1 , 90 , 1.2};
    std::vector<double> output_data = {16.1,45,19,1,2,18};
    std::vector<size_t> layer_sizes = {15,6,10,15,6};
    Network nn = Network(layer_sizes,input_data, output_data);
    
    nn.ForwardPass();
    std::vector<double> generated_outputs = nn.FinalOutput();
    PrintVector(generated_outputs);

    for (int i = 0; i<100; i++) {
        nn.BackwardPass();
        nn.ForwardPass();
        std::vector<double> generated_outputs2 = nn.FinalOutput();
        PrintVector(generated_outputs2);
    }
    

    return 0;
}

