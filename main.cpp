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

    std::vector<double> data = {4.5 , 5.6 , 7.8 , 10 , 11 , 15 , 15.1 , 90 , 1.2};
    std::vector<int> layer_sizes = {15,6,10,15,6};
    Network nn = Network(layer_sizes,data);

    nn.ForwardPass();

    std::vector<double> outputs = nn.FinalOutput();
    PrintVector(outputs);

    return 0;
}

