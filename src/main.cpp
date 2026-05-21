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

std::vector<double> NormalizedVector(const std::vector<double>& vec) {
    double sum = 0;
    for (const double& num : vec) {
        sum+=num;
    }
    double mean = sum/vec.size();

    double var = 0;
    for (const double& num : vec) {
        var+=((num-mean)*(num-mean));
    }
    var/=(vec.size()-1);
    double sd = std::sqrt(var);

    std::vector<double> normalized;
    for (const double& num : vec) {
        normalized.push_back((num-mean)/sd);
    }

    return normalized;
}

int main () {

    std::vector<double> input_data = {4.5 , 5.6 , 7.8 , 10 , 11 , 15 , 15.1};
    std::vector<double> output_data = {16.1,45,19,1,2,18};

    // input_data = NormalizedVector(input_data);
    // output_data = NormalizedVector(output_data);
    //uncomment if you want to see the data turned into z-scores with respect to its own mean and std dev

    std::cout<<"input:";
    PrintVector(input_data);
    std::cout<<"output:";
    PrintVector(output_data);
    

    std::vector<size_t> layer_sizes = {15,6,10,15,6};
    Network nn = Network(layer_sizes,input_data, output_data);
    nn.ForwardPass();
    std::vector<double> generated_outputs = nn.FinalOutput();
    std::cout<<"pred: ";
    PrintVector(generated_outputs);
    for (int i = 0; i<75; i++) {
        nn.BackwardPass();
        nn.ForwardPass();
        generated_outputs = nn.FinalOutput();
        std::cout<<"pred: ";
        PrintVector(generated_outputs);
    }
    
    std::cout<<"input:";
    PrintVector(input_data);
    std::cout<<"output:";
    PrintVector(output_data);

    return 0;
}



