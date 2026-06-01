#include <iostream>
#include <fstream>
#include <sstream>
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

    //-------Reading in Data-------
    std::ifstream ifs("../data/sample_data.csv");
    if (!ifs.is_open()) {
        std::cerr<<"file unopened!"<<std::endl;
    }
    std::string line;
    std::vector<std::vector<double>> rows;

    while (std::getline(ifs,line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string entry;
        while (std::getline(ss,entry,',')) {
            row.push_back(std::stod(entry));
        }
        rows.push_back(row);
    }

    //-------Input-Output Split-------
    std::vector<std::vector<double>> input_data;
    std::vector<std::vector<double>> output_data;
    for (const std::vector<double>& vec : rows) {
        input_data.push_back(std::vector(vec.begin(),vec.begin()+vec.size()-1));
        output_data.push_back(std::vector(vec.begin()+vec.size()-1,vec.end()));
    }

    //-------Train-Test Split-------
    std::vector<std::vector<double>> input_test(input_data.begin(),input_data.begin()+input_data.size()/5);
    std::vector<std::vector<double>> input_train(input_data.begin()+input_data.size()/5,input_data.end());
    std::vector<std::vector<double>> output_test(output_data.begin(),output_data.begin()+output_data.size()/5);
    std::vector<std::vector<double>> output_train(output_data.begin()+output_data.size()/5,output_data.end());

    //-------Creating Neural Network-------
    std::vector<size_t> layer_sizes = {16,1}; //0th layer is the first hidden layer, last layer is the output layer. So size of output vectors must equal the last number in layer_sizes
    double learning_rate = .0001;
    Network nn = Network(layer_sizes, input_train, output_train, learning_rate);


    //-------Training Loop-------
    for (unsigned int epoch = 0; epoch<500; epoch++) {
        nn.ForwardBackwardPass(output_train.size()); //Peforms a forward pass followed by a backward pass on every single input-output pairing in the train split.
    }

    
    //-------Testing-------
    std::vector<std::vector<double>> predicted_output = nn.Test(input_test);
    double mae = 0.0;
    for (size_t i = 0; i<predicted_output.size(); i++) {
        std::cout<<"predicted: "; PrintVector(predicted_output[i]);
        std::cout<<"expected: "; PrintVector(output_test[i]);
        mae += std::abs(predicted_output[i][0]-output_test[i][0]); //i know these are 1x1 vectors, so i'm indexing by 0.
    }
    mae/=predicted_output.size();
    std::cout<<"Final MAE is "<<mae<<std::endl;
    std::cout<<"Error relative to range (NMAE) is "<<(mae/40)*100<<"%"<<std::endl;
   

    return 0;
}



