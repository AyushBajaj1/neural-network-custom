#include "Network.h"

int main () {

    
    Matrix data;
    data.read_csv("./data/sample_data.csv");
    
    Matrix input_data = data.splice_cols(0,2);
    Matrix output_data = data.splice_cols(2,3);

    Matrix input_train = input_data.splice_rows(input_data.get_rows()/5,input_data.get_rows()); 
    Matrix output_train = output_data.splice_rows(output_data.get_rows()/5,output_data.get_rows()); 
    Matrix input_test = input_data.splice_rows(0,input_data.get_rows()/5);
    Matrix output_test = output_data.splice_rows(0,output_data.get_rows()/5);
    
    

    std::vector<size_t> layer_sizes = {2,16,1}; 
    std::vector<std::string> hidden_activations = {"relu"}; 
    std::string loss = "mse";
    Network nn = Network(layer_sizes, hidden_activations, loss);


    double learning_rate = .00001;
    size_t batch_size = 100;
    size_t epochs = 10000;
    nn.fit(input_train, output_train, learning_rate, batch_size, epochs);


    Matrix predicted_output = nn.Test(input_test,1);
    for (size_t i = 0; i<predicted_output.get_rows(); i++) {
        std::cout<<predicted_output.at(i,0)<<", "<<output_test.at(i,0)<<std::endl;
    }

    std::cout<<nn.nmae_accuracy(input_test, output_test)<<"%"<<std::endl;
   

    return 0;
}



