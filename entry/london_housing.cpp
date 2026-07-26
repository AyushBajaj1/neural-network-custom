#include "../src/Network.h"
#include "../src/Normalizer.h"

int main () {
    Matrix data;
    data.read_csv("./data/london_houses_transformed.csv");
    data = data.shuffle_rows();
    
    Matrix input_data = data.splice_cols(0,87);
    Matrix output_data = data.splice_cols(87,88);

    Normalizer input_normalizer(input_data);
    Normalizer output_normalizer(output_data);

    Matrix input_train = input_data.splice_rows(input_data.get_rows()/5,input_data.get_rows()); 
    Matrix output_train = output_data.splice_rows(output_data.get_rows()/5,output_data.get_rows()); 
    Matrix input_test = input_data.splice_rows(0,input_data.get_rows()/5);
    Matrix output_test = output_data.splice_rows(0,output_data.get_rows()/5);

    std::vector<size_t> layer_sizes = {input_data.get_cols(),32,output_data.get_cols()}; 
    std::vector<std::string> hidden_activations = {"relu"}; 
    std::string loss = "mse";
    Network nn = Network(layer_sizes, hidden_activations, loss);

    double learning_rate = .1;
    size_t batch_size = 40;
    size_t epochs = 1000;
    nn.fit(input_train, output_train, learning_rate, batch_size, epochs);

    Matrix predicted_output = nn.predict(input_test);

    output_normalizer.denormalize(predicted_output);
    output_normalizer.denormalize(output_test);

    for (size_t i = 0; i<predicted_output.get_rows(); i++) {
        std::cout<<predicted_output.at(i,0)<<", "<<output_test.at(i,0)<<std::endl;
    }
    std::cout<<"Predicted, Actual"<<std::endl;
    
    double mae = (predicted_output - output_test).abs().mean();
    double nmae = mae/(output_test.max()-output_test.min());
    

    std::cout<<std::endl;
    std::cout<<mae<<" Mean Absolute Error"<<std::endl;
    std::cout<<nmae*100.0<<"%"<<" error relative to range"<<std::endl;
    return 0;
}



