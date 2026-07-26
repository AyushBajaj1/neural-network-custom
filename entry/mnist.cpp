#include "../src/Network.h"

int main () {

    Matrix train;
    Matrix test;
    train.read_csv("./data/mnist_train.csv"); 
    test.read_csv("./data/mnist_test.csv");

    Matrix input_train = train.splice_cols(1,785); input_train = input_train.scalarmul(1/255.0);
    Matrix input_test = test.splice_cols(1,785); input_test = input_test.scalarmul(1/255.0);
    Matrix output_train = train.splice_cols(0,1); output_train=output_train.digit_to_one_hot_vector();
    Matrix output_test = test.splice_cols(0,1); output_test=output_test.digit_to_one_hot_vector();
    
    std::vector<size_t> layer_sizes = {input_train.get_cols(),64,output_train.get_cols()}; 
    std::vector<std::string> hidden_activations = {"relu"}; 
    std::string loss = "cross-entropy";
    Network nn = Network(layer_sizes, hidden_activations, loss);

    double learning_rate = .1;
    size_t batch_size = 32;
    size_t epochs = 10;
    
    nn.fit(input_train, output_train, learning_rate, batch_size, epochs);

    Matrix predicted_output = nn.predict(input_test);

    
    output_test = output_test.hot_vector_to_digit();
    predicted_output = predicted_output.hot_vector_to_digit();

    int count = 0;
    for (size_t i = 0; i<predicted_output.get_rows(); i++) {
        
        std::cout<<"---"<<std::endl;
        std::cout<<predicted_output.at(i,0)<<std::endl;
        std::cout<<output_test.at(i,0)<<std::endl;

        if (predicted_output.at(i,0)==output_test.at(i,0)) {
            count++;
        }
    }
    std::cout<<"---"<<std::endl;
        std::cout<<"Predicted"<<std::endl;
        std::cout<<"Actual"<<std::endl;
        std::cout<<"---"<<std::endl;

    double accuracy = static_cast<double>(count) / predicted_output.get_rows();
    std::cout<<accuracy*100<<"% accuracy"<<std::endl;

    return 0;
}



