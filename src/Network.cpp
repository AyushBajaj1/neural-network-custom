#include "Network.h"


    Network::Network(std::vector<size_t> layer_sizes, std::vector<std::string> hidden_activations, std::string loss) : activations(hidden_activations), loss(loss){
        if (layer_sizes.size()<3) throw std::invalid_argument("must have at least 3 layers"); 
        if (layer_sizes.size()!=hidden_activations.size()+2) throw std::invalid_argument("number of activation functions must match number of hidden layers");

        layer_weights.resize(layer_sizes.size()-1);
        layer_biases.resize(layer_sizes.size()-1);
        layer_outputs.resize(layer_sizes.size()-1);

        for (size_t i = 0; i<layer_sizes.size()-1; i++) {
            layer_weights[i] = Matrix(layer_sizes[i],layer_sizes[i+1]);
            layer_weights[i].randfill_inplace(std::sqrt(6.0/layer_sizes[0]));

            layer_biases[i] = Matrix(1,layer_sizes[i+1]);
        }

        if (loss=="mse") activations.push_back("linear");
        if (loss=="cross-entropy") activations.push_back("softmax");
    }

    void Network::fit(const Matrix& input_data, const Matrix& output_data, double learning_rate, size_t batch_size, size_t epochs) {
        weight_updates.resize(layer_outputs.size());
        bias_updates.resize(layer_outputs.size());

        for (size_t i = 0; i<epochs; i++) {
            for (size_t j = 0; j<input_data.get_rows(); j+=batch_size) { 
                if (j + batch_size > input_data.get_rows()) {
                    // forward_pass(input_data.splice_rows(j,input_data.get_rows()));
                    // backward_pass(input_data.splice_rows(j,input_data.get_rows()),output_data.splice_rows(j,input_data.get_rows()), learning_rate);
                    break;
                }

                Matrix input_batch = input_data.splice_rows(j,j+batch_size);
                Matrix output_batch = output_data.splice_rows(j,j+batch_size);
                
                forward_pass(input_batch);
                backward_pass(input_batch, output_batch, learning_rate);
            }
        }
    }

    void Network::forward_pass(const Matrix& input_data_batch) {
        for (size_t i = 0; i<layer_weights.size(); i++) {
            if (i==0) {
                layer_outputs[i] = input_data_batch.matmul(layer_weights[i]);
            }
            else {
                layer_outputs[i] = layer_outputs[i-1].matmul(layer_weights[i]);
            }

            layer_outputs[i].add_row_vector_inplace(layer_biases[i]);

            if (activations[i]=="relu") {
                layer_outputs[i].relu_inplace();
            }
            else if (activations[i]=="softmax") {
                layer_outputs[i].softmax_inplace();
            }
        }
    }
    
    void Network::backward_pass(const Matrix& input_data_batch, const Matrix& output_data_batch, double learning_rate) {
        Matrix loss_grad = final_output();
        loss_grad-=output_data_batch;
        loss_grad.scalarmul_inplace(1.0/output_data_batch.get_rows());

        for (int i = layer_weights.size()-1; i>=0; i--) {
            if (i!=0) {
                weight_updates[i] = layer_outputs[i-1].transpose().matmul(loss_grad);
                bias_updates[i] = loss_grad.column_wise_sum();

                loss_grad = loss_grad.matmul(layer_weights[i].transpose());
                layer_outputs[i-1].heavside_inplace();
                loss_grad.hadamard_product_inplace(layer_outputs[i-1]);
            }
            else {
                weight_updates[0] = input_data_batch.transpose().matmul(loss_grad);
                bias_updates[0] = loss_grad.column_wise_sum();
            }

            weight_updates[i].scalarmul_inplace(learning_rate);
            bias_updates[i].scalarmul_inplace(learning_rate);
        }
        
        for (size_t i = 0; i<layer_weights.size(); i++) {
            layer_weights[i]-=weight_updates[i];
            layer_biases[i]-=bias_updates[i];
        }
    }

    Matrix Network::predict(const Matrix& test_input) {
        forward_pass(test_input);
        return layer_outputs[layer_outputs.size()-1];
    }

    Matrix Network::final_output() { return layer_outputs[layer_outputs.size()-1]; }

    double Network::mae(const Matrix& input, const Matrix& output) {
        Matrix predicted_output = predict(input);
        Matrix diff = output - predicted_output;
        diff = diff.abs();
        return diff.mean();
        // double mae = 0.0;

        // for (size_t i = 0; i<diff.get_rows(); i++) {
        //     mae+=std::abs(diff.at(i,0));
        // }

        // mae/=output.get_rows();

        // return mae;
    }

    double Network::nmae(const Matrix& input, const Matrix& output) {
        double range = 40;
        return (mae(input,output)/range);
    }

    double Network::classification_accuracy(const Matrix& input, const Matrix& output) {
        Matrix predicted_output = predict(input);
        Matrix actual_output = output.hot_vector_to_digit();
        predicted_output = predicted_output.hot_vector_to_digit();

        int count = 0;
        for (size_t i = 0; i<predicted_output.get_rows(); i++) {
            if (predicted_output.at(i,0)==actual_output.at(i,0)) {
                count++;
            }
        }

        return static_cast<double>(count) / static_cast<double>(predicted_output.get_rows());
    }

    
