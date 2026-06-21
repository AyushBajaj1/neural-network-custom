#pragma once 

#include "Matrix.h"

class Network {
    public:

        Network(std::vector<size_t> layer_sizes, std::vector<std::string> hidden_activations, std::string loss) : activations(hidden_activations), loss(loss){
            if (layer_sizes.size()<3) throw std::invalid_argument("must have at least 3 layers"); 
            if (layer_sizes.size()!=hidden_activations.size()+2) throw std::invalid_argument("number of activation functions must match number of hidden layers");

            for (size_t i = 0; i<layer_sizes.size()-1; i++) {
                layer_weights.push_back(Matrix(layer_sizes[i],layer_sizes[i+1]).randfill(1));
                layer_biases.push_back(Matrix(1,layer_sizes[i+1]).randfill(1));
            }

            layer_outputs.resize(layer_weights.size());
            weight_updates.resize(layer_weights.size());
            bias_updates.resize(layer_biases.size());

            if (loss=="mse") activations.push_back("linear");
            if (loss=="cross-entropy") activations.push_back("softmax");
        }

        void fit(const Matrix& input_data, const Matrix& output_data, double learning_rate, size_t batch_size, size_t epochs) {
            this->learning_rate = learning_rate;

            for (size_t i = 0; i<epochs; i++) {
                for (size_t j = 0; j<input_data.get_rows(); j+=batch_size) { 
                    
                    if (j + batch_size > input_data.get_rows()) {
                        ForwardPass(input_data.splice_rows(j,input_data.get_rows()));
                        BackwardPass(input_data.splice_rows(j,input_data.get_rows()),output_data.splice_rows(j,input_data.get_rows()));
                        break;
                    }

                    ForwardPass(input_data.splice_rows(j,j+batch_size));
                    BackwardPass(input_data.splice_rows(j,j+batch_size),output_data.splice_rows(j,j+batch_size));
                    // std::cout<<i<<" "<<j<<std::endl;
                }
            }
        }

        void ForwardPass(const Matrix& input_data_batch) {
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
                //else if "linear", do nothing;
            }
        }
       
        void BackwardPass(const Matrix& input_data_batch, const Matrix& output_data_batch) {
            Matrix mse_loss_grad = (layer_outputs[layer_outputs.size()-1]-output_data_batch).scalarmul(2.0/output_data_batch.get_rows());
            
            for (int i = layer_weights.size()-1; i>0; i--) {
                weight_updates[i] = layer_outputs[i-1].transpose().matmul(mse_loss_grad);
                bias_updates[i] = mse_loss_grad.column_wise_sum();

                mse_loss_grad = mse_loss_grad.matmul(layer_weights[i].transpose());
            }
            weight_updates[0] = input_data_batch.transpose().matmul(mse_loss_grad);
            bias_updates[0] = mse_loss_grad.column_wise_sum();
            
            for (size_t i = 0; i<layer_weights.size(); i++) {
                layer_weights[i]-=weight_updates[i].scalarmul(learning_rate);
                layer_biases[i]-=bias_updates[i].scalarmul(learning_rate);
            }
        }

        Matrix TestSingleBatch(const Matrix& test_input) {
            ForwardPass(test_input);
            return FinalOutput();
        }

        Matrix Test(const Matrix& test_input, size_t batch_size) {
            Matrix predicted_output(test_input.get_rows(), layer_weights[layer_weights.size()-1].get_cols());

            std::vector<Matrix> batches;
            for (size_t i = 0; i<test_input.get_rows(); i+=batch_size) {
                batches.push_back(TestSingleBatch(test_input.splice_rows(i,i+batch_size)));
            }

            Matrix output = batches[0];
            for (size_t i = 1; i<batches.size(); i++) {
                output = output.combine_rows(batches[i]);
            }

            return output;
        }

        double nmae_accuracy(const Matrix& input, const Matrix& output) {
            Matrix predicted_output = Test(input, 1);
            Matrix diff = output - predicted_output;

            double sum = 0.0;
            for (size_t i = 0; i<diff.get_rows(); i++) {
                for (size_t j = 0; j<diff.get_cols(); j++) {
                    sum+=std::abs(diff.at(i,j));
                }
            }
            sum/=(diff.get_rows()*diff.get_cols());

            return (sum/40.0)*100;
        }

        Matrix FinalOutput() {
            return layer_outputs[layer_outputs.size()-1];
        }

        std::vector<Matrix> layer_weights;
        std::vector<Matrix> layer_biases;
        std::vector<Matrix> layer_outputs;

        std::vector<Matrix> weight_updates;
        std::vector<Matrix> bias_updates;
    private:
        
        double learning_rate = 0.0;
        std::vector<std::string> activations;
        std::string loss;

        
};