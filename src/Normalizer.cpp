#include "Normalizer.h"

    Normalizer::Normalizer(Matrix& matrix) {
        normalize(matrix);
    }
    
    void Normalizer::normalize (Matrix& matrix) {
        mean.resize(matrix.get_cols());
        std.resize(matrix.get_cols());

        for (size_t i = 0; i<matrix.get_cols(); i++) {
            mean[i] = matrix.splice_cols(i,i+1).mean();
            std[i] = std::sqrt(matrix.splice_cols(i,i+1).var());
        }

        for (size_t i = 0; i<matrix.get_rows(); i++) {
            for (size_t j = 0; j<matrix.get_cols(); j++) {
                matrix.at(i,j)-=mean[j];
                matrix.at(i,j)/=std[j];
            }
        }
    }

    void Normalizer::denormalize(Matrix& matrix) {
        if (mean.empty()||std.empty()) {
            throw std::runtime_error("cannot denormalize without first normalizing");
        }

        for (size_t i = 0; i<matrix.get_rows(); i++) {
            for (size_t j = 0; j<matrix.get_cols(); j++) {
                matrix.at(i,j)*=std[j];
                matrix.at(i,j)+=mean[j];

            }
        }
    }