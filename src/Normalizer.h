#pragma once

#include "Matrix.h"

class Normalizer {
    public:
        Normalizer() = default;

        Normalizer(Matrix& matrix);
        void normalize (Matrix& matrix);
        void denormalize(Matrix& matrix);

    private:
        std::vector<double> mean;
        std::vector<double> std;


};