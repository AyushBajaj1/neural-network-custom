#pragma once
#include <iostream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>

class Matrix {
    public:
        Matrix() = default;

        Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
            if (rows==0||cols==0) {
                list=nullptr;
            }
            else {
                list = new double[rows*cols];
            }
            
        }

        double& at(size_t row, size_t col) {
            if (row<rows&&col<cols) {
                return list[row*cols+col];
            }
            else {
                throw std::invalid_argument("matrix out of bounds row or col given");
            }
        }
        const double& at(size_t row, size_t col) const {
            if (row<rows&&col<cols) {
                return list[row*cols+col];
            }
            else {
                throw std::invalid_argument("matrix out of bounds row or col given");
            }
        }

        Matrix matmul(const Matrix& other) {
            if (cols!=other.rows) {
                throw std::invalid_argument("inner dimensions do not match!");
            }  
            Matrix c(rows,other.cols);
            for (size_t i = 0; i<rows*other.cols; i++) {
                double sum = 0.0;
                for (size_t j = 0; j<cols; j++) {
                    sum+=at(i/other.cols,j)*other.at(j,i%other.cols);
                }
                c.list[i] = sum;
            }

            return c;
        }

        

        void randfill(double range) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<double> dist(-1*range, range);

            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = dist(gen);
            }
        }


        void print() {
            std::cout<<"[";
            for (size_t i = 0; i<rows*cols; i++) {
                if (i%cols==0) {
                    std::cout<<"[";
                }
                std::cout<<list[i];
                if (i%cols!=cols-1) {
                    std::cout<<", ";
                }
                else {
                    if (i==rows*cols-1) {
                        std::cout<<"]";
                    }
                    else {
                        std::cout<<"],";
                    }
                }
            }
            std::cout<<"]";
            std::cout<<std::endl;
        }

        void relu_() {
            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = std::max(list[i],0.0);
            }
        }

        void softmax_() {
            for (size_t i = 0; i<rows; i++) {
                double exp_sum = 0.0;
                for (size_t j = 0; j<cols; j++) {
                    at(i,j)=std::exp(at(i,j));
                    exp_sum+=at(i,j);
                }
                for (size_t j = 0; j<cols; j++) {
                    at(i,j)/=exp_sum;
                }
            }
        }


        //rule of 3
        ~Matrix() {
            clear();
        }
        Matrix(const Matrix& other) {
            copy(other);
        }
        Matrix& operator=(const Matrix& other) {
            if (this!=&other) {
                clear();
                copy(other);
            }
            return *this;
        }

        
    private:
        double* list = nullptr;
        size_t rows = 0;
        size_t cols = 0;

        void copy(const Matrix& other) {
            rows = other.rows;
            cols = other.cols;

            if (other.list!=nullptr) {
                list = new double[other.rows*other.cols];
                for (size_t i = 0; i<rows*cols; i++) {
                    list[i] = other.list[i];
                }
            }
            else {
                list = nullptr;
            }
            
        }

        void clear() {
            delete[] list;
            list = nullptr;
            rows=0;
            cols=0;
        }

};

