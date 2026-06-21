#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>

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

        Matrix matmul(const Matrix& other) const {
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

       Matrix scalarmul(const double& scalar) const {
            Matrix c(rows,cols);
            for (size_t i = 0; i<rows*cols; i++) {
                c.list[i] = list[i]*scalar;
            }

            return c;
       }

       Matrix transpose() const {
            Matrix c(cols,rows);

            for (size_t i = 0; i<cols*rows; i++) {
                c.list[(i%cols)*rows+i/cols] = list[i];
            }

            return c;
       }

        Matrix splice_rows(size_t start, size_t end) const {
            Matrix A(end-start,cols);
            for (size_t i = 0; i<A.rows; i++) {
                for (size_t j = 0; j<A.cols; j++) {
                    A.at(i,j) = at(i+start,j);
                }
            }

            return A;
        }

        Matrix splice_cols(size_t start, size_t end) const {
            Matrix A(rows,end-start);
            for (size_t i = 0; i<A.rows; i++) {
                for (size_t j = 0; j<A.cols; j++) {
                    A.at(i,j) = at(i,j+(start));
                }
            }

            return A;
        }

        Matrix combine_rows(const Matrix& other) {
            if (cols!=other.cols) {
                throw std::invalid_argument("matricies must have identical number of columns for rowwise_append");
            }

            Matrix temp(rows+other.rows,cols);
            for (size_t i = 0; i<cols*rows; i++) {
                temp.list[i] = list[i];
            }
            for (size_t i = cols*rows; i<cols*(rows+other.rows); i++) {
                temp.list[i] = other.list[i-cols*rows];
            }

            return temp;
        }
        

        void randfill_inplace(double range) {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<double> dist(-1*range, range);

            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = dist(gen);
            }
        }
        Matrix randfill(double range) const {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<double> dist(-1*range, range);
            Matrix c(rows,cols);

            for (size_t i = 0; i<rows*cols; i++) {
                c.list[i] = dist(gen);
            }

            return c;
        }
        void relu_inplace() {
            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = std::max(list[i],0.0);
            }
        }
        void softmax_inplace() {
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

        Matrix& operator-=(const Matrix& other) {
            if (other.rows!=rows||other.cols!=cols) {
                throw std::invalid_argument("cant subtract matrices of different dimensions");
            }
            for (size_t i = 0; i<rows*cols; i++) {
                list[i]-=other.list[i];
            }
            return *this;
        }

        Matrix operator-(const Matrix& other) const {
            Matrix c = *this;
            c-=other;
            return c;
        }

        Matrix column_wise_sum() {
            Matrix c(1,cols);

            for (size_t i = 0; i<rows; i++) {
                for (size_t j = 0; j<cols; j++) {
                    c.at(0,j)+=at(i,j);
                }
            }

            return c;
        }

        void add_row_vector_inplace(const Matrix& other) {
            if (other.rows!=1) {
                throw std::invalid_argument("add_row_vector must take in a rows=1 matrix");
            }
            if (other.cols!=this->cols) {
                throw std::invalid_argument("row_vector must have same columns as matrix in add_row_vector");
            }

            for (size_t i = 0; i<rows; i++) {
                for (size_t j = 0; j<cols; j++) {
                    at(i,j) += other.at(0,j);
                }
            }
        }

        void read_csv(std::string filename) {
            std::ifstream ifs(filename);
            if (!ifs.is_open()) {
                throw std::invalid_argument("file unopened");
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

            clear();
            list = new double[rows.size()*rows[0].size()];
            this->rows=rows.size();
            this->cols=rows[0].size();

            for (size_t i = 0; i<rows.size(); i++) {
                for (size_t j = 0; j<rows[0].size(); j++) {
                    list[i*rows[0].size()+j]= rows[i][j];
                }
            }
        }

        void printline() const {
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

        void printblock() const {
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
                        std::cout<<std::endl;
                    }
                }
            }
            std::cout<<"]";
            std::cout<<std::endl;
        }

        //rule of 5
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

        Matrix(Matrix&& other) noexcept {
            rows = other.rows;
            cols = other.cols;
            list = other.list;

            other.list = nullptr;
            other.rows = 0;
            other.cols = 0;
        }

        Matrix& operator=(Matrix&& other) noexcept {
            if (this!=&other) {
                clear();
                rows=other.rows;
                cols=other.cols;
                list = other.list;

                other.rows = 0;
                other.cols = 0;
                other.list = nullptr;
            }
            return *this;
        }

        size_t get_rows() const { return rows; } 
        size_t get_cols() const {return cols; }
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

