#include "Matrix.h"

        Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
            if (rows==0||cols==0) {
                list=nullptr;
            }
            else {
                list = new double[rows*cols]{};
            } 
        }

        size_t Matrix::get_rows() const { return rows; } 
        size_t Matrix::get_cols() const { return cols; }

        double& Matrix::at(size_t row, size_t col) {
            if (row<rows&&col<cols) {
                return list[row*cols+col];
            }
            else {
                throw std::invalid_argument("matrix out of bounds row or col given");
            }
        }

        const double& Matrix::at(size_t row, size_t col) const {
            if (row<rows&&col<cols) {
                return list[row*cols+col];
            }
            else {
                throw std::invalid_argument("matrix out of bounds row or col given");
            }
        }

        Matrix Matrix::matmul(const Matrix& other) const {
            if (cols!=other.rows) {
                throw std::invalid_argument("inner dimensions do not match!");
            }  
            
            Matrix c(rows,other.cols);
            for (size_t i = 0; i<rows; i++) {
                for (size_t k = 0; k<cols; k++) {
                    for (size_t j = 0; j<other.cols; j++) {
                        c.at(i,j)+=at(i,k)*other.at(k,j);
                    }
                }
            }

            // for (size_t i = 0; i<rows; i++) {
            //     for (size_t j = 0; j<other.cols; j++) {
            //         double sum = 0.0;
            //         for (size_t k = 0; k<cols; k++) {
            //             sum+=at(i,k)*other.at(k,j);
            //         }   
            //         c.at(i,j) = sum;
            //     }
            // }

            return c;
        }

       Matrix Matrix::scalarmul(const double& scalar) const {
            Matrix c(rows,cols);
            for (size_t i = 0; i<rows*cols; i++) {
                c.list[i] = list[i]*scalar;
            }

            return c;
       }

       void Matrix::scalarmul_inplace(const double& scalar) const {
            for (size_t i = 0; i<rows*cols; i++) {
                list[i]*=scalar;
            }
       }

       Matrix Matrix::transpose() const {
            Matrix c(cols,rows);
            for (size_t i = 0; i<cols*rows; i++) {
                c.list[(i%cols)*rows+i/cols] = list[i];
            }
            return c;
       }

        Matrix Matrix::splice_rows(size_t start, size_t end) const {
            Matrix A(end-start,cols);
            for (size_t i = 0; i<A.rows; i++) {
                for (size_t j = 0; j<A.cols; j++) {
                    A.at(i,j) = at(i+start,j);
                }
            }
            return A;
        }

        Matrix Matrix::splice_cols(size_t start, size_t end) const {
            Matrix A(rows,end-start);
            for (size_t i = 0; i<A.rows; i++) {
                for (size_t j = 0; j<A.cols; j++) {
                    A.at(i,j) = at(i,j+(start));
                }
            }
            return A;
        }

        Matrix Matrix::combine_rows(const Matrix& other) {
            if (cols!=other.cols) {
                throw std::invalid_argument("matricies must have identical number of columns for combining rows");
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

        

        void Matrix::randfill_inplace(const double& range) {
            static std::random_device rd;
            static std::mt19937 gen(seed+1);
            static std::uniform_real_distribution<double> dist(-1*range, range);

            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = dist(gen);
            }
        }

        Matrix Matrix::randfill(const double& range) const {
            static std::random_device rd;
            static std::mt19937 gen(seed);
            static std::uniform_real_distribution<double> dist(-1*range, range);
            Matrix c(rows,cols);

            for (size_t i = 0; i<rows*cols; i++) {
                c.list[i] = dist(gen);
            }

            return c;
        }

        void Matrix::fill_inplace(const double& value ) {
            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = value;
            }
        }

        // Matrix he_init(size_t fan_in) {
        //     static std::random_device rd;
        //     static std::mt19937 gen(seed);
        //     static std::uniform_real_distribution<double> dist(-std::sqrt(6.0/fan_in),std::sqrt(6.0/fan_in));
        //     Matrix c(rows,cols);

        //     for (size_t i = 0; i<rows*cols; i++) {
        //         c.list[i] = dist(gen);
        //     }

        //     return c;

        // }



        Matrix Matrix::fill(double num) const {
            Matrix c(rows,cols);

            for (size_t i = 0; i<rows*cols; i++) {
                c.list[i] = num;
            }

            return c;
        }

        Matrix Matrix::heavside() {
            Matrix c = *this;
            for (size_t i = 0; i<rows*cols; i++) {
                if (c.list[i]<=0) {
                    c.list[i] = 0;
                }
                else {
                    c.list[i] = 1;
                }
            }
            return c;
        }
        void Matrix::heavside_inplace() {
            for (size_t i = 0; i<rows*cols; i++) {
                if (list[i]<=0) {
                    list[i] = 0;
                }
                else {
                    list[i] = 1;
                }
            }
        }
        void Matrix::leaky_heavside_inplace() {
            for (size_t i = 0; i<rows*cols; i++) {
                if (list[i]<=0) {
                    list[i] = 0.01;
                }
                else {
                    list[i] = 1;
                }
            }
        }

        Matrix Matrix::hadamard_product(const Matrix& other) const {
            if (rows!=other.rows || cols!=other.cols) throw std::invalid_argument("matricies must have identical dimensions for hadamard_product");
            Matrix c(rows,cols);

            for (size_t i = 0; i<rows*cols; i++) {
                c.list[i] = list[i]*other.list[i];
            }

            return c;
        }

        void Matrix::hadamard_product_inplace(const Matrix& other) {
            if (rows!=other.rows || cols!=other.cols) throw std::invalid_argument("matricies must have identical dimensions for hadamard_product");
            
            for (size_t i = 0; i<rows*cols; i++) {
                list[i] *= other.list[i];
            }
        }

        Matrix Matrix::digit_to_one_hot_vector() const {
            //other is a column vector of where each entry is a decimal digit (0-9). 
            //Return a matrix where each decimal digits is expanded to a 1 x 10 one hot vector
            if (get_cols()!=1) {
                throw std::invalid_argument("argument to digit_to_one_hot_vector must be column vector");
            }

            Matrix matrix(0,10);
            for (size_t i = 0; i<get_rows(); i++) {
                Matrix row(1,10);
                row.fill_inplace(0);
                for (size_t j = 0; j<10; j++) {
                    if (j==at(i,0)) {
                        row.at(0,j) = 1;
                    }
                }
                matrix = matrix.combine_rows(row);
            }

            return matrix;
        }

        Matrix Matrix::hot_vector_to_digit() const {
            Matrix c(get_rows(),1);
            for (size_t i = 0; i<get_rows(); i++) {
                int max_j = 0;
                for (size_t j = 0; j<get_cols(); j++) {
                    if (at(i,j)>at(i,max_j)) {
                        max_j = j;
                    }
                }
                c.at(i,0)=max_j;
            }
            return c;
        }


        void Matrix::relu_inplace() {
            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = std::max(list[i],0.0);
            }
        }
        void Matrix::leaky_relu_inplace() {
            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = list[i] > 0.0 ? list[i] : list[i]*0.01;
            }
        }
        void Matrix::softmax_inplace() {
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

        Matrix& Matrix::operator-=(const Matrix& other) {
            if (other.rows!=rows||other.cols!=cols) {
                throw std::invalid_argument("cant subtract matrices of different dimensions");
            }
            for (size_t i = 0; i<rows*cols; i++) {
                list[i]-=other.list[i];
            }
            return *this;
        }


        Matrix Matrix::operator-(const Matrix& other) const {
            Matrix c = *this;
            c-=other;
            return c;
        }

        Matrix Matrix::column_wise_sum() {
            Matrix c(1,cols);

            for (size_t i = 0; i<rows; i++) {
                for (size_t j = 0; j<cols; j++) {
                    c.at(0,j)+=at(i,j);
                }
            }

            return c;
        }

        void Matrix::add_row_vector_inplace(const Matrix& other) {
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

        void Matrix::abs_inplace() {
            for (size_t i = 0; i<rows*cols; i++) {
                list[i] = std::abs(list[i]);
            }
        }
        Matrix Matrix::abs() {
            Matrix c(rows,cols);
            for (size_t i = 0; i<rows*cols; i++) {
                c.list[i] = std::abs(list[i]);
            }

            return c;
        }

        Matrix Matrix::shuffle_rows() {
            std::vector<int> list(get_rows(),-1);
            static std::mt19937 gen(seed);
            static std::uniform_int_distribution dist(0,static_cast<int>(get_rows()-1));
            
            Matrix c(0,cols);

            while (c.rows!=rows) {
                int i = dist(gen);
                while (list[i]!=-1) {
                    i = dist(gen);
                }
                c = c.combine_rows(splice_rows(i,i+1));
                list[i] = 1;
            }

            return c;
        }

        void Matrix::read_csv(std::string filename) {
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

        void Matrix::printline() const {
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

        void Matrix::printblock() const {
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

        double Matrix::frobenius_norm_sq() const {
            double norm = 0.0;
            for (size_t i = 0; i<rows*cols; i++) {
                norm += list[i]*list[i];
            }

            return norm;
        }

        double Matrix::cross_entropy_loss(const Matrix& Y) const {
            double loss = 0.0;
            for (size_t i = 0; i<Y.get_rows(); i++) {
                loss+=std::log(at(i,Y.splice_rows(i,i+1).hot_vector_to_digit().at(0,0)));
            }

            return loss*-(1.0/Y.get_rows());
        }

        double Matrix::mean() {
            double num = 0.0;
            for (size_t i = 0; i<rows*cols; i++) {
                num+=list[i];
            }

            return num/(rows*cols);
        }

        double Matrix::var() {
            double mean_of_squares = 0.0;
            for (size_t i = 0; i<rows*cols; i++) {
                mean_of_squares+=(list[i]*list[i]);
            }
            mean_of_squares/=(rows*cols);
            double mean = this->mean();
            double mean_squared = mean*mean;
            return mean_of_squares - mean_squared;
        }

        double Matrix::max() {
            double max = list[0];
            for (size_t i = 1; i<rows*cols; i++) {
                if (list[i]>max) {
                    max=list[i];
                }
            }

            return max;
        }

        double Matrix::min() {
            double min = list[0];
            for (size_t i = 1; i<rows*cols; i++) {
                if (list[i]<min) {
                    min=list[i];
                }
            }

            return min;
        }

        //rule of 5
        Matrix::~Matrix() {
            clear();
        }
        Matrix::Matrix(const Matrix& other) {
            copy(other);
        }
        Matrix& Matrix::operator=(const Matrix& other) {
            if (this!=&other) {
                clear();
                copy(other);
            }
            return *this;
        }
        Matrix::Matrix(Matrix&& other) noexcept {
            rows = other.rows;
            cols = other.cols;
            list = other.list;

            other.list = nullptr;
            other.rows = 0;
            other.cols = 0;
        }
        Matrix& Matrix::operator=(Matrix&& other) noexcept {
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

        void Matrix::copy(const Matrix& other) {
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


        void Matrix::clear() {
            delete[] list;
            list = nullptr;
            rows=0;
            cols=0;
        }

