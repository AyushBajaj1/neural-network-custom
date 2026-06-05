#include "Matrix.h"



int main () {
    std::cout<<"Matrix Testing"<<std::endl;

    Matrix A(1,4);
    A.randfill(3); //20 is the range of values, so A will have random uniform values between -20 and 20
    A.print();
    A.softmax_();
    A.print();


    // Matrix B(4,3);
    // B.randfill(10);
    // B.print();

    // (A.matmul(B)).print();


    return 0;
}