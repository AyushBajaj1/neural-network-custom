#include "Matrix.h"



int main () {
    std::cout<<"Matrix Testing"<<std::endl;

    Matrix A(3,3);
    A.randfill_inplace(10);
    A.printblock();

    A.column_wise_sum().printblock();


    // Matrix A(3,3);
    // A.randfill_inplace(10);
    // A.printblock();
    // std::cout<<std::endl;

    // Matrix B(2,3);
    // B.randfill_inplace(20);
    // B.printblock();
    // std::cout<<std::endl;

    // A.combine_rows(B).printblock();

    return 0;
}