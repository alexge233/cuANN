#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    std::cout << "Abelone Test" << std::endl;

    cuANN::ann network = cuANN::ann(  10, 20, 4, 1 );
    cuANN::data train_data = cuANN::data( "abelone.train" );
    network.train( train_data, 0.02f, 1 ); 

    return 0;
}
