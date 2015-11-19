#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    std::cout << "Abelone Test" << std::endl;

    // WARNING: Too many hidden layers may return zero gradients!
    cuANN::ann network = cuANN::ann(  10, 5, 1, 1 );
    cuANN::data train_data = cuANN::data( "abelone.train" );
    auto mse = network.train( train_data, 0.02f, 10000, 1000, false ); 
    std::cout << "Trained Abelone with MSE: " << mse << std::endl;

    return 0;
}
