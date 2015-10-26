#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    std::cout << "Abelone Test" << std::endl;

    // Create a XOR Network: 10 input, 0 hidden neurons, 0 hidden layers, 1 output neuron
    cuANN::ann network = cuANN::ann(  10, 0, 0, 1 );
    cuANN::data train_data = cuANN::data( "abelone.train" );

    //float mse = network.epoch ( train_data );
    //std::cout << "epoch mse: " << mse << std::endl;

    return 0;
}
