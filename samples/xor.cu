#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    std::cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl;

    // XOR Example
    // Create a XOR Network: 2 input, 2 hidden neurons, 1 hidden layer, 1 output neurons
    cuANN::ann xor_net = cuANN::ann(  2, 2, 1, 1 );
    cuANN::data train_data = cuANN::data( "xor.data" );

    // Train on this data, MSE, Epochs
    float mse = xor_net.train( train_data, 0.05f, 10000, 1000, false );
    std::cout << "Trained Network with MSE: " << mse << std::endl;

    std::cout << "Testing with [1,0] as input" << std::endl;
    float x_in[2] {1.f, 0.f};
    thrust::device_vector<float> in_vec(x_in,x_in+2);
    thrust::device_vector<float> output = xor_net.propagate( in_vec );
    std::cout << "output" << std::endl;
    for ( auto val : output )
        std::cout << val << std::endl;

    return 0;
}
