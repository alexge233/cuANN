#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;
    std::cout << "Thrust v" << major << "." << minor << std::endl;

    // TODO: Test a network without hidden layers, 
    //       one with 1 hidden, 
    //       and a deep network with many hidden layers

    // Create a XOR Network: 2 input, 2 hidden neurons, 1 hidden layer, 2 output neurons
    cuANN::ann test = cuANN::ann(  10, 5000, 10, 2 );

    // Load from file (deserialise) the XOR data (see FANN xor data)

    //float x[2] {1.f, 0.f};
    float x[10] {1.f, 0.f, 0.3f, .98f, 0.01f, 0.57f, 0.88f,.99f,0.001f, 1.f};
    thrust::device_vector<float> in_vec( x, x + 10 );

    // Propagates through all layers
    thrust::host_vector<float> output = test.propagate( in_vec );

    std::cout << "propagated output" << std::endl;
    for ( int i = 0; i < output.size(); i++ )
        std::cout << output[i] << std::endl;

    std::cout << std::endl;

    return 0;
}
