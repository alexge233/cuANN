#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    std::cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl;

    // XOR Example
    // Create a XOR Network: 2 input, 2 hidden neurons, 1 hidden layer, 1 output neurons
    cuANN::ann test = cuANN::ann(  2, 2, 1, 1 );
    cuANN::data train_data = cuANN::data( "xor.data" );
    //cuANN::data train_data = cuANN::data("abelone.train");
    /*
    float x[10] {1.f, 0.f, 0.3f, .98f, 0.01f, 0.57f, 0.88f,.99f,0.001f, 1.f};
    thrust::device_vector<float> in_vec( x, x + 10 );
    thrust::host_vector<float> output = test.propagate( in_vec );
    std::cout << "propagated output" << std::endl;
    for ( int i = 0; i < output.size(); i++ )
        std::cout << output[i] << std::endl;
    std::cout << std::endl;
    */

    return 0;
}
