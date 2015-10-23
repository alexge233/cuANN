#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    std::cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl;

    // Deep network: input, hidden, hidden layers, output
    cuANN::ann large_net = cuANN::ann( 4, 8, 2, 2 );

    //float x[10] {1.f, 0.f, 0.3f, .98f, 0.01f, 0.57f, 0.88f,.99f,0.001f, 1.f};
    //thrust::device_vector<float> in_vec( x, x + 10 );
    
    float x_in[4] {1.f, 1.f, 1.f, 1.f};
    thrust::device_vector<float> in_vec(x_in,x_in+4);
    float x_out[2] { 1.f, 0.f};
    thrust::device_vector<float> out_vec(x_out,x_out+2);

    // Propagate
    thrust::host_vector<float> output = large_net.propagate( in_vec );

    return 0;
}
