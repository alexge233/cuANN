#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    std::cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl;

    // Deep network: 4 input, 6 hidden, 1 hidden layer, 2 output neurons
    cuANN::ann large_net = cuANN::ann( 4, 4, 2, 2 );

    float x_in[4] {1.f, 1.f, 1.f, 1.f};
    thrust::device_vector<float> in_vec(x_in,x_in+4);
    float x_out[2] { 1.f, 0.f};
    thrust::device_vector<float> out_vec(x_out,x_out+2);
    thrust::host_vector<float> output = large_net.propagate(in_vec);
    std::cout << "output" << std::endl;
    for ( auto val : output )
        std::cout << val << std::endl;
    return 0;
}
