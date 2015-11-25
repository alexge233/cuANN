#include "../src/ann/ann.hpp"
#include <iostream>
#include <thrust/version.h>

int main (void)
{
    std::cout << "Thrust v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl;

    // XOR Example
    // Create a XOR Network: 2 input, 4 hidden neurons, 2 hidden layer, 1 output neurons
    cuANN::ann xor_net = cuANN::ann(  2, 4, 2, 1 );
    cuANN::data train_data = cuANN::data( "xor.data" );

    // Train on this data, MSE, Epochs
    float mse = xor_net.train( train_data, 0.02f, 20000, 1000, false );
    std::cout << "Trained Network with MSE: " << mse << std::endl;

    std::cout << "Input: [1,0]" << std::endl;
    float x_in1[2] {1.f, 0.f};
    thrust::device_vector<float> in_vec1(x_in1,x_in1+2);
    auto output1 = xor_net.propagate( in_vec1 );
    std::cout << "output: ";
    for ( auto val : output1 )
        std::cout << val << " Ideal: 1.0" <<std::endl;

    std::cout << "Input: [0,1]" << std::endl;
    float x_in2[2] {0.f, 1.f};
    thrust::device_vector<float> in_vec2(x_in2,x_in2+2);
    auto output2 = xor_net.propagate( in_vec2 );
    std::cout << "output: ";
    for ( auto val : output2 )
        std::cout << val << " Ideal: 1.0" <<std::endl;

    std::cout << "Input: [0,0]" << std::endl;
    float x_in3[2] {0.f, 0.f};
    thrust::device_vector<float> in_vec3(x_in3,x_in3+2);
    auto output3 = xor_net.propagate( in_vec3 );
    std::cout << "output: ";
    for ( auto val : output3 )
        std::cout << val << " Ideal: 0.0" <<std::endl;

    std::cout << "Input: [1,1]" << std::endl;
    float x_in4[2] {1.f, 1.f};
    thrust::device_vector<float> in_vec4(x_in4,x_in4+2);
    auto output4 = xor_net.propagate( in_vec4 );
    std::cout << "output: ";
    for ( auto val : output4 )
        std::cout << val << " Ideal: 0.0" <<std::endl;

    return 0;
}
