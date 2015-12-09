#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"

#include <iostream>
#include <thrust/version.h>

int main (void)
{
    // Instantiate the activation functor and its derivative
    cuANN::tanh_norm func;
    cuANN::tanh_norm_deriv deriv;

    // Create a XOR Network: 2 input, 4 hidden neurons, 1 hidden layer, 1 output neurons
    cuANN::ann xor_net = cuANN::ann(2,4,1,1);
    //xor_net.print_weights();

    // load the training data & print it on screen
    cuANN::data train_data = cuANN::data("xor.data");

    // Train: Activation, Derivative, Data, Epochs, Reports, Threads, Stop Error
    float mse = xor_net.train(func,
                              deriv,
                              train_data,
                              10000,
                              1000,
                              1,
                              0.002f,
                              0.2f,
                              0.9f);

    std::cout << "XOR Network using TANH trained MSE: " << mse << std::endl;

    std::cout << "Testing with [1,0] as input" << std::endl;
    float x_in1[2] {1.f, 0.f};
    thrust::device_vector<float> in_vec1(x_in1,x_in1+2);
    auto output1 = xor_net.propagate( func, in_vec1 );
    std::cout << "output: ";
    for ( auto val : output1 )
        std::cout << val << std::endl;

    std::cout << "Testing with [0,1] as input" << std::endl;
    float x_in2[2] {0.f, 1.f};
    thrust::device_vector<float> in_vec2(x_in2,x_in2+2);
    auto output2 = xor_net.propagate<cuANN::tanh_norm>( func, in_vec2 );
    std::cout << "output: ";
    for ( auto val : output2 )
        std::cout << val << std::endl;

    std::cout << "Testing with [0,0] as input" << std::endl;
    float x_in3[2] {0.f, 0.f};
    thrust::device_vector<float> in_vec3(x_in3,x_in3+2);
    auto output3 = xor_net.propagate<cuANN::tanh_norm>( func, in_vec3 );
    std::cout << "output: ";
    for ( auto val : output3 )
        std::cout << val << std::endl;

    std::cout << "Testing with [1,1] as input" << std::endl;
    float x_in4[2] {1.f, 1.f};
    thrust::device_vector<float> in_vec4(x_in4,x_in4+2);
    auto output4 = xor_net.propagate<cuANN::tanh_norm>( func, in_vec4 );
    std::cout << "output: ";
    for ( auto val : output4 )
        std::cout << val << std::endl;
    
    return 0;
}
