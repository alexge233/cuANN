#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>

int main (void)
{
    // Instantiate the activation functor and its derivative
    cuANN::tanh_scaled func;
    cuANN::tanh_scaled_deriv deriv;

    // Create a XOR Network: 
    // 2 input nodes, 
    // 4 hidden nodes, 
    // 1 hidden layer, 
    // 1 output node
    //
    // We do not implement BIAS nodes, thus our hidden layer should have more than 2 hidden nodes
    // In this example we use TANH as activation function, in order to compare it to sigmoid_bipolar
    //
    cuANN::ann net = cuANN::ann(2,4,1,1);

    // load the training data & print it on screen
    cuANN::data train_data = cuANN::data("../data/xor.data");

    // When training pass as the first two params the activation functor and it's derivative.
    // Then the training data, the Epochs for which the network will be trained,
    // the amount of CPU threads (each CPU thread "learns" a pattern)
    // the stop-error, e.g., when should the network stop learning
    // the learning rate, and the momentum rate.
    float mse = net.train(func,deriv,train_data,20000,1000,1,.0001,.2,.9);
    cudaDeviceSynchronize();
    std::cout << "XOR network using tanh_caled back-prop MSE: " << std::fixed << mse << std::endl;
    // Let's cross-reference the MSE we acquired during back-prop training, with a test
    std::cout << "XOR network test MSE: " << net.test(func,train_data) << std::endl;

    // Lets do some manual Propagations and see what the Network Output is
    //
    std::cout << "test [1,0] as input; ";
    float x_in1[2] {1,0};
    thrust::device_vector<float> in_vec1(x_in1,x_in1+2);
    auto output1 = net.propagate( func, in_vec1 );
    std::cout << "output: ";
    for ( auto val : output1 ) std::cout << std::fixed << val << " (expecting 1)" << std::endl;

    std::cout << "test [0,1] as input; ";
    float x_in2[2] {0,1};
    thrust::device_vector<float> in_vec2(x_in2,x_in2+2);
    auto output2 = net.propagate( func, in_vec2 );
    cudaDeviceSynchronize();
    std::cout << "output: ";
    for ( auto val : output2 ) std::cout << std::fixed << val << " (expecting 1)" << std::endl;

    std::cout << "test [0,0] as input; ";
    float x_in3[2] {0,0};
    thrust::device_vector<float> in_vec3(x_in3,x_in3+2);
    auto output3 = net.propagate( func, in_vec3 );
    cudaDeviceSynchronize();
    std::cout << "output: ";
    for ( auto val : output3 ) std::cout << std::fixed << val << " (expecting 0)" << std::endl;

    std::cout << "test [1,1] as input; ";
    float x_in4[2] {1,1};
    thrust::device_vector<float> in_vec4(x_in4,x_in4+2);
    auto output4 = net.propagate( func, in_vec4 );
    cudaDeviceSynchronize();
    std::cout << "output: ";
    for ( auto val : output4 ) std::cout << std::fixed << val << " (expecting 0)" << std::endl;
    
    // save data to archive
    std::ofstream ofs("xor_tanh_scaled.net");
    boost::archive::text_oarchive oa(ofs);
    oa << net;

    return 0;
}
