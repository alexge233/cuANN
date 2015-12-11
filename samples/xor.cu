#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <fstream>
#include <boost/archive/binary_oarchive.hpp>

int main (void)
{
    // Instantiate the activation functor and its derivative
    cuANN::sigmoid_bipolar func;
    cuANN::sigmoid_bipolar_deriv deriv;

    // Create a XOR Network: 
    // 2 input nodes, 
    // 4 hidden nodes, 
    // 1 hidden layer, 
    // 1 output node
    //
    // We do not implement BIAS nodes, thus our hidden layer should have more than 2 hidden nodes
    // In this example we use TANH as activation function, in order to compare it to sigmoid_bipolar
    //
    cuANN::ann xor_net = cuANN::ann(2,4,1,1);

    // load the training data & print it on screen
    cuANN::data train_data = cuANN::data("xor.data");

    // print data on stdout
    train_data.print();

    // When training pass as the first two params the activation functor and it's derivative.
    // Then the training data, the Epochs for which the network will be trained,
    // the amount of CPU threads (each CPU thread "learns" a pattern)
    // the stop-error, e.g., when should the network stop learning
    // the learning rate, and the momentum rate.
    float mse = xor_net.train(func,deriv,train_data,100000,100,4,.002,.2,.9);

    std::cout << "XOR Network using sigmoid bipolar trained MSE: " << mse << std::endl;

    // Lets do some manual Propagations and see what the Network Output is
    //
    // Test with [1,0] should give us [1]
    std::cout << "Testing with [1,0] as input" << std::endl;
    float x_in1[2] {1.f, 0.f};
    thrust::device_vector<float> in_vec1(x_in1,x_in1+2);
    auto output1 = xor_net.propagate( func, in_vec1 );
    std::cout << "output: ";
    for ( auto val : output1 ) std::cout << val << " (expecting 1)" <<std::endl;

    // Test with [0,1] should give us [1]
    std::cout << "Testing with [0,1] as input" << std::endl;
    float x_in2[2] {0.f, 1.f};
    thrust::device_vector<float> in_vec2(x_in2,x_in2+2);
    auto output2 = xor_net.propagate<cuANN::sigmoid_bipolar>( func, in_vec2 );
    std::cout << "output: ";
    for ( auto val : output2 ) std::cout << val << " (expecting 1)" <<std::endl;

    // Test with [0,0] should give us [0]
    std::cout << "Testing with [0,0] as input" << std::endl;
    float x_in3[2] {0.f, 0.f};
    thrust::device_vector<float> in_vec3(x_in3,x_in3+2);
    auto output3 = xor_net.propagate<cuANN::sigmoid_bipolar>( func, in_vec3 );
    std::cout << "output: ";
    for ( auto val : output3 ) std::cout << val << " (expecting 0)" <<std::endl;

    // Test with [1,1] should give us [0]
    std::cout << "Testing with [1,1] as input" << std::endl;
    float x_in4[2] {1.f, 1.f};
    thrust::device_vector<float> in_vec4(x_in4,x_in4+2);
    auto output4 = xor_net.propagate<cuANN::sigmoid_bipolar>( func, in_vec4 );
    std::cout << "output: ";
    for ( auto val : output4 ) std::cout << val << " (expecting 0)" <<std::endl;
    
    cuANN::data test_data = cuANN::data("xor.data");

    // Let's cross-reference the MSE we acquired during back-prop training, with a test
    mse = xor_net.test(func,test_data);
    std::cout << "XOR network test MSE: " << mse << std::endl;

    // save data to a binary archive   
    std::ofstream ofs("xor_sigmoid.bin");
    boost::archive::binary_oarchive oa(ofs);
    oa << xor_net;

    return 0;
}
