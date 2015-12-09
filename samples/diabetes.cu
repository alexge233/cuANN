#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"

#include <iostream>

int main (void)
{
    // Instantiate the activation functor and its derivative
    cuANN::tanh_scaled func;
    cuANN::tanh_scaled_deriv deriv;

    // Create a Network: 8 input, 8 hidden neurons, 1 hidden layer, 2 output neurons
    cuANN::ann net = cuANN::ann(8,8,1,2);

    // load the training data (384 samples)
    cuANN::data train_data = cuANN::data("diabetes.train");

    // Train: Activation, Derivative, Data, Epochs, Reports, Threads, Stop Error
    float mse = net.train(func,deriv,train_data,10000,100,8,0.002f,0.2f,0.9f);

    // Print MSE
    std::cout << "diabetes Network using TANH trained MSE: " << mse << std::endl;
    
    return 0;
}
