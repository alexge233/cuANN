#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <thrust/version.h>

int main ()
{
    // Activation functor and its derivative
    cuANN::sigmoid func;
    cuANN::sigmoid_deriv deriv;

    // Abelone Network: 10 input neurons, 20 hidden, 2 hidden layers
    cuANN::ann network = cuANN::ann(10,20,2,1);

    // Train: Load from File
    cuANN::data train_data = cuANN::data("abelone.train");

    // Train: Activation, Derivative, Data, Epochs, Reports, Threads, Stop Error
    auto mse = network.train(func,deriv,train_data,1000,100,8,0.02f);
    std::cout << "Trained Abelone with MSE: " << mse << std::endl;

    return 0;
}
