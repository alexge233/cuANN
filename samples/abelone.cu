#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <thrust/version.h>
///
/// @author Alex Giokas <a.gkiokas@warwick.ac.uk>
/// @date December 2015
/// 
/// This is an example of using a Network with 10 input nodes, 20 hidden nodes, 2 layers and 1 output node.
/// The dataset used is Abelone: https://archive.ics.uci.edu/ml/datasets/Abalone
/// This example uses the Hyperbolic Tangent Scaled Function, although the Abelone Dataset is scaled to [0,1]
///
/// We spawn up to 8 Parallel CPU threads, in order to Maximize GPU utilisation.
/// Expected stop error (MSE) is 20%, using 0.7 learning rate and 0.1 momentum
///
int main ()
{
    // Activation functor and its derivative
    cuANN::soft_sign func;
    cuANN::soft_sign_deriv deriv;
    // Create ANN
    cuANN::ann network = cuANN::ann(10,40,4,1);
    // Train: Load from File
    cuANN::data train_data = cuANN::data("abelone.train");
    // Train: Activation, Derivative, Data, Epochs, Reports, Threads, Stop Error, Learning Rate, Momentum
    auto mse = network.train(func,deriv,train_data,10000,10,8,0.02f,0.7f,0.1f);
    std::cout << "Trained Abelone with MSE: " << mse << std::endl;

    return 0;
}
