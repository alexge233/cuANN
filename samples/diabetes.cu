#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>

int main ()
{
    // Instantiate the activation functor and its derivative
    cuANN::tanh_norm func;
    cuANN::tanh_norm_deriv deriv;

    // Create the Neural Network
    // It must match the input size of the training data (see `data/diabetes.train`) and the output size.
    // Input nodes: 8
    // Hidden nodes: 8
    // Hidden layers: 1
    // Output nodes: 2
    //
    // In order to decide on the amount of hidden neurons, use the upper-bound rule:
    //             `# of hidden nodes = (# of training samples) / (alpha * (# of input nodes + # of output nodes)`
    //
    // where alpha = [5,10]
    // Due to the fact that the upper-bound rule for hidden nodes provides a very small number:
    // 384 / ( 5 * (8+2) ) = 7.68, we use one single hidden layer with 8 hidden nodes.
    //
    // Remeber: more than one hidden layers, and the network starts to become "deep"
    //
    cuANN::ann net = cuANN::ann(8,6,1,2);

    // load the training data (384 samples)
    cuANN::data train_data = cuANN::data("../data/diabetes.train");

    // When training pass as the first two params the activation functor and it's derivative.
    // Then the training data, the Epochs for which the network will be trained,
    // the amount of CPU threads (each CPU thread "learns" a pattern)
    // the stop-error, e.g., when should the network stop learning
    // the learning rate, and the momentum rate.
    float mse = net.train(func,deriv,train_data,500000,100,1,.002,.7,.2);

    // Print MSE
    std::cout << "diabetes network trained MSE: " << mse << std::endl;

    // Test the Network with the "test" data-set
    cuANN::data test_data = cuANN::data("../data/diabetes.test");
    std::cout << "diabetes network test MSE: " << net.test(func,test_data) << std::endl;
   
    // Save on disk
    std::cout << "saving diabetes network on disk" << std::endl;
    std::ofstream ofs("diabetes.net");
    boost::archive::text_oarchive oa(ofs);
    oa << net;
    
    return 0;
}
