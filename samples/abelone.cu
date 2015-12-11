#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>

int main ()
{
    // Activation functor and its derivative
    // The `Abelone` data-set is real-valued and scaled between 0 and 1.
    // Because the network we create below is a deep network, we need to use soft_sign (or ramp)
    cuANN::soft_sign func;
    cuANN::soft_sign_deriv deriv;

    // Create the Neural Network
    // It must match the input size of your data (in this case, have a look at `data/abelone.train`
    // Input nodes: 10 real-valued nodes [0,1] scaled
    // Hidden nodes: 30
    // Hidden layers: 3
    // Output nodes: 1
    //
    // In order to decide on the amount of hidden neurons, use the upper-bound rule:
    //             `# of hidden nodes = (# of training samples) / (alpha * (# of input nodes + # of output nodes)`
    //
    // where alpha = [5,10]
    // In this case, our input is 10, output is 1, samples are 2088, and we use a medium alpha = 7
    // Hence hidden nodes = 27. 
    // Because 27 nodes are too many to fit into one hidden layer (the network's hidden layer will be massive), 
    // we will spread them over 3 hidden layers and round them up to 30.
    // This in effect creates a "deep network" which as a requirement needs the `soft_sign` activation function.
    //
    // REMEMBER: There is no guarantee that the network will learn - it depends on weight initialisation
    //           and the training algorithm, as well as the "representative power" of the training data.
    //
    cuANN::ann net = cuANN::ann(10,30,3,1);

    // Train: Load from File
    cuANN::data train_data = cuANN::data("../data/abelone.train");

    // When training pass as the first two params the activation functor and it's derivative.
    // Then the training data, the Epochs for which the network will be trained,
    // the amount of CPU threads (each CPU thread "learns" a pattern)
    // the stop-error, e.g., when should the network stop learning
    // the learning rate, and the momentum rate.
    auto mse = net.train(func,deriv,train_data,50000,100,4,.02,.2,.5);

    std::cout << "Abelone deep network using soft_sign back-prop MSE: " << mse << std::endl;

    cuANN::data test_data = cuANN::data("../data/abelone.test");
    mse = net.test(func,test_data);
    std::cout << "abelone network test MSE: " << mse << std::endl;
   
    // Save on disk
    std::cout << "saving abelone network on disk" << std::endl;
    std::ofstream ofs("abelone.net");
    boost::archive::text_oarchive oa(ofs);
    oa << net;

    return 0;
}
