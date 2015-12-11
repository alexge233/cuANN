#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>

int main ()
{
    // Instantiate the activation functor and its derivative
    // The `thyroid.train` data is real-valued and scaled between 0 and 1.
    // However, because we use two hidden layers, we have to employ soft-sign
    // and it's derivative
    cuANN::soft_sign func;
    cuANN::soft_sign_deriv deriv;

    // Create the Neural Network
    // It must match the input size of your data (in this case, have a look at `data/thyroid.train`
    // Input nodes: 21
    // Hidden nodes: 18
    // Hidden layers: 2
    // Output nodes: 3
    //
    // In order to decide on the amount of hidden neurons, use the upper-bound rule:
    //             `# of hidden nodes = (# of training samples) / (alpha * (# of input nodes + # of output nodes)`
    //
    // where alpha = [5,10]
    // In this case, our input is 21, output is 3, samples are 3600, and we use alpha = 8
    // Hence we'll use 18 hidden nodes. Those will be within two hidden layers.
    //
    // REMEMBER: There is no guarantee that the network will learn - it depends on weight initialisation
    //           and the training algorithm, as well as the "representative power" of the training data.
    //
    cuANN::ann net = cuANN::ann(82,0,0,19);

    // load the training data (3600 samples)
    cuANN::data train_data = cuANN::data("../data/thyroid.train");

    // When training pass as the first two params the activation functor and it's derivative.
    // Then the training data, the Epochs for which the network will be trained,
    // the amount of CPU threads (each CPU thread "learns" a pattern)
    // the stop-error, e.g., when should the network stop learning
    // the learning rate, and the momentum rate.
    float mse = net.train(func,deriv,train_data,100000,100,4,.002,.2,.7);

    // Print back-prop MSE
    std::cout << "thyroid net using tanhh_norm back-prop MSE: " << mse << std::endl;

    cuANN::data test_data = cuANN::data("../data/thyroid.test");
    mse = net.test(func,test_data);
    std::cout << "thyroid network test MSE: " << mse << std::endl;
   
    // Save on disk
    std::cout << "saving network on disk" << std::endl;
    std::ofstream ofs("thyroid.net");
    boost::archive::text_oarchive oa(ofs);
    oa << net;
    
    return 0;
}
