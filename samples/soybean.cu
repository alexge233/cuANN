#include "../src/ann/ann.hpp"
#include "../src/kernel/kernel.hpp"
#include <iostream>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>

int main ()
{
    // Instantiate the activation functor and its derivative
    // The `soybean.train` data is a mix of real-valued and binary inputs, thus we use Tanh (scaled)
    // and it's derivative
    cuANN::tanh_scaled func;
    cuANN::tanh_scaled_deriv deriv;

    // Create the Neural Network
    // It must match the input size of your data (in this case, have a look at `data/soybean.train`
    // Input nodes: 82
    // Hidden nodes: 0
    // Hidden layers: 0
    // Output nodes: 19
    //
    // In order to decide on the amount of hidden neurons, use the upper-bound rule:
    //             `# of hidden nodes = (# of training samples) / (alpha * (# of input nodes + # of output nodes)`
    //
    // where alpha = [5,10]
    // In this case, our input is 82, output is 19, samples are 342, and we use alpha = 6
    // Hence no hidden nodes (or layers).
    //
    // REMEMBER: There is no guarantee that the network will learn - it depends on weight initialisation
    //           and the training algorithm, as well as the "representative power" of the training data.
    //
    cuANN::ann net = cuANN::ann(82,0,0,19);

    // load the training data (342 samples)
    cuANN::data train_data = cuANN::data("../data/soybean.train");

    // When training pass as the first two params the activation functor and it's derivative.
    // Then the training data, the Epochs for which the network will be trained,
    // the amount of CPU threads (each CPU thread "learns" a pattern)
    // the stop-error, e.g., when should the network stop learning
    // the learning rate, and the momentum rate.
    float mse = net.train(func,deriv,train_data,100000,100,4,.002,.3,.6);

    // Print back-prop MSE
    std::cout << "soybean net using tanhh_norm back-prop MSE: " << mse << std::endl;

    cuANN::data test_data = cuANN::data("../data/soybean.test");
    mse = net.test(func,test_data);
    std::cout << "soybean network test MSE: " << mse << std::endl;
   
    // Save on disk
    std::cout << "saving network on disk" << std::endl;
    std::ofstream ofs("soybean.net");
    boost::archive::text_oarchive oa(ofs);
    oa << net;
    
    return 0;
}
