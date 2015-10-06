#ifndef _cuANN_ann_HPP_
#define _cuANN_ann_HPP_
#include "Includes.hxx"

namespace cuANN
{
/**
 * @class ann
 * @brief A simple Feedforward artificial neural network
 * @date 6th October 2015
 * @author Alex Giokas <alexge233@hotmail.com>
 * @version 1
 */
template <class NumType> class ann : public network
{
public:

    ann ( 
            const unsigned int input_neurons,
            const unsigned int hidden_neurons,
            const unsigned int hidden_layers,
            const unsigned int output_neurons
        );

    /**
     * @param train_set contains the training data
     * @param cross_set contains another sample of data used for cross-validation (Early Stopping)
     */
    NumType train ( 
                    const cuANN::train_data & train_set,
                    const cuANN::validation_data & cross_set
                  );

    /**
     * @param test_input will propagate the input and produce output
     * @return a vector of output the size of ann.output_neurons
     * TODO: maybe pass Activation Function as parameter?
     */
    thrust::host_vector<NumType> test ( const cuANN::test_data & test_input );

private:

    /// Activation Function
    NumType sigmoid ( const NumType input );


    unsigned int layers_num__;

    unsigned int input_neurons__;

    unsigned int hidden_neurons__;

    unsigned int output_neurons__;

    NumType learning_rate__;

    /// Network Layers
    thrust::device_vector<NumType> layers;

    // Our Weights
    //     ...
    //     TODO...
    //     Bias Neurons
    //     ...
};

//NOTE: Because this is a template class, and because the implementation is in a cubin
//      We have to include the cubin FROM the HPP header, and not the other way around
#include "ann.cu"

}
#endif
