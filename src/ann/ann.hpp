#ifndef _cuANN_ann_HPP_
#define _cuANN_ann_HPP_
#include "includes.ihh"
namespace cuANN
{
/**
 * @class ann
 * @brief A simple Feedforward artificial neural network
 * @date 6th October 2015
 * @author Alex Giokas <alexge233@hotmail.com>
 * @version 1
 */
class ann
{
public:

    /**
     * Construct a new ANN
     * @param input_neurons must match your input data
     */
    ann ( );

    /**
     * @brief This is a Training Epoch (a full iteration of the data)
     * @note  Uses the Batch Training MSE, not incremental
     * @return Mean-Squared Error
     */
    float epoch (
                    const cuANN::data & input,
                    const float stop_error,
                    const float alpha
                );

    /**
     * @brief Propagate the input through the network, and get an output
     * @return a vector of output the size of ann.output_neurons
     */
    thrust::host_vector<float> prop ( thrust::host_vector<float> test_input );

private:

    /// Sigmoid Activation Function: TODO Move to another struct
    __device__ __host__ float sigmoid__ ( const float x ) const;

    /// Fast Sigmoid Activation Function: TODO Move to another struct
    __device__ __host__ float fast_sigmoid__ ( const float x ) const;



    /// Setup the Network here
    //  TODO: 0.2 version will be parametrised
    unsigned int input_neurons__ = 2;
    unsigned int hidden_neurons__ = 2;
    unsigned int output_neurons__ = 1;
    float learning_rate__;

    /// Input Layer
    thrust::device_vector<float> input__;

    /// Hidden Layer
    thrust::device_vector<float> hidden__;

    // NOTE: If using many hidden layers, or if we parametrise
    // std::vector<thrust::device_vector<float>> hidden__;

    /// Output Layer
    thrust::device_vector<float> output__;

    // Bias Neurons ?
};
}
#endif
