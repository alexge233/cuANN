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
     *
     */
    ann ( 
            const unsigned int input_neurons,
            const unsigned int hidden_neurons,
            const unsigned int hidden_layers,
            const unsigned int output_neurons
        );

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
     * @return a vector of output the size of ann.output_neurons
     */
    thrust::host_vector<float> test ( thrust::host_vector<float> test_input );

private:

    /// Sigmoid Activation Function
    __device__ __host__ float sigmoid__ ( const float x ) const;

    /// Fast Sigmoid Activation Function
    __device__ __host__ float fast_sigmoid__ ( const float x ) const;

    /// Pseudo-Random Number Generator 
    __device__ __host__ float prng__ (
                                        const float min,
                                        const float max
                                     ) const;



    unsigned int layers_num__;
    unsigned int input_neurons__;
    unsigned int hidden_neurons__;
    unsigned int output_neurons__;
    float learning_rate__;

    /// Network Layers
    thrust::device_vector<float> layers;

    // Our Weights
    //     ...
    //     TODO...
    //     Bias Neurons
    //     ...
};
}
#endif
