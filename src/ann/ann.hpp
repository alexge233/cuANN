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
 *
 * TODO: ann must be (de)serialisable
 */
class ann
{
public:

    typedef thrust::host_vector<float> h_vector;
    typedef thrust::device_vector<float> d_vector;

    /**
     * @brief Construct a new ANN
     * @param input_neurons must match your input data
     * @param hidden_layers will be used to calculate hidden neurons per layer
     * @param output_neurons must match your training data
     */
    ann ( 
            unsigned int input_neurons,
            unsigned int hidden_neurons,
            unsigned int hidden_layers,
            unsigned int output_neurons
        );

    /**
     * @brief This is a Training Epoch (an iteration of the data)
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
    h_vector propagate ( d_vector input ) const;

private:

    /// Propagate input through a single layer
    /// @param activaction_func may be a sigmoid, tahn, etc.
    d_vector prop_layer (
                          d_vector weights,
                          d_vector input,
                          std::function<float(float)> activation_func
                        ) const;

    // TODO: Calculate MSE / RMSE and create all training methods needed
    //

    unsigned int input_neurons_;
    unsigned int hidden_neurons_;
    unsigned int output_neurons_;
    unsigned int hidden_layers_;
    unsigned int per_layer_;

    float learning_rate_;

    /// Input Weights
    thrust::device_vector<float> weights_input_;

    /// Hidden Weights - WARNING: This is a vectorised Matrix!
    ///                - If more than one hidden layer is given
    ///                - This vector will contain ALL hidden weights
    ///                - in blocks of `layers_`, e.g.: 
    ///                -    first hidden vector weights will be from [0]-[per_layer_]_,
    ///                -    second hidden vector weights will be from [per_layer_] - [2 * per_layer_]
    ///                -    Furthermore, within a layer, they go as: [H1W2],[H2W2],etc.
    thrust::device_vector<float> weights_hidden_;

};
}
#endif
