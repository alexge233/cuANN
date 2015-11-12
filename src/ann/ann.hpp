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
     * @brief Train the Network using the training data
     * @param mse_stop will stop training if that MSE is achieved
     * @param epochs denotes for how many epochs will the network be trained
     * @param learning denotes the learning rate of the Gradient Descent
     * @param momentum is the momentum used to affect previous Gradients
     */
    float train (
                  const cuANN::data & train_data,
                  float mse_stop,
                  unsigned int epochs
                );
    
    /**
     * @brief Propagate the input through the network, and get an output
     * @return a vector of output the size of ann.output_neurons
     */
    h_vector propagate ( d_vector input ) const;

private:

    /**
     * @brief This is a Training Epoch
     * @return Mean-Squared Error
     * @param input is a continous memory of many input vectors,
     * which are separated at interval `input_len`.
     * @param output is also a continous memory of many output vectors,
     * also separated at interval `output_len`
     */
    float epoch ( 
                    d_vector & input,
                    unsigned int input_len,
                    h_vector & output,
                    unsigned int output_len,
                    unsigned int total
                );


    /// Propagate input via single layer
    /// @param weights_begin is the weights[index] start range
    /// @param weights_end is the weights[index] end range
    /// @param input is the actual input vector (device mem)
    /// @return the Sum of (Input * Weight)
    /// @warning the return vector is NOT sigmoid activated!
    d_vector prop_layer (
                          unsigned int weights_begin,
                          unsigned int weights_end,
                          unsigned int output_size,
                          const d_vector & input
                        ) const;

    /// Calculate Output's Squared Errors
    /// @param ideal output will be compared to @param actual and error is squared
    d_vector output_errors (
                               d_vector ideal,
                               d_vector actual
                            ) const;


    // ANN Private Vars
    unsigned int input_neurons_;
    unsigned int hidden_neurons_;
    unsigned int output_neurons_;
    unsigned int hidden_layers_;
    unsigned int per_layer_;

    /// WARNING: This is a vectorised Matrix!
    ///                - This vector will contain ALL weights
    ///                - in blocks of `layers_`, e.g.: 
    ///                -    first hidden vector weights will be from [0]-[per_layer_]_,
    ///                -    second hidden vector weights will be from [per_layer_] - [2 * per_layer_]
    ///                -    Furthermore, within a layer, they go as: [H1W2],[H2W2],etc.
    thrust::device_vector<float> weights_;

    /// This index keeps track of where Weights begin and end (per layer increments)
    std::vector<std::pair<int,int>> w_index_;

};
}
#endif
