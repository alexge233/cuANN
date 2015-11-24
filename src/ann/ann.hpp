#ifndef _cuANN_ann_HPP_
#define _cuANN_ann_HPP_
#include "includes.ihh"
namespace cuANN
{
#define MAX_THREADS 4
///
/// @class ann
/// @brief A simple Feedforward artificial neural network
/// @date November 2015
/// @author Alexander Giokas <a.gkiokas@warwick.ac.uk>
/// @version 2
///
/// TODO: ann must be (de)serialisable
/// TODO: I would like to template the activation function for the ann class.
///
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
     * @param train_data is used to train the network
     * @param mse_stop will stop training if that MSE is achieved
     * @param epochs denotes for how many epochs will the network be trained
     * @param reports defines the interval of epochs used to report on MSE on screen (use 0 for no reports)
     * @param online defines online training if TRUE, or batch training if FALSE
     */
    float train (
                  const cuANN::data & train_data,
                  const float mse_stop,
                  unsigned int epochs,
                  unsigned int reports
                );
    
    /**
     * @brief Propagate the input through the network, and get an output
     * @return The Output array `O[k]` from the Output layer and nodes.
     */
    h_vector propagate ( thrust::device_vector<float> & input ) const;

protected:

    /**
     * @brief This is a Training Epoch
     * @param dataset is the training data set used to train the network.
     * @param thread_pool is the trainer worker pool of parallel threads.
     * @param thread_data is a vector of totally allocated thread data objects.
     * @param gradients is the array of `∂E/∂W[ik]`
     * @param updates is the array of `Δw(t-1)`
     * @param errors is the array of squared errors  
     *
     * @return MSE: Mean-Square Error
     * 
     * Calculate the Delta Rule: `σ'( Σ[ji] ) * Σ( W[ik] * δ[k] )`
     * Using Sigmoid Prime σ'(x) = σ(x) * ( 1 - σ(x) )
     * Then get the Gradient of each Weight: `∂E / ∂W[ik]`
     * Finally, use Back-Propagation (Either Online or Batch)
     */
    float epoch ( 
                    const cuANN::data & dataset,
                    cuANN::trainer_pool & thread_pool,
                    thrust::device_vector<float> & gradients,
                    thrust::device_vector<float> & updates,
                    thrust::device_vector<float> & errors
                );

    /// @brief Propagate input via single layer: `O[j] * W[i]`
    /// @param weights_begin is the weights[index] start range
    /// @param weights_end is the weights[index] end range
    /// @param input is the actual input vector (device mem)
    /// @return the Sum: `Σ( O[j] * W[i]`
    /// @warning the return vector is NOT sigmoid activated!
    d_vector prop_layer (
                          unsigned int weights_begin,
                          unsigned int weights_end,
                          const thrust::device_vector<float> & input
                        ) const;

private:

    unsigned int input_neurons_;
    unsigned int hidden_neurons_;
    unsigned int output_neurons_;
    unsigned int hidden_layers_;
    unsigned int per_layer_;
    float alpha_;
    float epsilon_;

    /// @note This is a vectorised Matrix!
    /// @note This vector contains all weights.
    ///       They are separated in blocks of `layers_`, e.g.: 
    ///         - First hidden vector weights will be from [0]-[input*per_layer_]_,
    ///         - Second hidden vector weights will be from [per_layer_] - [2*per_layer_]
    ///       They are further partitions as [Node(x),Weight(x1,x2,...xN) within a layer
    /// @see w_index_ which indexes the weight range per layer.
    thrust::device_vector<float> weights_;

    /// Index tracks of where Weights begin and end (per layer increments) for fully connected network
    std::vector<std::pair<int,int>> w_index_;
};
}
#endif
