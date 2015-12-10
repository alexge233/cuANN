#ifndef _cuANN_ann_HPP_
#define _cuANN_ann_HPP_
#include "includes.ihh"
namespace cuANN
{
/// @class ann
/// @brief A simple Feedforward artificial neural network
/// @date November 2015
/// @author Alexander Giokas <a.gkiokas@warwick.ac.uk>
/// @version 2
class ann
{
public:

    /// @brief Empty Constructor
    /// @warning only to be used in order to de-serialize (load) an object from disk
    ann () = default;

    /// @brief Construct a new ANN
    /// @param input_neurons must match your input data
    /// @param hidden_layers will be used to calculate hidden neurons per layer
    /// @param output_neurons must match your training data
    ann (
            unsigned int input_neurons,
            unsigned int hidden_neurons,
            unsigned int hidden_layers,
            unsigned int output_neurons
        );

    /// Print on stdout all weight values
    void print_weights()const;

    /// @brief Train the Network using the training data
    /// @param func is the activation functor
    /// @param deriv is the activation derivative functor
    /// @param train_data is used to train the network
    /// @param stop_error will stop training if that MSE is achieved
    /// @param epochs denotes for how many epochs will the network be trained
    /// @param reports defines the interval of epochs used to report on MSE on screen (use 0 for no reports)
    /// @param learning sets the Back-propagation Learning rate
    /// @param momentum sets the Back-Propagation Momentum
    template <class A,class D>
    float train (
                  A const& func,
                  D const& deriv,
                  cuANN::data & train_data,
                  unsigned int epochs,
                  unsigned int reports,
                  unsigned int max_threads,
                  float stop_error,
                  float learning,
                  float momentum
                );

    /// @brief Test the Network for Accuracy/Performance
    /// @template class A is the activation functor
    /// @param func is the activation functor instance
    /// @param test_data is the test data-set which will be used to test the network
    /// @return Mean-Square-Error
    template <class A>
    float test (
                   A const& func,
                   const cuANN::data & test_data
               ) const;

    /// @brief Propagate the input through the network, and get an output
    /// @note template class A defines the activation function type
    /// @return The Output array `O[k]` from the Output layer and nodes.
    template <class A>
    thrust::device_vector<float> propagate ( 
                                              A const& func,
                                              thrust::device_vector<float> & input 
                                           ) const;
protected:

    /// @brief This is a Back-Propagation Batch Training Epoch
    /// @param dataset is the training data set used to train the network.
    /// @param thread_pool is the trainer worker pool of parallel threads.
    /// @param thread_data is a vector of totally allocated thread data objects.
    /// @param gradients is the array of `∂E/∂W[ik]`
    /// @param updates is the array of `Δw(t-1)`
    /// @param errors is the array of squared errors  
    /// @return MSE: Mean-Square Error
    /// Calculate the Delta Rule: `σ'( Σ[ji] ) * Σ( W[ik] * δ[k] )`
    /// Using Sigmoid Prime σ'(x) = σ(x) * ( 1 - σ(x) )
    /// Then get the Gradient of each Weight: `∂E/∂W[ik]`
    /// Finally, use Back-Propagation (Batch Training)
    template <class A,class D>
    float epoch (
                    A const& func,
                    D const& deriv,
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
    /// @warning the return vector is NOT activated!
    thrust::device_vector<float> prop_layer (
                                              unsigned int weights_begin,
                                              unsigned int weights_end,
                                              const thrust::device_vector<float> & input
                                            ) const;


    friend class boost::serialization::access; 

    /// Serialize method
    template<class Archive> 
    void save(Archive & ar, const unsigned int) const;

    /// Deserialize method
    template<class Archive>
    void load(Archive & ar, const unsigned int);

    /// Split load/save methods
    BOOST_SERIALIZATION_SPLIT_MEMBER()
    
    /// Various Network Settings
    unsigned int input_neurons_;
    unsigned int hidden_neurons_;
    unsigned int output_neurons_;
    unsigned int hidden_layers_;
    unsigned int per_layer_;

    /// Back-Prop Learning and Momentum
    float alpha_;
    float epsilon_;

    /// Weight matrix for all nodes
    thrust::device_vector<float> weights_;

    /// Index tracks of where Weights begin and end (per layer increments)
    std::vector<std::pair<int,int>> w_index_;
};
}
// Include our template implementation header
#include "ann.hxx"
#endif
