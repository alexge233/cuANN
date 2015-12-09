#ifndef _cuANN_trainer_HPP_
#define _cuANN_trainer_HPP_
#include "includes.ihh"
#include "trainer_data.hpp"
namespace cuANN
{
/// @author Alexander Giokas <a.gkiokas@warwick.ac.uk>
/// @date   November 2015
/// @version 1
/// 
/// The trainer is a worker object used by CPU threads.
/// Each input pattern is handled by a worker object within a (thread) `worker_pool`
/// The `worker_data` is a pre-allocated device array wrapper, which aims at avoiding
/// continuous allocations and re-allocations on the device memory.
template <class A, class D>
class trainer
{
public:

    /// @brief run an Epoch worker for one input pattern
    /// @param trainer_data is a unique per `trainer` collection of device arrays already allocated
    /// @param input is the pattern to be learnt
    /// @param output is the Target/Ideal output
    /// @param alpha is the momentum for back propagation
    /// @param epsilon is the learning rate for back propagation
    /// @param index is the index of input in the training data (used for squared error updating)
    /// @note: This worker class only performs Batch training (no online Training)
    trainer (
               A const& func,
               D const& deriv,
               const cuANN::row & pattern,
               unsigned int index
            );

    // This is the method that the thread pool calls
    void operator()(std::vector<std::shared_ptr<trainer_data>> & thread_data);
    
private:

    /// @brief Activate Input and forward-propagate: `Σ( O[j] * W[ji] )`
    /// @param ptr is the `trainer data` object we use for device memory calculations
    /// @note Store all node output & all node input sums    
    void fw_propagate(const std::shared_ptr<trainer_data> & ptr);

    // Sum Input * Weight Propagation: `Σ( O[j] * W[ji] )` for a layer
    thrust::device_vector<float> layer_product( 
                                                 unsigned int weights_begin,
                                                 unsigned int weights_end,
                                                 const thrust::device_vector<float> & input,
                                                 const std::shared_ptr<trainer_data> & ptr
                                              );

    // Calculate Output Node Delta: `-E * σ'(Σ(O[i])`
    void output_node_delta(const std::shared_ptr<trainer_data> & ptr);

    // Prime the Input Node Sums using the activation derivative: `σ'(Σ[ji])` 
    void primed_sums(const std::shared_ptr<trainer_data> & ptr);

    // Calculate Node Delta[i] for hidden layers: `σ'( Σ[ji] ) * Σ( W[ik] * δ[k] )`
    void hidden_node_delta(const std::shared_ptr<trainer_data> & ptr);

    // Calculate Weight Gradient: `∂E/∂W[ik] = δ[k] * O[i]`
    void calc_weight_gradients(const std::shared_ptr<trainer_data> &ptr);

    // Calculate Squared Errors: Error = (Ideal - Actual)^2 for each Output node value 
    void calc_squared_errors(const std::shared_ptr<trainer_data> &ptr);

    // Activation and Derivative
    const A & _func;
    const D & _deriv;

    // Pattern index
    const unsigned int _i;

    // Input and Output Patterns (device memory of cuANN::data::row)
    thrust::device_vector<float> ideal_input;
    thrust::device_vector<float> ideal_output;
};
}
#include "trainer.hxx"
#endif
