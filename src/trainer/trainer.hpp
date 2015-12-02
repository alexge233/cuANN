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
               const std::shared_ptr<cuANN::trainer_data> trainer_data,
               const thrust::host_vector<float> & input,
               const thrust::host_vector<float> & output,
               float alpha,
               float epsilon,
               unsigned int index
            );

    // This is the method that the thread pool calls
    void operator()( std::vector<std::shared_ptr<trainer_data>> & thread_data ) const;
    
private:

    // TODO: Forward prop, store sums, ouputs and final actual out
    void forward_prop( );

    // TODO: Delta[k] last layer (output) delta
    void delta_output( );

    // TODO: Delta[i] for all hidden layers
    void delta_hidden( );

    // TODO: Gradient calculation: update global shared gradients array
    void grad_calc( );

    // TODO: Error calculation: update global shared errors array
    void error_calc( );

    A & _func;
    D & _deriv;
    const float _a;
    const float _e;
    const unsigned int _i;
    const std::shared_ptr<cuANN::trainer_data> _dmem;
};
}
#include "trainer.hxx"
#endif
