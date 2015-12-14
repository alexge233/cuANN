#ifndef _cuANN_trainer_HPP_
#define _cuANN_trainer_HPP_
#include "includes.ihh"
namespace cuANN
{
/// @author Alexander Giokas <a.gkiokas@warwick.ac.uk>
/// @date   November 2015
/// @version 1
/// 
/// The trainer is a worker object used by CPU threads.
/// Each input pattern is handled by a worker object within a (thread) `worker_pool`
///
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
               unsigned int index
            );

    // This is the method that the thread pool calls
    void operator()(std::vector<std::shared_ptr<pattern>> & patterns);
    
private:

    // Calculate Output Node Delta: `-E * σ'(Σ(O[i])`
    void output_node_delta(const std::shared_ptr<pattern> & ptr);

    // Prime the Input Node Sums using the activation derivative: `σ'(Σ[ji])` 
    void primed_sums(const std::shared_ptr<pattern> & ptr);

    // Calculate Node Delta[i] for hidden layers: `σ'( Σ[ji] ) * Σ( W[ik] * δ[k] )`
    void hidden_node_delta(const std::shared_ptr<pattern> & ptr);

    // Calculate Weight Gradient: `∂E/∂W[ik] = δ[k] * O[i]`
    void calc_weight_gradients(const std::shared_ptr<pattern> &ptr);

    // Calculate Squared Errors: Error = (Ideal - Actual)^2 for each Output node value 
    void calc_squared_errors(const std::shared_ptr<pattern> &ptr);

    // Activation and Derivative
    const A & _func;
    const D & _deriv;

    // Pattern index
    const unsigned int _i;
    // Current Pattern MSE
    float _mse = 0.f;
};
}
#include "trainer.hxx"
#endif
