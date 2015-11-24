#ifndef _cuANN_trainer_data_HPP_
#define _cuANN_trainer_data_HPP_
#include "includes.ihh"
namespace cuANN
{
/// @author Alexander Giokas <a.gkiokas@warwick.ac.uk>
/// @date   November 2015
/// @version 1
/// @brief The data needed by a worker object.
/// @note global squared errors are indexed by `input pattern` * `output size`
///       As such, we don't need to lock access, as each `trainer` is unique per `input pattern`
///
/// @warning Network weights are shared across multiple threads.
///          We do not update them, only read their values, therefore no lock access is needed.
///
/// @warning Gradient Sums are shared across multiple threads.
///          We DO NEED to lock access, as we will update them by summing them.
struct trainer_data
{
    /// 
    trainer_data ( 
                    const thrust::device_vector<float> & weights,
                    thrust::device_vector<float> & gradient_sums,
                    thrust::device_vector<float> & global_errors,
                    const std::vector<std::pair<int,int>> & weight_index,
                    std::mutex & gradient_mutex,
                    unsigned int size_output,
                    unsigned int size_input,
                    unsigned int size_hidden,
                    unsigned int nodes_per_hidden_layer
                 );

    /// Shared Weights
    const thrust::device_vector<float> & weight_ref;
    /// Shared Gradients Sums
    thrust::device_vector<float> & grad_sums_ref;
    /// Shared Squared Errors
    thrust::device_vector<float> & glob_errors_ref;

    /// Reference to Shared Weight Index
    const std::vector<std::pair<int,int>> & weight_idx_ref;

    /// Mutex for Read-Write gradient values
    std::mutex & grad_sums_mtx;
    /// Mutex for worker_data
    std::mutex available;

    /// Amount of Delta Nodes
    const unsigned int delta_size;
    /// Output Nodes
    const unsigned int output_size;
    /// Input Nodes
    const unsigned int input_size;
    /// Hidden Nodes
    const unsigned int hidden_size;
    /// Hidden Nodes per Hidden Layer
    const unsigned int n_per_hl;

    /// Node Sums Input `Σ( O[j] ) - The input to a node from all connecting nodes
    thrust::device_vector<float> node_sums;
    /// Node Deltas `δ[i]` - For all nodes
    thrust::device_vector<float> node_deltas;
    /// Primed Sums `σ'( Σ( O[j] * W[ij] ) )` - Used for Node Delta
    thrust::device_vector<float> primed_sums;
    /// Node Outputs `Ο[i]` of all nodes in all layers - Used for Node Delta
    thrust::device_vector<float> nodes_output;
    /// Weight Gradients `∂E/∂W[ik]`
    thrust::device_vector<float> gradients;
    /// Ideal/Target Output
    thrust::device_vector<float> ideal_out;
    /// Input values / Input Pattern
    thrust::device_vector<float> input;
    /// Squared Errors for a single Input Pattern
    thrust::device_vector<float> sq_errors;
};
}
#endif
