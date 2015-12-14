#ifndef _cuANN_pattern_HPP_
#define _cuANN_pattern_HPP_
#include "includes.ihh"
namespace cuANN
{
/// @author Alexander Giokas <a.gkiokas@warwick.ac.uk>
/// @date   December 2015
/// @version 1
/// @brief The data needed by a worker object.
///
struct pattern
{
    /// 
    pattern ( 
                thrust::host_vector<float> input,
                thrust::host_vector<float> output,
                const thrust::device_vector<float> & weights,
                thrust::device_vector<float> & gradient_sums,
                thrust::device_vector<float> & global_errors,
                const std::vector<std::pair<int,int>> & weight_index,
                std::mutex & rw_mutex,
                unsigned int size_output,
                unsigned int size_input,
                unsigned int size_hidden,
                unsigned int nodes_per_hidden_layer,
                unsigned int pattern_index
             );

    /// Zero-Fill local arrays
    void zero_fill();


    /// Pattern Index
    const unsigned int index;

    /// Shared Weights
    const thrust::device_vector<float> & weight_ref;
    /// Shared Gradients Sums - will be updated
    thrust::device_vector<float> & epoch_gradients;
    /// Shared Squared Errors - will be updated
    thrust::device_vector<float> & epoch_errors;
    /// Reference to Shared Weight Index
    const std::vector<std::pair<int,int>> & weight_idx_ref;

    /// Mutex for Updating global gradient values
    std::mutex & update_mtx;

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


    /// Ideal Input
    thrust::device_vector<float> ideal_input;
    /// Ideal Outpput
    thrust::device_vector<float> ideal_output;

    /// Node Sums Input `Σ( O[j] ) - The input to a node from all connecting nodes
    thrust::device_vector<float> node_sums;
    /// Node Deltas `δ[i]` - For all nodes
    thrust::device_vector<float> node_deltas;
    /// Primed Sums `σ'( Σ( O[j] * W[ij] ) )` - Used for Node Delta
    thrust::device_vector<float> primed_sums;
    /// Node Outputs `Ο[i]` of all nodes in all layers - Used for Node Delta
    thrust::device_vector<float> node_outputs;
    /// Weight Gradients `∂E/∂W[ik]`
    thrust::device_vector<float> gradients;

    // Forward Propagation Matrix Result: `I[j] * W[ji]` - max size: `weights size`
    thrust::device_vector<float> fw_prop_mtx;
    // Layer Sums (max size of `n_per_hl`): `Σ( O[j] * W[ji] )`
    thrust::device_vector<float> layer_sums;
    // Hidden Node Delta Matrix Result: `W[ik]*δ[k]` - max size: `weights size`
    thrust::device_vector<float> node_delta_mtx;
};
}

/// hashing functor for a shared pointer to a pattern - based upon pattern's index
namespace std 
{
template<> struct hash<cuANN::pattern>
{
    size_t operator()(const std::shared_ptr<cuANN::pattern> & ptr) const
    {
        assert(ptr);
        return boost::hash_value(ptr->index);
    }
};
}
#endif
