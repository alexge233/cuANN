#include "trainer_data.hpp"

namespace cuANN
{

__host__ trainer_data::trainer_data ( 
                                        const thrust::device_vector<float> & weights,
                                        thrust::device_vector<float> & gradient_sums,
                                        thrust::device_vector<float> & global_errors,
                                        const std::vector<std::pair<int,int>> & weight_index,
                                        std::mutex & gradient_mutex,
                                        unsigned int size_output,
                                        unsigned int size_input,
                                        unsigned int size_hidden,
                                        unsigned int nodes_per_hidden_layer
                                  )
: weight_ref(weights),
  grad_sums_ref(gradients),
  glob_errors_ref(global_errors),
  weight_idx_ref(weight_index),
  grad_sums_mtx(gradient_mutex),
  // delta size = hidden nodes + output nodes
  delta_size(size_hidden+size_output),
  output_size(size_output),
  input_size(size_input),
  hidden_size(size_hidden),
  n_per_hl(nodes_per_hidden_layer)
{
    // Node Input Sums - Hidden & Output (Not Input)
    node_sums = thrust::device_vector<float>(delta_size);

    // Node deltas - Hidden & Output (Not Input)
    node_deltas = thrust::device_vector<float>(delta_size);

    // Used by Node Delta = Node Outputs = #of Node Deltas
    primed_sums = thrust::device_vector<float>(delta_size);

    // Used for Node Delta - Input & Hidden & Output (node count)
    nodes_output = thrust::device_vector<float>(input_size+hidden_size+output_size);

    // Ideal/Target Output array
    ideal_out = thrust::device_vector<float>(output_size);

    // Input Array
    input = thrust::device_vector<float>(input_size);

    // Squared Errors
    sq_errors = thrust::device_vector<float>(output_size);
}
};
