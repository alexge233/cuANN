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
  epoch_gradients(gradient_sums),
  epoch_errors(global_errors),
  weight_idx_ref(weight_index),
  grad_sums_mtx(gradient_mutex),
  // delta size = hidden nodes + output nodes
  delta_size(size_hidden+size_output),
  output_size(size_output),
  input_size(size_input),
  hidden_size(size_hidden),
  n_per_hl(nodes_per_hidden_layer)
{
    node_sums = thrust::device_vector<float>(delta_size);
    node_deltas = thrust::device_vector<float>(delta_size);
    primed_sums = thrust::device_vector<float>(delta_size);
    node_outputs = thrust::device_vector<float>(input_size+hidden_size+output_size);
    actual_output = thrust::device_vector<float>(output_size);
    gradients = thrust::device_vector<float>(weight_ref.size()); 

    // Max size for Layer Sums and Tmp Layer Output
    unsigned int max_size = input_size > n_per_hl ? input_size : n_per_hl;

    fw_prop_mtx = thrust::device_vector<float>(weight_ref.size());
    layer_sums = thrust::device_vector<float>(max_size);
    node_delta_mtx = thrust::device_vector<float>(weight_ref.size());
}
};
