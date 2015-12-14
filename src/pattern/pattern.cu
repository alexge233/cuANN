#include "pattern.hpp"

namespace cuANN
{
__host__ pattern::pattern ( 
                            thrust::host_vector<float> input,
                            thrust::host_vector<float> output,
                            const thrust::device_vector<float> & weights,
                            thrust::device_vector<float> & gradient_sums,
                            thrust::device_vector<float> & global_errors,
                            const std::vector<std::pair<int,int>> & weight_index,
                            unsigned int size_output,
                            unsigned int size_input,
                            unsigned int size_hidden,
                            unsigned int nodes_per_hidden_layer,
                            unsigned int pattern_index
                          )
: weight_ref(weights),
  epoch_gradients(gradient_sums),
  epoch_errors(global_errors),
  weight_idx_ref(weight_index),
  // delta size = hidden nodes + output nodes
  delta_size(size_hidden+size_output),
  output_size(size_output),
  input_size(size_input),
  hidden_size(size_hidden),
  n_per_hl(nodes_per_hidden_layer),
  index(pattern_index)
{
    ideal_input = thrust::device_vector<float>(input);
    ideal_output = thrust::device_vector<float>(output);

    node_sums = thrust::device_vector<float>(delta_size);
    node_deltas = thrust::device_vector<float>(delta_size);
    primed_sums = thrust::device_vector<float>(delta_size);
    node_outputs = thrust::device_vector<float>(input_size+hidden_size+output_size);
    gradients = thrust::device_vector<float>(weight_ref.size()); 

    // Max size for Layer Sums and Tmp Layer Output
    unsigned int max_size = input_size > n_per_hl ? input_size : n_per_hl;

    // Forward Propagation temporary Matrix (size of ???)
    // TODO: max_size * `weights of layer with max_size`
    fw_prop_mtx = thrust::device_vector<float>(weight_ref.size());
    // TODO: this is a copy of node_sums, no need to use can deprecate
    layer_sums = thrust::device_vector<float>(max_size);
    node_delta_mtx = thrust::device_vector<float>(weight_ref.size());
}

void pattern::zero_fill()
{
    thrust::fill(thrust::device,gradients.begin(),gradients.end(),0.f);
    thrust::fill(thrust::device,node_sums.begin(),node_sums.end(),0.f);
    thrust::fill(thrust::device,node_deltas.begin(),node_deltas.end(),0.f);
    thrust::fill(thrust::device,primed_sums.begin(),primed_sums.end(),0.f);
    thrust::fill(thrust::device,node_outputs.begin(),node_outputs.end(),0.f);
}

};
