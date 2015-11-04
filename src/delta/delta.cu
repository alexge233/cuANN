#include "delta.hpp"

namespace cuANN
{

__global__ void delta_output_kernel (
                                        const float * w_sum,
                                        const float * out_err,
                                        float * delta,
                                        unsigned int size
                                    )
{
    // TODO: run a 1D grid, calculating: -E * f'( S(W*I) )
    //       where -E = -1.f * out_err
    // 
    //       float E = -1.f * out_err[x];
    //       delta[x] = E * sigmoid_prime( w_sum[x] )
}

__global__ void delta_hidden_kernel (
                                       const float * w_sum,
                                       unsigned int w_sum_size,
                                       const float * delta_k,
                                       unsigned int delta_k_size,
                                       float * delta_i,
                                       unsigned int delta_i_size,
                                       const float * weight_i,
                                       unsigned int weight_i_size
                                     )
{
    // TODO: think how this can be done in a single kernel
    //       because Im afraid it may require two kernels
}



};
